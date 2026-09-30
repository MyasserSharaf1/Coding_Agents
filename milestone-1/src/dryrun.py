"""Execute the v2 notebook end to end with a mocked Ollama, a fake NVML meter and
local copies of the benchmark data (HF is unreachable here)."""
import json, gzip, os, sys, types, random, hashlib, base64, zlib, pickle, re, time, traceback
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
NB = json.load(open(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "m1-repair-vs-baseline-v2.ipynb")))
OUT = os.path.join(HERE, "dry_out")
N_EACH = int(os.environ.get("N_EACH", "30"))

# ---------------- local benchmark data in HF shape ---------------------------
he = [json.loads(l) for l in gzip.open("/tmp/claude-0/he.jsonl.gz", "rt")]
mb = [json.loads(l) for l in gzip.open("/tmp/claude-0/mbpp.jsonl.gz", "rt")]

class FakeDS(list):
    @property
    def column_names(self):
        return list(self[0].keys())

def mbpp_rows():
    rows = []
    for r in mb:
        asserts = [l for l in r["assertion"].strip().splitlines() if l.startswith("assert")]
        text = r["prompt"].strip().strip('"').strip().split("\n")[0]
        rows.append({"task_id": int(r["task_id"].split("/")[1]), "code": r["canonical_solution"],
                     "prompt": text, "test_list": asserts[:3], "test_imports": [],
                     "test": r["assertion"]})       # stand-in for the EvalPlus suite
    return FakeDS(rows)

def he_rows():
    return FakeDS([{k: r[k] for k in ["task_id", "prompt", "canonical_solution", "entry_point", "test"]}
                   for r in he])

def fake_load_dataset(name, *a, **k):
    if "mbpp" in name:
        return {"test": mbpp_rows()}
    if "human" in name.lower():
        return {"test": he_rows()}
    raise RuntimeError("no such dataset " + name)

# ---------------- synthetic LiveCodeBench -------------------------------------
LCB = []
REF = {}          # problem text -> reference code (for the mock model)
def enc(obj):
    return base64.b64encode(zlib.compress(pickle.dumps(json.dumps(obj)))).decode()

for i in range(10):
    date = "2025-%02d-%02dT00:00:00" % (1 + i % 9, 1 + i)
    if i % 2 == 0:
        q = "Given integers a and b on one line (case %d), print a*b+%d." % (i, i)
        pub = [{"input": "2 3\n", "output": "%d\n" % (6 + i), "testtype": "stdin"}]
        priv = [{"input": "%d %d\n" % (x, x + 1), "output": "%d\n" % (x * (x + 1) + i), "testtype": "stdin"}
                for x in range(4, 9)]
        ref = "import sys\na, b = map(int, sys.stdin.read().split())\nprint(a*b+%d)\n" % i
        meta, starter = "{}", ""
    else:
        q = "Return the sum of nums plus %d (case %d)." % (i, i)
        pub = [{"input": "[1, 2]", "output": "%d" % (3 + i), "testtype": "functional"}]
        priv = [{"input": json.dumps(list(range(x))), "output": str(sum(range(x)) + i), "testtype": "functional"}
                for x in range(2, 7)]
        ref = "class Solution:\n    def addUp(self, nums: List[int]) -> int:\n        return sum(nums) + %d\n" % i
        meta, starter = json.dumps({"func_name": "addUp"}), "class Solution:\n    def addUp(self, nums: List[int]) -> int:\n        "
    LCB.append({"question_title": "q%d" % i, "question_content": q, "platform": "atcoder",
                "question_id": "Q%d" % i, "contest_id": "c", "contest_date": date,
                "starter_code": starter, "difficulty": "easy",
                "public_test_cases": json.dumps(pub),
                "private_test_cases": enc(priv) if i % 3 else json.dumps(priv),
                "metadata": meta})
    REF[q] = ref
LCB_PATH = os.path.join(HERE, "lcb_test6.jsonl")
with open(LCB_PATH, "w") as f:
    for r in LCB:
        f.write(json.dumps(r) + "\n")

hf = types.ModuleType("huggingface_hub")
class _Api:
    def list_repo_files(self, repo, repo_type=None):
        return ["README.md", "test6.jsonl"]
hf.HfApi = _Api
hf.hf_hub_download = lambda repo, fn, repo_type=None: LCB_PATH
sys.modules["huggingface_hub"] = hf
tq = types.ModuleType("tqdm"); tqa = types.ModuleType("tqdm.auto")
tqa.tqdm = lambda it, **k: it
sys.modules["tqdm"] = tq; sys.modules["tqdm.auto"] = tqa
ds = types.ModuleType("datasets"); ds.load_dataset = fake_load_dataset
sys.modules["datasets"] = ds

for r in mb:
    REF[r["prompt"].strip().strip('"').strip().split("\n")[0]] = (r["canonical_solution"], r["entry_point"])
for r in he:
    REF[r["prompt"]] = (r["prompt"] + r["canonical_solution"], r["entry_point"])

# ---------------- mock Ollama -------------------------------------------------
import requests
os.environ["KIMI_API_KEY"] = "test-key"
LAST = {"dur": 0.1}
_calls = [0]

class Resp:
    def __init__(self, data): self._d = data; self.status_code = 200
    def json(self): return self._d
    def raise_for_status(self): pass

def find_problem(user):
    for text, ref in REF.items():
        if text and text in user:
            return text, ref
    return None, None

def bug(code, entry):
    if entry and entry != "main program":
        return code + "\n\ndef %s(*a, **k):\n    return None\n" % entry if "class Solution" not in code \
            else code.replace("return ", "return -1 + ", 1)
    return code.replace("print(", "print(1+", 1)

KIMI_N = [0]
def mock_post(url, json=None, timeout=None, **kw):
    msgs = json["messages"]
    if "chat/completions" in url:                       # the Kimi API
        KIMI_N[0] += 1
        if KIMI_N[0] == 4:                              # one transient rate limit
            r = Resp({}); r.status_code = 429; r.text = "rate limited"; return r
        txt = "Kimi: the loop boundary is off by one; iterate to len(x) inclusive and return the sum."
        u = msgs[-1]["content"]
        return Resp({"choices": [{"message": {"content": txt}}],
                     "usage": {"prompt_tokens": len(u) // 4, "completion_tokens": 25}})
    sysm = msgs[0]["content"] if len(msgs) > 1 else ""
    user = msgs[-1]["content"]
    h = int(hashlib.sha256(user.encode()).hexdigest(), 16)
    rnd = random.Random(h)
    text, ref = find_problem(user)
    if isinstance(ref, tuple):
        code, entry = ref
    else:
        code, entry = ref, ("main program" if ref and "class Solution" not in ref else "addUp")
    if "Reply with" in user:
        out = "OK"
    elif "test engineer" in sysm:
        m = re.search(r"called as `([^`]*)\(\.\.\.\)`", user)
        call = m.group(1) if m else "f"
        good = []
        for t in PUBLIC.get(text, [])[:3]:
            good.append(t)
        if rnd.random() < 0.35:
            good.append("assert %s(%s) == 123456789" % (call, "0" if "Solution" not in call else "[0]"))
        out = "```python\n" + "\n".join(good) + "\n```"
    elif "code reviewer" in sysm:
        out = "The function returns the wrong value for the failing input. Fix the return expression."
    elif "FAILED its tests" in user:
        out = "```python\n%s\n```" % (code if rnd.random() < 0.45 else bug(code, entry))
    else:
        out = "```python\n%s\n```" % (code if rnd.random() < 0.7 else bug(code, entry))
    pt, ct = len(user) // 4 + 40, len(out) // 4
    dur = 0.25 + 0.0004 * pt + 0.03 * ct
    _calls[0] += 1
    load = 3.0 if _calls[0] == 3 else 0.0          # one simulated reload to test the flag
    LAST["dur"] = dur + load
    return Resp({"message": {"content": out}, "prompt_eval_count": pt, "eval_count": ct,
                 "total_duration": int((dur + load) * 1e9), "load_duration": int(load * 1e9)})

requests.post = mock_post
class _PS:
    status_code = 200
    def json(self):
        return {"models": [{"name": "qwen3-coder:30b", "size_vram": 18.6e9},
                           {"name": "qwen2.5-coder:7b", "size_vram": 5.1e9}]}
requests.get = lambda url, **k: _PS()
PUBLIC = {}

class FakeMeter:
    ok, use_energy_counter, indices = True, True, [0, 1]
    def device_info(self): return [(0, "Tesla T4", 16.1, 9.5), (1, "Tesla T4", 16.1, 9.1)]
    def measure_idle_watts(self, seconds=5.0): return 54.9
    def start(self): return {"t0": time.perf_counter(), "ok": True}
    def stop(self, tok):
        w = LAST["dur"] * 1.03
        return {"wall_seconds": w, "joules": w * random.uniform(120, 145), "mean_power_w": 130.0}

# ---------------- run the cells ----------------------------------------------
ns = {"__name__": "__main__"}
SKIP_MARKERS = ["pip_install([", "curl -fsSL https://ollama.com/install.sh", "def ensure_model"]
plt.show = lambda *a, **k: plt.close("all")
for i, c in enumerate(NB["cells"]):
    if c["cell_type"] != "code":
        continue
    s = "".join(c["source"])
    if any(m in s for m in SKIP_MARKERS):
        if "install.sh" in s:
            ns["OLLAMA_HOST"] = "http://127.0.0.1:11434"
        if "def ensure_model" in s:
            ns["ensure_model"] = lambda n: True
            ns["_quick_probe"] = lambda n: ("OK", 0.1)
        print("---- cell %d skipped (install/pull)" % i)
        continue
    if s.startswith("import random, os, time"):            # config: shrink the run
        s += ("\nM1_OUTPUT_DIR = OUTPUT_DIR = %r\nos.makedirs(M1_OUTPUT_DIR, exist_ok=True)\n"
              "M1_N_PROBLEMS = {'mbpp_plus': %d, 'humaneval_plus': %d, 'livecodebench': None}\n"
              "ALLOW_SMALL_N = True\nKIMI_MIN_INTERVAL = 0\n" % (OUT, N_EACH, N_EACH)) + os.environ.get("EXTRA_CFG", "") + "\n"
        if os.environ.get("BUDGET"): s += "\nSESSION_BUDGET_HOURS = %s\n" % os.environ["BUDGET"]
    print("---- cell %d" % i)
    t0 = time.time()
    try:
        exec(compile(s, "<cell %d>" % i, "exec"), ns)
    except Exception:
        traceback.print_exc()
        print("FAILED at cell", i)
        sys.exit(1)
    if s.startswith("import time, threading"):            # GPUMeter defined -> swap in the fake
        ns["M1_METER"] = FakeMeter()
    if "BENCH_PROBLEMS = {}" in s:
        for tag, (probs, lab) in ns["BENCH_PROBLEMS"].items():
            for p in probs:
                PUBLIC[p["text"]] = p["public_tests"] if p["test_mode"] == "assert" else []
    print("     (%.1fs)" % (time.time() - t0))
for extra in ["cells/415.py"]:
    pass
print("DRY RUN OK")
