import json, re, base64, zlib, pickle
from datasets import load_dataset

M1_SOURCES = {
    ("mbpp", "plus"):       [("evalplus/mbppplus",)],
    ("humaneval", "plus"):  [("evalplus/humanevalplus",), ("evalplus/HumanEvalPlus",)],
}
M1_LABELS = {("mbpp", "plus"): "MBPP+", ("humaneval", "plus"): "HumanEval+",
             ("livecodebench", "lite"): "LiveCodeBench"}
M1_TAGS = {("mbpp", "plus"): "mbpp_plus", ("humaneval", "plus"): "humaneval_plus",
           ("livecodebench", "lite"): "livecodebench"}


def _validate_public(probs):
    """Drop public tests that the reference solution itself fails (a handful of MBPP
    asserts disagree with the MBPP+ corrected reference). Returns #tests dropped."""
    dropped = 0
    for p in probs:
        ref, pub = p.get("reference_code"), p.get("public_tests") or []
        if not ref or not pub:
            continue
        res = run_in_sandbox(ref, pub, p["test_setup_code"], timeout=p["timeout"])
        if res["import_ok"] and not res.get("timed_out"):
            keep = [pub[r["index"]] for r in res["results"] if r["passed"]]
            dropped += len(pub) - len(keep)
            p["public_tests"] = keep
    return dropped


def _load_evalplus(benchmark, variant):
    dsd, last = None, None
    for cfg in M1_SOURCES[(benchmark, variant)]:
        try:
            dsd = load_dataset(*cfg); break
        except Exception as e:
            last = e
    if dsd is None:
        raise RuntimeError("Failed to load %s: %s" % (M1_LABELS[(benchmark, variant)], last))
    split = "test" if "test" in dsd else list(dsd.keys())[0]
    d = dsd[split]
    probs = []
    if benchmark == "humaneval":
        for i in range(len(d)):
            r = d[i]
            ep = r.get("entry_point")
            prompt = r.get("prompt") or ""
            probs.append({
                "task_id": r.get("task_id"), "text": prompt,
                "reference_code": prompt + (r.get("canonical_solution") or ""),
                "test_list": [wrap_composite_test(r.get("test"), ep)],     # hidden
                "public_tests": humaneval_public_tests(prompt, ep),       # docstring examples
                "test_setup_code": HUMANEVAL_SETUP, "entry_point": ep,
                "test_mode": "assert", "timeout": TIMEOUT_SECONDS,
                "signature_hint": None,   # the prompt already carries the signature
                "ta_supported": True, "ta_call": "%s(...)" % ep,
            })
    else:
        tcol = "text" if "text" in d.column_names else "prompt"
        scol = ("test_setup_code" if "test_setup_code" in d.column_names
                else ("test_imports" if "test_imports" in d.column_names else None))
        for i in range(len(d)):
            r = d[i]
            setup = ""
            if scol:
                raw = r.get(scol, "") or ""
                setup = "\n".join(raw) if isinstance(raw, (list, tuple)) else str(raw)
            original = list(r.get("test_list") or [])
            ep = extract_entry_point(original)
            hidden = original
            ext = r.get("test") or r.get("assertion") or ""
            if isinstance(ext, (list, tuple)):
                ext = "\n".join(ext)
            if MBPP_PLUS_USE_EXTENDED_TESTS and ext and ext.strip() and ep:
                hidden = [wrap_composite_test(ext, ep)]
            probs.append({
                "task_id": r.get("task_id"), "text": r.get(tcol) or "",
                "reference_code": r.get("code", ""),
                "test_list": hidden, "public_tests": original,
                "test_setup_code": setup, "entry_point": ep,
                "test_mode": "assert", "timeout": TIMEOUT_SECONDS,
                "signature_hint": extract_call_signature(original, ep),
                "ta_supported": True, "ta_call": "%s(...)" % ep,
            })
    return [p for p in probs if p["entry_point"] and p["test_list"]]


# ---- LiveCodeBench ----------------------------------------------------------
LCB_SETUP = ("from typing import *\nfrom collections import *\nfrom itertools import *\n"
             "from functools import *\nfrom heapq import *\nfrom bisect import *\n"
             "import math, string, re, sys, random, collections, heapq, bisect, itertools, functools\n"
             "import json as _json\n"
             "sys.setrecursionlimit(10**6)\n"
             "def _lcb_norm(x):\n"
             "    if isinstance(x, tuple): x = list(x)\n"
             "    if isinstance(x, list): return [_lcb_norm(v) for v in x]\n"
             "    if isinstance(x, float): return round(x, 6)\n"
             "    return x\n")

LCB_STDIN_INSTR = ("Read the inputs from stdin, solve the problem and write the answer to stdout "
                   "(do not directly test on the sample inputs). Return one complete Python "
                   "program in one python code block.")
LCB_STDIN_REPAIR = ("Return a corrected, complete Python program in one python code block. "
                    "It must read from stdin and write to stdout.")


def _lcb_decode_private(s):
    if not s:
        return []
    try:
        return json.loads(s)
    except Exception:
        return json.loads(pickle.loads(zlib.decompress(base64.b64decode(s.encode("utf-8")))))


def _lcb_functional_asserts(cases, func):
    out = []
    for c in cases:
        args = [ln for ln in c["input"].split("\n") if ln.strip()]
        out.append("assert _lcb_norm(Solution().%s(*[_json.loads(_a) for _a in %r])) == "
                   "_lcb_norm(_json.loads(%r))" % (func, args, c["output"].strip()))
    return out


def _load_livecodebench():
    from huggingface_hub import HfApi, hf_hub_download
    files = LCB_FILES
    if not files:
        files = sorted(f for f in HfApi().list_repo_files(LCB_REPO, repo_type="dataset")
                       if re.fullmatch(r"test\d*\.jsonl", f))
    seen, raw = set(), []
    for fn in files:
        path = hf_hub_download(LCB_REPO, fn, repo_type="dataset")
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r["question_id"] in seen:
                    continue
                seen.add(r["question_id"]); raw.append(r)
    print("  LiveCodeBench files: %s -> %d unique problems" % (", ".join(files), len(raw)))

    probs = []
    for r in raw:
        public = json.loads(r["public_test_cases"]) if r.get("public_test_cases") else []
        private = _lcb_decode_private(r.get("private_test_cases"))
        if not public and not private:
            continue
        meta = json.loads(r["metadata"]) if r.get("metadata") else {}
        ttype = (public or private)[0].get("testtype", "stdin")
        p = {"task_id": "LCB/%s" % r["question_id"], "text": r["question_content"],
             "contest_date": str(r.get("contest_date", ""))[:10],
             "difficulty": r.get("difficulty"), "platform": r.get("platform"),
             "reference_code": None, "test_setup_code": LCB_SETUP,
             "timeout": LCB_TIMEOUT_S, "mem_gb": 3, "signature_hint": None}
        if ttype == "functional":
            func = meta.get("func_name")
            if not func:
                continue
            starter = r.get("starter_code") or ""
            p.update({
                "entry_point": func, "test_mode": "assert",
                "public_tests": _lcb_functional_asserts(public, func),
                "test_list": _lcb_functional_asserts(public + private, func),
                "instruction": ("Complete the following starter code and return the whole "
                                "class in one python code block:\n```python\n%s\n```" % starter),
                "repair_instruction": ("Return the corrected, complete `Solution` class in one "
                                       "python code block; keep the method name `%s` and its "
                                       "signature." % func),
                "ta_supported": True, "ta_call": "Solution().%s(...)" % func,
            })
        else:
            clean = lambda cs: [{"input": c["input"], "output": c["output"]} for c in cs]
            p.update({
                "entry_point": "main program", "test_mode": "stdin",
                "public_tests": clean(public), "test_list": clean(public + private),
                "instruction": LCB_STDIN_INSTR, "repair_instruction": LCB_STDIN_REPAIR,
                "ta_supported": False,     # asserts cannot drive a stdin program
            })
        probs.append(p)

    window = [p for p in probs if p["contest_date"] >= LCB_MIN_DATE]
    if len(window) < M1_MIN_PROBLEMS:
        print("  only %d problems on/after %s -> taking the newest %d instead"
              % (len(window), LCB_MIN_DATE, M1_MIN_PROBLEMS))
        window = sorted(probs, key=lambda p: p["contest_date"], reverse=True)[:M1_MIN_PROBLEMS]
    if window:
        print("  contest dates %s .. %s  (check these against the model's training cutoff)"
              % (min(p["contest_date"] for p in window), max(p["contest_date"] for p in window)))
    return window


def prepare_benchmark_m1(benchmark, variant=None):
    """-> (problems, label, tag)"""
    key = (benchmark, variant)
    if benchmark == "livecodebench":
        probs = _load_livecodebench()
    elif key in M1_SOURCES:
        probs = _load_evalplus(benchmark, variant)
    else:
        raise ValueError("Unknown benchmark/variant: %r" % (key,))
    return probs, M1_LABELS[key], M1_TAGS[key]


BENCH_PROBLEMS = {}
for _k in M1_BENCHMARKS:
    try:
        _p, _l, _t = prepare_benchmark_m1(*_k)
        _drop = _validate_public(_p)
        BENCH_PROBLEMS[_t] = (_p, _l)
        n_pub = sum(1 for p in _p if p["public_tests"])
        n_ta = sum(1 for p in _p if p.get("ta_supported"))
        print("%-15s %-14s %4d problems | %d with public tests (%d reference-failing public tests dropped)"
              " | test agent usable on %d"
              % (_t, _l, len(_p), n_pub, _drop, n_ta))
    except Exception as e:
        print("%-15s FAILED: %s" % (_k, e))
