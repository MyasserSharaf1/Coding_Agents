import json, copy, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
_o = os.path.join(HERE, "orig.ipynb")
if not os.path.exists(_o):     # in the repository the original v1 notebook is the base
    _o = os.path.join(HERE, "..", "notebooks", "01_m1-v1-baseline-vs-repair_n20.ipynb")
orig = json.load(open(_o))
# usage: python build.py <output.ipynb> [evalplus|livecodebench]
PART = sys.argv[2] if len(sys.argv) > 2 else "evalplus"
assert PART in ("evalplus", "livecodebench"), PART
PART_TITLE = {"evalplus": "Part A: MBPP+ and HumanEval+",
              "livecodebench": "Part B: LiveCodeBench"}[PART]
O = orig["cells"]

def src(i):
    return "".join(O[i]["source"])

def rd(name):
    return open(os.path.join(HERE, "cells", name)).read().rstrip() + "\n"

deps = src(3).replace('"nvidia-ml-py",        # NVML: GPU power and energy',
                      '"nvidia-ml-py",        # NVML: GPU power and energy\n'
                      '    "huggingface_hub>=0.20",  # LiveCodeBench files\n'
                      '    "scipy>=1.10",           # rank correlations')
assert "huggingface_hub" in deps

seq = [
    ("md", rd("000.md").replace("{PART_TITLE}", PART_TITLE)),
    ("md", "## 1 · Environment"), ("code", src(1)),
    ("md", "## 2 · Dependencies"), ("code", deps),
    ("md", "### 2b · Claude Code (optional)\nInstalls the `claude` command on this VM so you can ask it about "
           "the results. Not part of the experiment; off by default."), ("code", rd("025.py")),
    ("md", "## 3 · Configuration — the only cell you should need to edit"), ("code", rd("030.py").replace('M1_PART = "evalplus"', 'M1_PART = "%s"' % PART)),
    ("md", "### 3b · Resume from an earlier session\nCopies results saved by an earlier, unfinished run "
           "(attached as an input) into this run's output folder."), ("code", rd("035.py")),
    ("md", src(6)), ("code", src(7)),
    ("md", src(8)), ("code", src(9).replace('os.environ["OLLAMA_MAX_LOADED_MODELS"] = "1"  # nothing else resident in VRAM',
                               'os.environ["OLLAMA_MAX_LOADED_MODELS"] = "2" if REVIEWER_BACKEND == "local" else "1"'
                               '  # code model + local reviewer stay resident')),
    ("md", src(10)), ("code", src(11)), ("code", rd("115.py")),
    ("md", "## 7 · Benchmark helpers\nTest-block wrapping, call-signature hints, and the HumanEval "
           "public tests (the worked examples in each docstring)."), ("code", rd("130.py")),
    ("md", "## 8 · Execution sandbox\nEvery program runs in a separate process with a time and memory "
           "limit. `assert` mode for function tests, `stdin` mode for LiveCodeBench programs. Grading "
           "results are memoised so a shared draft is executed once."), ("code", rd("150.py")),
    ("md", "## 9 · LLM interface"), ("code", rd("170.py")),
    ("md", "### 9b · Remote reviewer (only when `REVIEWER_BACKEND = \"kimi\"`)\nOpenAI-compatible client for "
           "Kimi K2.6. Needs a Kaggle secret named `KIMI_API_KEY`. With the default local reviewer this cell "
           "does nothing."),
    ("code", rd("175.py")),
    ("md", "## 10 · Agents\n**Code agent** writes and repairs code · **Test agent** writes extra "
           "asserts (system `repair_tests` only) · **Supervisor** diagnoses failures."),
    ("code", rd("190.py")), ("code", rd("200.py")), ("code", src(21)),
    ("md", "## 11 · Repair orchestrator\nGenerate → [test agent] → run visible tests → "
           "(supervisor → repair → run) × up to `MAX_RETRIES`. Grading happens outside, on the hidden suite."),
    ("code", rd("230.py")),
    ("md", "## 12 · Baseline, evaluation and checkpointing\nEach question runs every system in turn "
           "(baseline first, so it pays for the shared draft), then the evaluation-only checks, then "
           "is appended to `rows.jsonl`. A killed session loses at most the question in flight."),
    ("code", rd("250.py")),
    ("md", src(26)), ("code", src(27)), ("code", src(28)),
    ("md", "## 14 · Metered model calls\n\nEvery agent goes through `OllamaLLM.chat`, so metering it "
           "captures every cost axis. Each call is tagged with its role. The systems share one first "
           "draft through the prompt cache; the meter **replays the original cost on a cache hit** so "
           "each system is charged for it, and flags the replay so per-call analyses can drop it."),
    ("code", rd("300.py")),
    ("md", "## 15 · Load MBPP+, HumanEval+ and LiveCodeBench\nPublic tests that the reference "
           "solution itself fails are dropped before the run."), ("code", rd("330.py")),
    ("md", "## 16 · Run the experiment\nThe long cell. Safe to re-run: finished questions are skipped. "
           "If the session budget runs out, start a new session and run all cells again."),
    ("code", rd("350.py")),
    ("md", "## 17 · Metrics per benchmark and system\n`false_accept_%` = programs the loop accepted "
           "(visible tests pass) that fail the hidden suite."), ("code", rd("370.py")),
    ("md", "## 18 · Paired statistics\nSame questions, every pair of systems → exact McNemar on the "
           "discordant pairs, Holm-corrected across all comparisons.\n- **Do-no-harm gate:** "
           "regressions must be 0 · **n ≥ 45** before a p-value means anything"),
    ("code", rd("390.py")),
    ("md", "## 19 · Test agent diagnostics\nIs a generated test correct (the reference solution "
           "passes it)? Did it catch a bug the public tests missed, or raise a false alarm on a "
           "correct draft?"), ("code", rd("410.py")),
    ("md", "### 19b · Regressions and the keep-the-best rule\nWhy a system lost questions the baseline "
           "solved, and what pass@1 would be if the loop returned its best program by visible tests "
           "instead of its last one. Re-scored from saved traces: no model calls, same cost."),
    ("code", rd("415.py")),
    ("md", "### 19c · Walkthrough: single questions, step by step\nThe draft, the generated tests, every "
           "review and repair, and the visible / hidden result after each round. Set `WALKTHROUGH_TASKS` to "
           "choose questions; by default it picks the most informative ones."),
    ("code", rd("418.py")),
*([] if PART != "livecodebench" else [
    ("md", "### 19d · LiveCodeBench sanity check and re-grade\nBreaks the LiveCodeBench baseline down by "
           "problem type, difficulty and failure reason, and re-grades every saved program without the "
           "whole-suite time cap (per-case limits only, as the official harness does). CPU only."),
    ("code", rd("422.py")),
    ]),
    ("md", "## 20 · Charts\nBlue is always the baseline, orange repair, aqua (hatched) repair + test "
           "agent. Every figure is saved as a PNG."), ("code", rd("430.py")),
    ("code", rd("450.py")), ("code", rd("460.py")), ("code", rd("470.py")), ("code", rd("480.py")),
    ("code", rd("490.py")), ("code", rd("500.py")), ("code", rd("510.py")),
    ("md", "## 21 · Cost of each agent inside each system"), ("code", rd("530.py")),
    ("code", rd("540.py")), ("code", rd("550.py")),
    ("md", rd("570.md")), ("code", rd("575.py")), ("code", rd("580.py")), ("code", rd("590.py")),
    ("code", rd("600.py")),
    ("md", "## 23 · Summary report"), ("code", rd("620.py")),
    ("md", rd("660.md")), ("code", rd("660.py")),
    ("md", rd("640.md")),
]

cells = []
for kind, s in seq:
    lines = s.splitlines(keepends=True)
    if kind == "md":
        cells.append({"cell_type": "markdown", "metadata": {}, "source": lines})
    else:
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                      "outputs": [], "source": lines})
nb = copy.deepcopy(orig)
nb["cells"] = cells
for i, c in enumerate(nb["cells"]):
    c["id"] = "c%03d" % i
nb["nbformat"], nb["nbformat_minor"] = 4, 5
out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "m1-repair-vs-baseline-v2.ipynb")
json.dump(nb, open(out, "w"), indent=1, ensure_ascii=False)
print("wrote", out, len(cells), "cells")
