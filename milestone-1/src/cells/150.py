import tempfile, subprocess, os, json, shutil, sys, hashlib

# Runner executed inside the subprocess. Two modes:
#   "assert": exec the solution once, then run each test string in that namespace
#   "stdin" : each test is {"input", "output"}; the program runs once per test with
#             that stdin and its stdout is compared token by token (LiveCodeBench)
RUNNER_SRC = r'''
import json, sys, os, ast, io, traceback

def _describe(test_src, ns):
    # For 'assert <left> <op> <right>' return the actual vs expected values.
    try:
        node = ast.parse(test_src.strip()).body[0]
        if isinstance(node, ast.Assert) and isinstance(node.test, ast.Compare) \
           and len(node.test.ops) == 1:
            left = ast.Expression(node.test.left)
            right = ast.Expression(node.test.comparators[0])
            lval = eval(compile(left, '<l>', 'eval'), ns)
            rval = eval(compile(right, '<r>', 'eval'), ns)
            op = type(node.test.ops[0]).__name__
            return ('got %r, expected (%s) %r' % (lval, op, rval))[:600]
    except Exception:
        return None
    return None

def _tokens_equal(got, exp):
    g, e = got.split(), exp.split()
    if len(g) != len(e):
        return False
    for a, b in zip(g, e):
        if a == b:
            continue
        try:
            fa, fb = float(a), float(b)
            if abs(fa - fb) <= 1e-6 * max(1.0, abs(fb)):
                continue
        except ValueError:
            pass
        return False
    return True

def _clip(s, n=300):
    s = s if isinstance(s, str) else repr(s)
    return s if len(s) <= n else s[:n] + '...'

def run_assert(payload, out):
    ns = {}
    try:
        if payload.get('setup'):
            exec(compile(payload['setup'], '<setup>', 'exec'), ns)
        exec(compile(payload['code'], '<solution>', 'exec'), ns)
    except BaseException:
        out['import_ok'] = False
        out['fatal'] = traceback.format_exc()
        return
    for i, t in enumerate(payload['tests']):
        r = {'index': i, 'passed': False, 'error': None, 'test': t[:200], 'detail': None}
        try:
            exec(compile(t, '<test%d>' % i, 'exec'), ns)
            r['passed'] = True
        except AssertionError:
            r['error'] = traceback.format_exc()
            r['detail'] = _describe(t, ns)
        except BaseException:
            r['error'] = traceback.format_exc()
        out['results'].append(r)
        if payload.get('stop_on_fail') and not r['passed']:
            break

def run_stdin(payload, out):
    # Each case runs the program as its own process with the case on real stdin,
    # so input(), sys.stdin.buffer and open(0) all behave as on a judge.
    import subprocess
    try:
        compile(payload['code'], '<solution>', 'exec')
    except BaseException:
        out['import_ok'] = False
        out['fatal'] = traceback.format_exc()
        return
    with open('solution.py', 'w') as f:
        f.write(payload['code'])
    for i, t in enumerate(payload['tests']):
        r = {'index': i, 'passed': False, 'error': None,
             'test': 'stdin case %d' % i, 'detail': None}
        err, got = None, ''
        try:
            pr = subprocess.run([sys.executable, 'solution.py'], input=t['input'],
                                capture_output=True, text=True,
                                timeout=payload.get('case_timeout', 10))
            got = pr.stdout
            if pr.returncode != 0:
                err = (pr.stderr or 'exit code %d' % pr.returncode)[-1500:]
        except subprocess.TimeoutExpired:
            err = 'TimeoutExpired: case %d exceeded the per-case limit' % i
        if err:
            r['error'] = err
        elif _tokens_equal(got, t['output']):
            r['passed'] = True
        else:
            r['error'] = 'AssertionError: wrong answer'
            r['detail'] = 'input %s -> got %s, expected %s' % (
                _clip(t['input']), _clip(got), _clip(t['output']))
        out['results'].append(r)
        if payload.get('stop_on_fail') and not r['passed']:
            break

def main():
    d = sys.argv[1]
    with open(os.path.join(d, 'payload.json')) as f:
        payload = json.load(f)
    out = {'import_ok': True, 'fatal': None, 'results': []}
    (run_stdin if payload.get('mode') == 'stdin' else run_assert)(payload, out)
    sys.stdout.write('\n' + json.dumps(out) + '\n')

if __name__ == '__main__':
    main()
'''

def _rlimit_preexec(timeout, mem_gb):
    def _apply():
        try:
            import resource
            resource.setrlimit(resource.RLIMIT_CPU, (timeout + 1, timeout + 2))
            resource.setrlimit(resource.RLIMIT_AS, (mem_gb * 1024**3, mem_gb * 1024**3))
        except Exception:
            pass
    return _apply

def run_in_sandbox(code, tests, setup="", timeout=None, mode="assert",
                   stop_on_fail=False, mem_gb=2):
    if timeout is None:
        timeout = TIMEOUT_SECONDS
    d = tempfile.mkdtemp(prefix="mas_")
    empty = {"timed_out": False, "import_ok": False, "fatal": None,
             "results": [], "n_tests": len(tests), "n_passed": 0,
             "n_failed": len(tests), "stdout": "", "stderr": ""}
    try:
        with open(os.path.join(d, "runner.py"), "w") as f:
            f.write(RUNNER_SRC)
        with open(os.path.join(d, "payload.json"), "w") as f:
            json.dump({"code": code, "setup": setup, "tests": tests, "mode": mode,
                       "stop_on_fail": stop_on_fail}, f)
        kw = dict(capture_output=True, text=True, timeout=timeout, cwd=d,
                  stdin=subprocess.DEVNULL)
        if os.name == "posix":
            kw["preexec_fn"] = _rlimit_preexec(timeout, mem_gb)
        try:
            proc = subprocess.run([sys.executable, "runner.py", d], **kw)
        except subprocess.TimeoutExpired:
            r = dict(empty); r["timed_out"] = True; r["fatal"] = "TimeoutExpired"
            r["stderr"] = "Timeout after %ss" % timeout; return r
        parsed = None
        for line in reversed(proc.stdout.strip().splitlines()):
            line = line.strip()
            if line.startswith("{"):
                try:
                    parsed = json.loads(line); break
                except Exception:
                    pass
        if parsed is None:
            r = dict(empty); r["fatal"] = (proc.stderr or "no runner output")[:2000]
            r["stdout"] = proc.stdout[-2000:]; r["stderr"] = proc.stderr[-2000:]; return r
        n_pass = sum(1 for x in parsed["results"] if x["passed"])
        return {"timed_out": False, "import_ok": parsed["import_ok"],
                "fatal": parsed["fatal"], "results": parsed["results"],
                "n_tests": len(tests), "n_passed": n_pass,
                "n_failed": len(tests) - n_pass,
                "stdout": proc.stdout[-2000:], "stderr": proc.stderr[-2000:]}
    finally:
        shutil.rmtree(d, ignore_errors=True)


def all_passed(res):
    return bool(res["import_ok"]) and res["n_tests"] > 0 and res["n_failed"] == 0


# ---- problem-aware wrappers -------------------------------------------------
# A problem carries: test_list (HIDDEN grading suite), public_tests, test_mode,
# test_setup_code, timeout. Grading results are memoised by (code, suite) so the
# same draft is executed once, not once per system.
_GRADE_CACHE = {}

def run_suite(p, code, tests, stop_on_fail=False, mode=None):
    mode = mode or p.get("test_mode", "assert")
    key = hashlib.sha256(json.dumps([code, tests, mode, stop_on_fail, p["task_id"]],
                                    sort_keys=True, default=str).encode()).hexdigest()
    if key in _GRADE_CACHE:
        return _GRADE_CACHE[key]
    res = run_in_sandbox(code, tests, p.get("test_setup_code", ""),
                         timeout=p.get("timeout", TIMEOUT_SECONDS), mode=mode,
                         stop_on_fail=stop_on_fail, mem_gb=p.get("mem_gb", 2))
    _GRADE_CACHE[key] = res
    return res

def grade_hidden(p, code):
    """Pass/fail on the hidden grading suite (evaluation only, never shown to agents)."""
    return all_passed(run_suite(p, code, p["test_list"], stop_on_fail=True))

def passes_public(p, code):
    t = p.get("public_tests") or []
    return all_passed(run_suite(p, code, t)) if t else None


def first_error(res):
    if res.get("timed_out"):
        return "TimeoutExpired: execution exceeded the time limit"
    if res.get("fatal"):
        return res["fatal"][:1500]
    for x in res.get("results", []):
        if not x["passed"] and x.get("error"):
            # BARE mode: raw traceback only - reproduces the weak original setup
            if globals().get("FEEDBACK_MODE", "structured") == "bare":
                return x["error"][:1500]
            parts = []
            t = x.get("test") or ""
            # only echo short, single-assert tests; composite check() bodies are
            # noise (the traceback already pinpoints the failing line)
            if t and len(t) < 200 and t.count("assert") <= 1:
                parts.append("Failing test: " + t)
            if x.get("detail"):          # e.g. "got 8, expected (Eq) 6"
                parts.append("Observed: " + x["detail"])
            parts.append(x["error"][:1200])
            return "\n".join(parts)[:1800]
    return ""

def classify_failure(res):
    if res.get("timed_out"):
        return "Timeout"
    text = first_error(res)
    if not text:
        return "None"
    if "SyntaxError" in text or "IndentationError" in text:
        return "SyntaxError"
    if "ModuleNotFoundError" in text or "ImportError" in text:
        return "Missing import"
    if "AssertionError" in text:
        return "AssertionError"
    if "TypeError" in text and ("argument" in text or "positional" in text):
        return "Wrong function signature"
    if "NameError" in text:
        return "Wrong function signature"
    if "Error" in text or "Exception" in text:
        return "RuntimeError"
    return "Other"

# self-tests: assert mode, stdin mode, wrong answers in both
_t = run_in_sandbox("def add(a,b):\n    return a+b\n", ["assert add(2,3)==5"])
print("Sandbox (assert) passes correct code  :", all_passed(_t))
_t2 = run_in_sandbox("def add(a,b):\n    return a-b\n", ["assert add(2,3)==5"])
print("Sandbox (assert) catches wrong answer :", not all_passed(_t2), "->", classify_failure(_t2))
_prog = "a, b = map(int, input().split())\nprint(a + b)\n"
_t3 = run_in_sandbox(_prog, [{"input": "2 3\n", "output": "5\n"}], mode="stdin")
print("Sandbox (stdin)  passes correct code  :", all_passed(_t3))
_t4 = run_in_sandbox(_prog, [{"input": "2 3\n", "output": "6\n"}], mode="stdin")
print("Sandbox (stdin)  catches wrong answer :", not all_passed(_t4), "->", _t4["results"][0]["detail"])
