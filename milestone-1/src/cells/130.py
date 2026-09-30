import re, random, ast, doctest

# HumanEval signatures use typing names; provide them so a missing import is not
# scored as a logic failure (standard in HumanEval harnesses).
HUMANEVAL_SETUP = ("from typing import List, Tuple, Dict, Set, Optional, Any, Union\n"
                   "import math\n")


def extract_entry_point(test_list):
    for t in test_list:
        m = re.search(r'assert\s+([A-Za-z_]\w*)\s*\(', t)
        if m:
            return m.group(1)
        m = re.search(r'([A-Za-z_]\w*)\s*\(', t)
        if m:
            return m.group(1)
    return None


def wrap_composite_test(test_code, entry):
    """Turn a benchmark test BLOCK into one runnable unit.
      1. defines `def check(candidate)`  -> call check(entry)
      2. refers to a bare `candidate`    -> bind candidate = entry first
      3. plain assert statements         -> run as-is
    """
    t = test_code or ""
    if "def check(" in t:
        return "%s\n\ncheck(%s)\n" % (t, entry)
    if "candidate" in t:
        return "candidate = %s\n%s\n" % (entry, t)
    return t


def extract_call_signature(asserts, entry, max_len=160):
    """Hint showing HOW the function is called, taken from the example tests."""
    if not entry:
        return None
    for a in asserts:
        i = a.find(entry + "(")
        if i < 0:
            continue
        j = i + len(entry)
        depth, k = 0, j
        for k in range(j, len(a)):           # balanced-paren scan
            if a[k] == "(":
                depth += 1
            elif a[k] == ")":
                depth -= 1
                if depth == 0:
                    break
        call = a[i:k + 1]
        inner = call[len(entry) + 1:-1]
        depth, n_args = 0, (1 if inner.strip() else 0)
        for ch in inner:
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth -= 1
            elif ch == "," and depth == 0:
                n_args += 1
        if len(call) > max_len:
            call = call[:max_len] + "...)"
        return ("The function is called with %d positional argument(s). "
                "Example call from the tests: %s" % (n_args, call))
    return None


# ---- HumanEval public tests: the examples in the docstring ------------------
_ARROWS = ["==>", "➞", "=>", "->", "==", "should return", "returns", "="]

def _is_expr(src):
    try:
        ast.parse(src.strip(), mode="eval"); return True
    except Exception:
        return False

def _literal(src):
    try:
        ast.literal_eval(src.strip()); return True
    except Exception:
        return False

def humaneval_public_tests(prompt, entry):
    """Public tests = the worked examples a HumanEval prompt shows the model.
    Handles doctest blocks (>>> f(x) / expected) and one-line forms such as
    `f(x) ==> y`, `f(x) -> y`, `f(x) == y`. Only literal expected values are kept."""
    tests = []
    lines = prompt.splitlines()
    i = 0
    while i < len(lines):                     # doctest-style: ">>> call" then the value
        s = lines[i].strip()
        if s.startswith(">>>"):
            src = s[3:].strip()
            want, j = [], i + 1
            while j < len(lines) and lines[j].strip() and not lines[j].strip().startswith(">>>") \
                    and not lines[j].strip().startswith('"""'):
                want.append(lines[j].strip()); j += 1
            w = " ".join(want)
            if entry + "(" in src and w and _is_expr(src) and _literal(w):
                tests.append("assert (%s) == (%s)" % (src, w))
            i = j
        else:
            i += 1
    if tests:
        return tests
    for line in lines:
        s = line.strip().lstrip("#").strip()
        if not s.startswith(entry + "("):
            continue
        for arrow in _ARROWS:
            if arrow in s:
                left, right = s.split(arrow, 1)
                right = right.strip().rstrip(".").strip()
                right = {"true": "True", "false": "False"}.get(right, right)
                if _is_expr(left) and _literal(right):
                    tests.append("assert (%s) == (%s)" % (left.strip(), right))
                break
    return tests


def sample_main_set(problems, n, seed=SEED):
    """Reproducible sample: shuffle once with SEED, take the first n (None = all)."""
    rr = random.Random(seed)
    order = list(range(len(problems)))
    rr.shuffle(order)
    n_eff = len(problems) if n is None else min(n, len(problems))
    return [problems[i] for i in order[:n_eff]]

print("Benchmark helpers ready.")
