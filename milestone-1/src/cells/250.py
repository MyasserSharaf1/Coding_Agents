import json, os, time
import pandas as pd
from tqdm.auto import tqdm

class BaselineSystem:
    """One generation. No tests are run as part of the pipeline."""
    def __init__(self, code_agent):
        self.code_agent = code_agent
        self.name = "baseline"

    def solve(self, p):
        hint = p.get("signature_hint") if USE_SIGNATURE_HINT else None
        code, _ = self.code_agent.generate(p["text"], p["entry_point"], hint,
                                           p.get("instruction"))
        rec = {"task_id": p["task_id"], "system": "baseline", "iterations": 0,
               "executions": 0, "exec_seconds": 0.0, "stop_reason": "single sample",
               "n_visible_tests": 0, "n_generated_tests": 0,
               "draft_visible_pass": None, "final_visible_pass": None,
               "final_code_changed": False}
        trace = {"task_id": p["task_id"], "system": "baseline", "initial_code": code,
                 "final_code": code}
        return rec, trace, code, code, []


def build_systems(num_predict):
    """Fresh agents for one benchmark. Every system shares ONE local llm (and its cache),
    so they all start from the same first draft."""
    global llm, supervisor_llm, code_agent, test_agent, supervisor, SYSTEM_OBJS
    llm = OllamaLLM(MODEL_NAME, temperature=TEMPERATURE, num_predict=num_predict)
    supervisor_llm = (OllamaLLM(SUPERVISOR_MODEL, temperature=TEMPERATURE,
                                num_predict=num_predict)
                      if SUPERVISOR_MODEL and SUPERVISOR_MODEL != MODEL_NAME else llm)
    code_agent = CodeGenerationAgent(llm)
    test_agent = TestGenerationAgent(llm)
    supervisor = SupervisorAgent(supervisor_llm)
    if REVIEWER_BACKEND == "kimi":
        reviewer_llm = KIMI_LLM
    else:   # a second local model: metered like every other local call
        reviewer_llm = OllamaLLM(REVIEWER_MODEL, temperature=TEMPERATURE, num_predict=num_predict)
    reviewer = SupervisorAgent(reviewer_llm) if reviewer_llm is not None else None
    common = dict(max_retries=MAX_RETRIES, combine_supervisor_repair=COMBINE_SUPERVISOR_REPAIR,
                  early_stop_no_progress=EARLY_STOP_NO_PROGRESS)
    SYSTEM_OBJS = {}
    for name in SYSTEMS:
        if name == "baseline":
            SYSTEM_OBJS[name] = BaselineSystem(code_agent)
            continue
        who = SUPERVISOR_BY_SYSTEM.get(name, "self")
        if who == "reviewer" and reviewer is None:
            raise RuntimeError("%s needs the reviewer, but it was not created." % name)
        SYSTEM_OBJS[name] = MultiAgentOrchestrator(
            code_agent, test_agent, reviewer if who == "reviewer" else supervisor,
            use_test_agent=name in TEST_AGENT_SYSTEMS, name=name, **common)
    return llm


def _evaluate(p, rec, draft, final, gtests):
    """Evaluation-only checks. Nothing here is charged to any system."""
    rec["draft_hidden_pass"] = grade_hidden(p, draft)
    rec["final_hidden_pass"] = grade_hidden(p, final)
    rec["draft_public_pass"] = passes_public(p, draft)
    rec["final_public_pass"] = passes_public(p, final)
    rec["failure_type"] = ("None" if rec["draft_hidden_pass"] else
                           classify_failure(run_suite(p, draft, p["test_list"], stop_on_fail=True)))
    # test-agent diagnostics
    rec["ta_n_valid"] = float("nan")
    rec["ta_flags_draft"] = None
    rec["ta_caught"] = None
    rec["ta_false_alarm"] = None
    if gtests:
        if p.get("reference_code"):
            rr = run_suite(p, p["reference_code"], gtests, mode="assert")
            rec["ta_n_valid"] = float(rr["n_passed"]) if rr["import_ok"] else 0.0
        flags = not all_passed(run_suite(p, draft, gtests, mode="assert"))
        rec["ta_flags_draft"] = flags
        # caught: public tests missed a real bug that the generated tests flagged
        rec["ta_caught"] = bool(flags and not rec["draft_hidden_pass"]
                                and rec["draft_public_pass"] is not False)
        # false alarm: generated tests reject a draft that is actually correct
        rec["ta_false_alarm"] = bool(flags and rec["draft_hidden_pass"])
    return rec


def _append_jsonl(path, records):
    # a session killed mid-write leaves a line without its newline; start a fresh line
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, "rb") as f:
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n":
                with open(path, "a") as g:
                    g.write("\n")
    with open(path, "a") as f:
        for r in records:
            f.write(json.dumps(r, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _read_jsonl(path):
    out = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass        # a line truncated by a killed session is skipped
    return out


def evaluate_question(p, tag):
    """Run every system on one question; return rows, traces and the calls it made."""
    rows, traces = [], []
    i_start = len(CALL_LOG)
    for name in SYSTEMS:
        i0 = len(CALL_LOG)
        rec, trace, draft, final, gtests = SYSTEM_OBJS[name].solve(p)
        cost = _cost_slice(i0)
        _tag_calls(i0, p["task_id"], name)
        rec = _evaluate(p, rec, draft, final, gtests)
        rec.update(cost)
        rec["benchmark"] = tag
        rows.append(rec)
        traces.append(trace)
    return rows, traces, CALL_LOG[i_start:]

print("Systems:", ", ".join(SYSTEM_LABEL[s] for s in SYSTEMS),
      "| feedback tests:", FEEDBACK_TESTS)
