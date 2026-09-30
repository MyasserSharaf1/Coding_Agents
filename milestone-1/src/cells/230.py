class MultiAgentOrchestrator:
    """generate -> [test agent] -> run VISIBLE tests -> (supervisor -> repair -> run) x N.

    Visible tests = public tests (or the hidden suite when FEEDBACK_TESTS == "hidden")
    plus, for repair_tests, the test agent's asserts. The program the loop ends with is
    graded on the HIDDEN suite, which the agents never see in "public" mode."""

    def __init__(self, code_agent, test_agent, supervisor, max_retries=3,
                 use_test_agent=False, combine_supervisor_repair=False,
                 early_stop_no_progress=True, name="repair"):
        self.code_agent = code_agent
        self.test_agent = test_agent
        self.supervisor = supervisor
        self.max_retries = max_retries
        self.use_test_agent = use_test_agent
        self.combine = combine_supervisor_repair
        self.early_stop = early_stop_no_progress
        self.name = name

    def solve(self, p):
        text, entry = p["text"], p["entry_point"]
        hint = p.get("signature_hint") if USE_SIGNATURE_HINT else None
        rec = {"task_id": p["task_id"], "system": self.name, "iterations": 0,
               "executions": 0, "exec_seconds": 0.0, "stop_reason": ""}
        trace = {"task_id": p["task_id"], "system": self.name,
                 "generated_tests": [], "feedback": [], "repairs": []}

        # 1) first draft (identical prompt to the baseline -> served from the cache)
        draft, _ = self.code_agent.generate(text, entry, hint, p.get("instruction"))
        trace["initial_code"] = draft

        # 2) test agent (only for function-style problems)
        gtests = []
        if self.use_test_agent and p.get("ta_supported", True):
            gtests, _ = self.test_agent.generate(text, entry, draft, p.get("ta_call"),
                                                 TEST_AGENT_MAX_TESTS)
        trace["generated_tests"] = gtests

        base = p["public_tests"] if FEEDBACK_TESTS == "public" else p["test_list"]
        visible = list(base or []) + list(gtests)
        rec["n_visible_tests"] = len(visible)
        rec["n_generated_tests"] = len(gtests)

        def run_visible(src):
            t0 = time.time()
            res = run_suite(p, src, visible)
            rec["exec_seconds"] += time.time() - t0
            rec["executions"] += 1          # a pipeline execution: counts toward cost
            return res

        # 3) evaluate the draft on the visible tests
        if visible:
            res = run_visible(draft)
            vis_pass = all_passed(res)
            progress = res["n_passed"]
        else:                                # nothing to test against -> nothing to repair
            res, vis_pass, progress = None, True, 0
            rec["stop_reason"] = "no visible tests"
        rec["draft_visible_pass"] = vis_pass

        cur, last_err, best = draft, (first_error(res) if res else ""), progress
        can_measure_progress = len(visible) > 1

        # 4) repair loop
        while (not vis_pass) and rec["iterations"] < self.max_retries:
            if self.combine:
                fb = "Fix the code so all tests pass. Failing output:\n" + last_err
                trace["feedback"].append("[direct] " + last_err[:400])
            else:
                fb, _ = self.supervisor.review(text, cur, last_err)
                trace["feedback"].append(fb)
            cur, _ = self.code_agent.repair(text, entry, cur, fb, hint,
                                            p.get("repair_instruction"))
            trace["repairs"].append(cur)
            rec["iterations"] += 1
            res = run_visible(cur)
            vis_pass, progress = all_passed(res), res["n_passed"]
            last_err = first_error(res)
            if vis_pass:
                rec["stop_reason"] = "visible tests pass"
                break
            if self.early_stop and can_measure_progress and progress <= best:
                rec["stop_reason"] = "no progress"
                break
            best = max(best, progress)
        if not rec["stop_reason"]:
            rec["stop_reason"] = "draft passed" if rec["iterations"] == 0 else "max retries"

        rec["final_visible_pass"] = vis_pass
        rec["final_code_changed"] = cur != draft
        trace["final_code"] = cur
        return rec, trace, draft, cur, gtests
