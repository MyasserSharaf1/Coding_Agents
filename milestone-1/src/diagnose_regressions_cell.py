# --- Why did a system lose questions the baseline solved? + "keep the best candidate" (no GPU) ---
# Part 1 lists every regression (baseline passes, system fails) and how the loop ended.
# Part 2 re-scores the saved traces: instead of returning the LAST program the loop wrote,
# return the one that passed the most visible tests (ties -> the earliest, so an unimproved
# draft is kept). Same model calls, same cost - only the final choice changes.
reg_rows, best_rows = [], []
for tag, R in M1_RESULTS.items():
    df = R["df"]
    w = df.pivot(index="task_id", columns="system", values="final_hidden_pass").astype(bool)
    for arm in [s for s in SYSTEMS if s != "baseline"]:
        broke = w[w["baseline"] & ~w[arm]].index
        sub = df[(df.system == arm) & df.task_id.isin(broke)]
        for r in sub.to_dict("records"):
            reg_rows.append({"benchmark": R["label"], "system": arm, "task_id": r["task_id"],
                             "draft_visible_pass": r["draft_visible_pass"],
                             "ta_false_alarm": r.get("ta_false_alarm"),
                             "iterations": r["iterations"], "stop_reason": r["stop_reason"]})

    probs = {p["task_id"]: p for p in BENCH_PROBLEMS[tag][0]}
    uids = set(df["run_uid"])
    for t in _read_jsonl(os.path.join(R["outdir"], "traces.jsonl")):
        if t.get("run_uid") not in uids or t["system"] == "baseline":
            continue
        p = probs[t["task_id"]]
        base = p["public_tests"] if FEEDBACK_TESTS == "public" else p["test_list"]
        visible = list(base or []) + list(t.get("generated_tests") or [])
        cands = [t["initial_code"]] + list(t.get("repairs") or [])
        if visible and len(cands) > 1:
            score = []
            for c in cands:
                res = run_suite(p, c, visible)
                score.append(res["n_passed"] if res["import_ok"] and not res["timed_out"] else -1)
            best = cands[max(range(len(cands)), key=lambda i: (score[i], -i))]
        else:
            best = cands[-1]
        best_rows.append({"benchmark": R["label"], "system": t["system"], "task_id": t["task_id"],
                          "last_pass": grade_hidden(p, cands[-1]), "best_pass": grade_hidden(p, best)})

REGRESSIONS = pd.DataFrame(reg_rows)
BEST = pd.DataFrame(best_rows)
REGRESSIONS.to_csv(os.path.join(M1_OUTPUT_DIR, "regressions.csv"), index=False)
BEST.to_csv(os.path.join(M1_OUTPUT_DIR, "keep_best_rescore.csv"), index=False)

if len(REGRESSIONS):
    print("Regressions vs baseline, by how the loop ended:")
    print(REGRESSIONS.groupby(["benchmark", "system", "stop_reason"]).size().to_string(), "\n")
    print("...of which the test agent raised a false alarm on the (correct) draft:")
    print(REGRESSIONS.groupby(["benchmark", "system"])["ta_false_alarm"]
          .apply(lambda s: int(s.fillna(False).astype(bool).sum())).to_string(), "\n")
print("pass@1 (%) - return the last program vs the best program by visible tests:")
print((BEST.groupby(["benchmark", "system"])[["last_pass", "best_pass"]].mean() * 100).round(1).to_string())
