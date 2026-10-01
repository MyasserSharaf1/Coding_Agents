def _f(v, fmt):
    return "-" if v is None or (isinstance(v, float) and v != v) else fmt % v

L = ["# Milestone 1 (v3e, part %s) results - %s" % (M1_PART, MODEL_NAME), "",
     "%s - %d GPU(s) - idle %s - repair feedback: %s tests - graded on hidden tests" % (
         time.strftime("%Y-%m-%d %H:%M"), M1_CONFIG["N_GPUS"],
         _f(M1_CONFIG["IDLE_WATTS"], "%.1f W"), FEEDBACK_TESTS), ""]
for t, R in M1_RESULTS.items():
    L.append("- %s: %d questions%s" % (R["label"], R["df"].task_id.nunique(),
                                       "" if R["complete"] else " (PARTIAL)"))

L += ["", "## Correctness and cost per question", "",
      "| benchmark | system | n | pass@1 (95% CI) | calls | calls + runs | tokens | GPU-s | J | GPU-s per solved | entered repair | false accept |",
      "|---|---|---|---|---|---|---|---|---|---|---|---|"]
for r in M1_METRICS.to_dict("records"):
    L.append("| %s | %s | %d | %.1f (%.1f-%.1f) | %.2f | %.2f | %.0f | %.1f | %s | %s | %.0f%% | %s |"
             % (r["benchmark"], SYSTEM_LABEL[r["system"]], r["n_tasks"], r["pass@1"], r["ci_lo"],
                r["ci_hi"], r["calls_per_task"], r["calls_plus_exec_per_task"], r["tokens_per_task"],
                r["gpu_s_per_task"], _f(r["joules_per_task"], "%.0f"),
                _f(r["gpu_s_per_solved"], "%.1f"), r["entered_repair_%"],
                _f(r["false_accept_%"], "%.0f%%")))

if (M1_METRICS["remote_calls_per_task"] > 0).any():
    L += ["", "## Remote API reviewer calls per question - not in GPU-s or joules", "",
          "| benchmark | system | API calls | API tokens | API seconds |", "|---|---|---|---|---|"]
for r in M1_METRICS.to_dict("records"):
    if r["remote_calls_per_task"] > 0:
        L.append("| %s | %s | %.2f | %.0f | %.1f |" % (r["benchmark"], SYSTEM_LABEL[r["system"]],
                 r["remote_calls_per_task"], r["remote_tokens_per_task"], r["remote_seconds_per_task"]))

L += ["", "## Paired comparisons (exact McNemar, Holm-corrected)", "",
      "| benchmark | comparison | n | delta (pp) | gains | regressions | p | p (Holm) | do-no-harm |",
      "|---|---|---|---|---|---|---|---|---|"]
for r in M1_STATS.to_dict("records"):
    L.append("| %s | %s | %d | %+.1f | %d | %d | %.4f | %.4f | %s |"
             % (r["benchmark"], r["comparison"], r["n"], r["delta_pp"], r["gains"],
                r["regressions"], r["p_exact"], r["p_holm"],
                "pass" if r["do_no_harm_pass"] else "FAIL"))

L += ["", "## Test agent", "",
      "| benchmark | system | questions with tests | tests/question | correct tests | caught real bug | false alarm | extra GPU-s vs same system without tests |",
      "|---|---|---|---|---|---|---|---|"]
for r in TA_TABLE.to_dict("records"):
    L.append("| %s | %s | %d / %d | %s | %s | %d | %d | %+.2f |"
             % (r["benchmark"], SYSTEM_LABEL[r["system"]], r["questions_with_generated_tests"], r["questions"],
                _f(r["tests_per_question"], "%.1f"), _f(r["valid_tests_%"], "%.0f%%"),
                r["caught_real_bug"], r["false_alarm_on_correct_draft"], r["extra_gpu_s_vs_repair"]))

L += ["", "## Per-call cost model (fresh calls)", "",
      "| benchmark | target | calls | per 1k prompt tok | per 1k completion tok | R^2 split | R^2 total tokens |",
      "|---|---|---|---|---|---|---|"]
for r in COST_MODEL.to_dict("records"):
    L.append("| %s | %s | %d | %.3f | %.3f | %.2f | %.2f |"
             % (r["benchmark"], r["target"], r["calls"], r["per_1k_prompt_tokens"],
                r["per_1k_completion_tokens"], r["r2_prompt+completion"], r["r2_total_tokens"]))

L += ["", "## Per question, repair systems: Spearman rho with GPU-seconds", "",
      "| benchmark | scope | n | calls | calls + runs | total tokens |", "|---|---|---|---|---|---|"]
for (b, sc), g in RELATIONS[RELATIONS.y == "gpu_seconds"].groupby(["benchmark", "scope"], sort=False):
    gi = g.set_index("x")
    cell = lambda x: "%.2f [%.2f, %.2f]" % (gi.loc[x, "spearman"], gi.loc[x, "ci_lo"], gi.loc[x, "ci_hi"])
    L.append("| %s | %s | %d | %s | %s | %s |" % (b, sc, int(gi["n"].iloc[0]), cell("calls_total"),
                                                   cell("calls_plus_exec"), cell("total_tokens")))

SUMMARY_MD = "\n".join(L)
with open(os.path.join(M1_OUTPUT_DIR, "summary.md"), "w") as f:
    f.write(SUMMARY_MD)
print(SUMMARY_MD)
