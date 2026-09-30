ta_rows = []
for (ta_sys, ctrl_sys), tag in [(kv, t) for kv in TEST_AGENT_SYSTEMS.items() for t in BENCH_TAGS]:
    d = ALL[(ALL.benchmark == tag) & (ALL.system == ta_sys)]
    if not len(d):
        continue
    used = d[d.n_generated_tests > 0]
    ref = used[used.ta_n_valid.notna()]
    n_tests = int(ref.n_generated_tests.sum())
    ta_rows.append({
        "benchmark": M1_RESULTS[tag]["label"], "system": ta_sys, "control": ctrl_sys,
        "questions": len(d),
        "questions_with_generated_tests": len(used),
        "tests_per_question": used.n_generated_tests.mean() if len(used) else float("nan"),
        # validity: a generated test the REFERENCE solution passes is a correct test
        "tests_checked_vs_reference": n_tests,
        "valid_tests_%": 100.0 * ref.ta_n_valid.sum() / n_tests if n_tests else float("nan"),
        "flagged_draft": int(used.ta_flags_draft.fillna(False).astype(bool).sum()),
        "caught_real_bug": int(used.ta_caught.fillna(False).astype(bool).sum()),
        "false_alarm_on_correct_draft": int(used.ta_false_alarm.fillna(False).astype(bool).sum()),
        "test_writer_calls_per_q": d.calls_test_writer.mean(),
        "extra_gpu_s_vs_repair": (d.gpu_seconds.mean() -
                                  ALL[(ALL.benchmark == tag) & (ALL.system == ctrl_sys)].gpu_seconds.mean()),
        "extra_joules_vs_repair": (d.joules.mean() -
                                   ALL[(ALL.benchmark == tag) & (ALL.system == ctrl_sys)].joules.mean()),
    })
TA_TABLE = pd.DataFrame(ta_rows)
TA_TABLE.to_csv(os.path.join(M1_OUTPUT_DIR, "test_agent_diagnostics.csv"), index=False)

for r in TA_TABLE.to_dict("records"):
    print("%s [%s]: test agent wrote tests for %d/%d questions (%.1f per question)"
          % (r["benchmark"], SYSTEM_LABEL[r["system"]], r["questions_with_generated_tests"], r["questions"],
             r["tests_per_question"]))
    if r["tests_checked_vs_reference"]:
        print("   %.0f%% of %d generated tests are correct (the reference solution passes them)"
              % (r["valid_tests_%"], r["tests_checked_vs_reference"]))
    else:
        print("   no reference solution -> test validity not measurable on this benchmark")
    print("   flagged the first draft on %d questions: %d real bugs the public tests missed, "
          "%d false alarms on a correct draft"
          % (r["flagged_draft"], r["caught_real_bug"], r["false_alarm_on_correct_draft"]))
    print("   extra cost vs %s: %+.2f GPU-s and %+.0f J per question"
          % (SYSTEM_LABEL[r["control"]], r["extra_gpu_s_vs_repair"], r["extra_joules_vs_repair"]))
TA_TABLE.round(2)
