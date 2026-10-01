# --- LiveCodeBench sanity check + re-grade with a longer time limit (CPU only, no model calls) ---
# Why: the grader gives each program LCB_TIMEOUT_S seconds for the WHOLE test suite. LiveCodeBench
# private suites can hold dozens of large cases, and every stdin case starts a fresh Python process,
# so a correct but ordinary-speed program can run out of time and be graded FAIL. The official
# LiveCodeBench harness limits each case, not the whole suite. This cell measures how often that
# happened and re-grades every saved program with per-case limits only.
REGRADE_SUITE_TIMEOUT_S = 900    # effectively "no suite cap"; each stdin case still has 10 s
REGRADE_BENCH = "livecodebench"

if REGRADE_BENCH in M1_RESULTS:
    R = M1_RESULTS[REGRADE_BENCH]
    df = R["df"]
    probs = {p["task_id"]: p for p in BENCH_PROBLEMS[REGRADE_BENCH][0]}
    base = df[df.system == "baseline"].copy()
    base["mode"] = base.task_id.map(lambda t: probs[t]["test_mode"])
    base["difficulty"] = base.task_id.map(lambda t: probs[t].get("difficulty"))
    base["n_hidden"] = base.task_id.map(lambda t: len(probs[t]["test_list"]))

    print("1) Baseline pass@1 by problem type and difficulty")
    print((base.groupby(["mode", "difficulty"]).final_hidden_pass
           .agg(["size", "mean"]).rename(columns={"size": "n", "mean": "pass@1"})
           .assign(**{"pass@1": lambda x: (100 * x["pass@1"]).round(1)})).to_string(), "\n")

    print("2) Why baseline drafts failed")
    print(base[~base.final_hidden_pass.astype(bool)].groupby(["mode", "failure_type"]).size().to_string(), "\n")

    gen = pd.DataFrame(R["calls"])
    gen = gen[(gen.role == "generator") & ~gen.cached.astype(bool)]
    cut = int((gen.completion_tokens >= NUM_PREDICT_LCB - 5).sum())
    print("3) First drafts cut off at NUM_PREDICT_LCB = %d tokens: %d of %d" % (NUM_PREDICT_LCB, cut, len(gen)))
    print("   Public tests passed but hidden failed (baseline): %d\n"
          % int(((base.draft_public_pass == True) & ~base.final_hidden_pass.astype(bool)).sum()))

    # 4) re-grade every saved program without the whole-suite cap
    uids = set(df.run_uid)
    traces = {(t["task_id"], t["system"]): t for t in _read_jsonl(os.path.join(R["outdir"], "traces.jsonl"))
              if t.get("run_uid") in uids}
    def regrade(p, code):
        res = run_in_sandbox(code, p["test_list"], p.get("test_setup_code", ""),
                             timeout=REGRADE_SUITE_TIMEOUT_S, mode=p["test_mode"],
                             stop_on_fail=True, mem_gb=p.get("mem_gb", 3))
        return all_passed(res)
    rows, memo = [], {}
    t0 = time.time()
    for (tid, s), tr in tqdm(traces.items(), desc="re-grading"):
        p = probs[tid]
        code = tr.get("final_code") or tr["initial_code"]
        key = hashlib.sha256((str(tid) + code).encode()).hexdigest()
        old = bool(df[(df.task_id == tid) & (df.system == s)].final_hidden_pass.iloc[0])
        if old:                       # a pass stays a pass with a longer limit: no need to re-run
            rows.append({"task_id": tid, "system": s, "old_pass": True, "new_pass": True})
            continue
        if key not in memo:
            memo[key] = regrade(p, code)
        rows.append({"task_id": tid, "system": s, "old_pass": old, "new_pass": memo[key]})
    REGRADE = pd.DataFrame(rows)
    REGRADE.to_csv(os.path.join(M1_OUTPUT_DIR, "livecodebench_regrade.csv"), index=False)
    flipped = REGRADE[REGRADE.old_pass != REGRADE.new_pass]
    print("4) Re-graded the failed ones of %d programs in %.1f min without the %d s whole-suite cap"
          % (len(REGRADE), (time.time() - t0) / 60, LCB_TIMEOUT_S))
    print("   programs whose verdict changed: %d (FAIL->PASS %d, PASS->FAIL %d)"
          % (len(flipped), int((~flipped.old_pass & flipped.new_pass).sum()),
             int((flipped.old_pass & ~flipped.new_pass).sum())))
    tab = (REGRADE.groupby("system")[["old_pass", "new_pass"]].mean() * 100).round(1)
    tab.index = [SYSTEM_LABEL[s] for s in tab.index]
    print("\n   pass@1 (%%) - graded with the %d s suite cap vs per-case limits only:" % LCB_TIMEOUT_S)
    print(tab.rename(columns={"old_pass": "suite cap", "new_pass": "per-case only"}).to_string())
else:
    print("No LiveCodeBench results loaded.")
