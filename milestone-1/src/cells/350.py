import uuid

M1_RESULTS = {}      # tag -> {"df", "calls", "label", "outdir", "complete", ...}
_STOPPED_FOR_BUDGET = False


def _warm_up():
    """One unmetered call per local model so a (re)load never lands inside a measured call."""
    models = [MODEL_NAME] + ([REVIEWER_MODEL] if REVIEWER_BACKEND == "local" else [])
    for m in models:
        try:
            requests.post(OLLAMA_HOST + "/api/chat", timeout=900, json={
                "model": m, "stream": False, "keep_alive": KEEP_ALIVE,
                "messages": [{"role": "user", "content": "Reply with OK"}],
                "options": {"temperature": 0.0, "num_predict": 4, "num_ctx": NUM_CTX}})
        except Exception as e:
            print("  warm-up of %s failed (continuing): %s" % (m, e))


def _n_for(tag, available):
    n = M1_N_PROBLEMS.get(tag)
    n_eff = available if n is None else min(n, available)
    if n_eff < M1_MIN_PROBLEMS and n_eff < available and not ALLOW_SMALL_N:
        raise ValueError("%s: %d questions is below the %d floor. Raise M1_N_PROBLEMS['%s'] "
                         "or set ALLOW_SMALL_N = True for a smoke test."
                         % (tag, n_eff, M1_MIN_PROBLEMS, tag))
    if n_eff < M1_MIN_PROBLEMS:
        print("  NOTE: %s has only %d questions (floor %d)." % (tag, n_eff, M1_MIN_PROBLEMS))
    return n_eff


def load_saved(tag, label):
    """Load everything already on disk for one benchmark (all sessions)."""
    outdir = os.path.join(M1_OUTPUT_DIR, tag)
    rows = _read_jsonl(os.path.join(outdir, "rows.jsonl"))
    calls = _read_jsonl(os.path.join(outdir, "calls.jsonl"))
    df = pd.DataFrame(rows)
    if len(df):
        # A question counts only when ONE run of it has a row for every system (a killed
        # session can leave a partial run behind). If a question ran twice, the later
        # complete run wins, and only that run's calls are kept.
        df["_order"] = range(len(df))
        per_run = df.groupby(["task_id", "run_uid"]).agg(n=("system", "nunique"),
                                                         last=("_order", "max")).reset_index()
        per_run = per_run[per_run.n == len(SYSTEMS)].sort_values("last")
        keep = per_run.drop_duplicates("task_id", keep="last")
        uids = set(keep.run_uid)
        df = (df[df.run_uid.isin(uids)].drop_duplicates(["task_id", "system"], keep="last")
              .drop(columns="_order"))
        calls = [c for c in calls if c.get("run_uid") in uids]
    return df.reset_index(drop=True), calls, outdir


def run_m1_benchmark(tag):
    global TIMEOUT_SECONDS, _STOPPED_FOR_BUDGET
    problems, label = BENCH_PROBLEMS[tag]
    main_set = sample_main_set(problems, _n_for(tag, len(problems)))
    outdir = os.path.join(M1_OUTPUT_DIR, tag)
    os.makedirs(outdir, exist_ok=True)

    prev, _, _ = load_saved(tag, label)
    done = set(prev["task_id"]) if (RESUME and len(prev)) else set()
    if not RESUME:
        for fn in ["rows.jsonl", "calls.jsonl", "traces.jsonl"]:
            if os.path.exists(os.path.join(outdir, fn)):
                os.remove(os.path.join(outdir, fn))
    todo = [p for p in main_set if p["task_id"] not in done]
    print("\n=== %s === %d questions (of %d available): %d done, %d to run"
          % (label, len(main_set), len(problems), len(main_set) - len(todo), len(todo)))
    with open(os.path.join(outdir, "m1_config.json"), "w") as f:
        json.dump({**EXPERIMENT_CONFIG, **M1_CONFIG, "tag": tag, "label": label,
                   "n_target": len(main_set),
                   "task_ids": [p["task_id"] for p in main_set]}, f, indent=2, default=str)

    build_systems(NUM_PREDICT_LCB if tag == "livecodebench" else NUM_PREDICT)
    _warm_up()
    CALL_LOG.clear()
    t0 = time.time()
    for p in tqdm(todo, desc="Evaluating [%s]" % label):
        if (time.time() - T_SESSION_START) / 3600.0 > SESSION_BUDGET_HOURS:
            _STOPPED_FOR_BUDGET = True
            print("  Session budget of %.2f h reached - stopping. Re-run this notebook in a new "
                  "session to resume." % SESSION_BUDGET_HOURS)
            break
        try:
            rows, traces, calls = evaluate_question(p, tag)
        except RemoteQuotaError as e:
            _STOPPED_FOR_BUDGET = True
            print("  The reviewer API stopped answering (%s).\n  Stopping here; everything so far is saved. "
                  "Resume later (or switch KIMI_MODEL / KIMI_PROVIDER)." % e)
            break
        except requests.exceptions.RequestException as e:
            print("  %s: model call failed (%s) - skipped, will retry on resume" % (p["task_id"], e))
            continue
        uid = uuid.uuid4().hex             # ties this question's rows to its calls
        for r in rows + calls + traces:
            r["run_uid"] = uid
        # rows.jsonl is written LAST: it is the commit marker for the question
        _append_jsonl(os.path.join(outdir, "calls.jsonl"), calls)
        _append_jsonl(os.path.join(outdir, "traces.jsonl"), traces)
        _append_jsonl(os.path.join(outdir, "rows.jsonl"), rows)
        CALL_LOG.clear()           # already on disk
    mins = (time.time() - t0) / 60.0

    df, calls, _ = load_saved(tag, label)
    complete = len(df) and df.task_id.nunique() >= len(main_set)
    M1_RESULTS[tag] = {"df": df, "calls": calls, "label": label, "outdir": outdir,
                       "n_target": len(main_set), "complete": bool(complete),
                       "minutes_this_session": mins}
    df.to_csv(os.path.join(outdir, "results_long.csv"), index=False)
    if len(df):
        pv = df.pivot(index="task_id", columns="system", values="final_hidden_pass").mean()
        print("  pass@1 so far (%d questions): %s   [%.1f min this session]"
              % (df.task_id.nunique(),
                 "  ".join("%s %.1f%%" % (SYSTEM_LABEL[s], 100 * pv[s]) for s in SYSTEMS if s in pv),
                 mins))
    return df


for _k in M1_BENCHMARKS:
    _tag = M1_TAGS[_k]
    if _tag not in BENCH_PROBLEMS:
        continue
    if _STOPPED_FOR_BUDGET:
        # still load what earlier sessions saved so the analysis covers it
        _df, _calls, _od = load_saved(_tag, BENCH_PROBLEMS[_tag][1])
        if len(_df):
            M1_RESULTS[_tag] = {"df": _df, "calls": _calls, "label": BENCH_PROBLEMS[_tag][1],
                                "outdir": _od, "n_target": None, "complete": False,
                                "minutes_this_session": 0.0}
        continue
    run_m1_benchmark(_tag)

M1_RESULTS = {t: R for t, R in M1_RESULTS.items() if len(R["df"])}
print("\nRuns finished ->", M1_OUTPUT_DIR)
for t, R in M1_RESULTS.items():
    print("  %-15s %4d questions%s" % (R["label"], R["df"].task_id.nunique(),
                                     "" if R["complete"] else "  (PARTIAL - resume to finish)"))
