# --- Walkthrough: follow single questions through every system ----------------------
# Prints what each agent actually wrote: the draft, the test agent's asserts, each review
# (and who wrote it), each repair, and after every round how many visible tests pass and
# whether the hidden suite passes. Re-executes saved programs only - no model calls.
WALKTHROUGH_TASKS     = None   # e.g. {"mbpp_plus": [11, 57], "livecodebench": ["LCB/abc123"]}
WALKTHROUGH_PER_BENCH = 2      # questions per benchmark when picking automatically
SHOW_CHARS            = 900    # truncate long code / feedback

def _clip(s, n=SHOW_CHARS):
    s = (s or "").rstrip()
    return s if len(s) <= n else s[:n] + "\n   ... [%d more characters]" % (len(s) - n)

def _indent(s, pad="      "):
    return "\n".join(pad + ln for ln in (s or "").splitlines())

def _pick(tag):
    """Most informative questions first: the reviewer fixed what self-review did not, a system broke a
    correct baseline, the test agent raised a false alarm, then any question that entered repair."""
    df = M1_RESULTS[tag]["df"]
    w = df.pivot(index="task_id", columns="system", values="final_hidden_pass").astype(bool)
    picks = []
    def add(ids):
        for t in ids:
            if t not in picks and len(picks) < WALKTHROUGH_PER_BENCH:
                picks.append(t)
    for a, b in COMPARISONS:
        if a != "baseline":
            add(w[~w[a] & w[b]].index)              # b gained over a
    for s in SYSTEMS:
        if s != "baseline":
            add(w[w["baseline"] & ~w[s]].index)     # s broke a correct baseline
    add(df[df.ta_false_alarm.fillna(False).astype(bool)].task_id)
    add(df[df.iterations > 0].task_id)
    return picks

def walkthrough(tag, task_id):
    R = M1_RESULTS[tag]
    df = R["df"]
    p = {q["task_id"]: q for q in BENCH_PROBLEMS[tag][0]}[task_id]
    uids = set(df[df.task_id == task_id].run_uid)
    traces = {t["system"]: t for t in _read_jsonl(os.path.join(R["outdir"], "traces.jsonl"))
              if t["task_id"] == task_id and t.get("run_uid") in uids}
    print("=" * 100)
    print("%s  |  %s" % (R["label"], task_id))
    print("=" * 100)
    print("PROBLEM:\n" + _indent(_clip(p["text"], 600)))
    pub = p.get("public_tests") or []
    print("PUBLIC TESTS the loop can see: %d" % len(pub))
    for t in pub[:3]:
        print("      " + (_clip(t, 160) if isinstance(t, str) else "stdin case: %r" % _clip(t["input"], 80)))
    base_tr = traces.get("baseline") or next(iter(traces.values()))
    print("\nFIRST DRAFT (shared by every system) - hidden suite: %s"
          % ("PASS" if grade_hidden(p, base_tr["initial_code"]) else "FAIL"))
    print(_indent(_clip(base_tr["initial_code"])))

    for s in SYSTEMS:
        row = df[(df.task_id == task_id) & (df.system == s)].iloc[0]
        tr = traces.get(s, {})
        print("\n--- %s: hidden %s | %d calls (%d remote) | %.1f GPU-s | %.0f J | %s"
              % (SYSTEM_LABEL[s], "PASS" if row.final_hidden_pass else "FAIL", row.calls_total,
                 int(row.get("remote_calls", 0) or 0), row.gpu_seconds, row.joules or 0,
                 row.stop_reason))
        if s == "baseline":
            continue
        gen = tr.get("generated_tests") or []
        if gen:
            print("   test agent wrote %d asserts%s:" % (len(gen),
                  "" if row.ta_n_valid != row.ta_n_valid else
                  " (%d correct against the reference)" % row.ta_n_valid))
            for t in gen:
                print("      " + _clip(t, 160))
            if row.ta_false_alarm:
                print("   !! false alarm: these tests rejected a draft that passes the hidden suite")
        base = p["public_tests"] if FEEDBACK_TESTS == "public" else p["test_list"]
        visible = list(base or []) + list(gen)
        reviewer = (REVIEWER_NAME if SUPERVISOR_BY_SYSTEM.get(s) == "reviewer"
                    else "self-review (%s)" % MODEL_NAME)
        for i, (fb, code) in enumerate(zip(tr.get("feedback", []), tr.get("repairs", [])), 1):
            res = run_suite(p, code, visible) if visible else None
            print("   round %d - review by %s:" % (i, reviewer))
            print(_indent(_clip(fb, 600), "        "))
            print("   round %d - repaired program: visible %s | hidden %s"
                  % (i, ("%d/%d" % (res["n_passed"], res["n_tests"])) if res else "-",
                     "PASS" if grade_hidden(p, code) else "FAIL"))
            print(_indent(_clip(code), "        "))

for tag in M1_RESULTS:
    ids = (WALKTHROUGH_TASKS or {}).get(tag) if WALKTHROUGH_TASKS else _pick(tag)
    for task_id in ids or []:
        walkthrough(tag, task_id)
        print()
