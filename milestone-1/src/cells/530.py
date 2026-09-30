# --- cost of each agent, per question, inside each system --------------------
CALLS = pd.concat([pd.DataFrame(R["calls"]).assign(benchmark=t, bench_label=R["label"])
                   for t, R in M1_RESULTS.items() if len(R["calls"])], ignore_index=True)
CALLS["total_tokens"] = CALLS["prompt_tokens"] + CALLS["completion_tokens"]
CALLS["cached"] = CALLS["cached"].astype(bool)
CALLS["remote"] = (CALLS["remote"] if "remote" in CALLS else False)
CALLS["remote"] = CALLS["remote"].fillna(False).astype(bool)

agent_rows, check_rows = [], []
for tag in BENCH_TAGS:
    nq = M1_RESULTS[tag]["df"].task_id.nunique()
    for s in SYSTEMS:
        cs = CALLS[(CALLS.benchmark == tag) & (CALLS.system == s)]
        tot_gpu, tot_j = cs.gpu_seconds.sum(), cs.joules.sum()
        for role in ROLE_KEYS:
            a = cs[cs.role == role]
            fresh = a[~a.cached & ~a.remote]   # per-call GPU averages: real local measurements only
            n = len(a)
            agent_rows.append({
                "benchmark": M1_RESULTS[tag]["label"], "tag": tag, "system": s, "role": role,
                "agent": ROLE_LABEL[role], "calls": n, "calls_per_question": n / nq,
                "tokens_per_question": a.total_tokens.sum() / nq,
                "gpu_s_per_question": a.gpu_seconds.sum() / nq,
                "joules_per_question": a.joules.sum() / nq,
                "prompt_tokens_per_call": fresh.prompt_tokens.mean() if len(fresh) else np.nan,
                "completion_tokens_per_call": fresh.completion_tokens.mean() if len(fresh) else np.nan,
                "gpu_s_per_call": fresh.gpu_seconds.mean() if len(fresh) else np.nan,
                "joules_per_call": fresh.joules.mean() if len(fresh) else np.nan,
                "share_gpu_s_%": 100.0 * a.gpu_seconds.sum() / tot_gpu if tot_gpu else np.nan,
                "share_energy_%": 100.0 * a.joules.sum() / tot_j if tot_j else np.nan,
            })
        d = ALL[(ALL.benchmark == tag) & (ALL.system == s)]
        check_rows.append((M1_RESULTS[tag]["label"], s, cs.gpu_seconds.sum(), d.gpu_seconds.sum(),
                           len(cs), int(d.calls_total.sum())))

AGENT_TABLE = pd.DataFrame(agent_rows)
AGENT_TABLE.to_csv(os.path.join(M1_OUTPUT_DIR, "per_agent_costs.csv"), index=False)
for lab, s, g1, g2, c1, c2 in check_rows:
    ok = abs(g1 - g2) < 1e-6 and c1 == c2
    print("%-14s %-20s per-agent sums %s the per-question totals (%.1f GPU-s, %d calls)"
          % (lab, SYSTEM_LABEL[s], "match" if ok else "DO NOT MATCH", g2, c2))

AGENT_TABLE[AGENT_TABLE.calls > 0][
    ["benchmark", "system", "agent", "calls_per_question", "tokens_per_question",
     "gpu_s_per_question", "joules_per_question", "prompt_tokens_per_call",
     "completion_tokens_per_call", "gpu_s_per_call", "share_gpu_s_%", "share_energy_%"]].round(2)
