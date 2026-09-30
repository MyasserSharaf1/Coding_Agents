import numpy as np
import pandas as pd
from math import sqrt

if not M1_RESULTS:
    raise RuntimeError("No finished questions yet - the analysis below needs at least one. "
                       "Run section 16 (it resumes where it stopped).")
BENCH_TAGS = list(M1_RESULTS.keys())
BENCH_LABELS = [M1_RESULTS[t]["label"] for t in BENCH_TAGS]
ALL = pd.concat([R["df"].assign(benchmark=t, bench_label=R["label"])
                 for t, R in M1_RESULTS.items()], ignore_index=True)
ALL["calls_plus_exec"] = ALL["calls_total"] + ALL["executions"]
for c in ["remote_calls", "remote_tokens", "remote_seconds"]:
    if c not in ALL:
        ALL[c] = 0.0
    ALL[c] = ALL[c].fillna(0.0)
for c in ["final_hidden_pass", "draft_hidden_pass"]:
    ALL[c] = ALL[c].astype(bool)


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    ph = k / n
    den = 1 + z * z / n
    mid = (ph + z * z / (2 * n)) / den
    half = z * sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return (100 * (mid - half), 100 * (mid + half))


def _frac(series):
    s = series.dropna()
    return float(s.astype(bool).mean()) if len(s) else float("nan")


rows = []
for tag in BENCH_TAGS:
    for s in SYSTEMS:
        d = ALL[(ALL.benchmark == tag) & (ALL.system == s)]
        n = len(d)
        if not n:
            continue
        solved = int(d.final_hidden_pass.sum())
        lo, hi = wilson(solved, n)
        vis = d["final_visible_pass"].dropna().astype(bool)
        accepted = d.loc[vis[vis].index, "final_hidden_pass"].astype(bool)
        rows.append({
            "benchmark": M1_RESULTS[tag]["label"], "tag": tag, "system": s, "n_tasks": n,
            "pass@1": 100.0 * solved / n, "ci_lo": lo, "ci_hi": hi, "n_solved": solved,
            "calls_per_task": d.calls_total.mean(),
            "calls_plus_exec_per_task": d.calls_plus_exec.mean(),
            "executions_per_task": d.executions.mean(),
            "tokens_per_task": d.total_tokens.mean(),
            "prompt_tokens_per_task": d.prompt_tokens.mean(),
            "completion_tokens_per_task": d.completion_tokens.mean(),
            "gpu_s_per_task": d.gpu_seconds.mean(),
            # remote API calls (Kimi): counted in calls, never in GPU-s or joules
            "remote_calls_per_task": d.remote_calls.mean(),
            "remote_tokens_per_task": d.remote_tokens.mean(),
            "remote_seconds_per_task": d.remote_seconds.mean(),
            "joules_per_task": d.joules.mean(),
            "joules_above_idle_per_task": d.joules_above_idle.mean(),
            "gpu_s_per_solved": d.gpu_seconds.sum() / solved if solved else float("nan"),
            "joules_per_solved": d.joules.sum() / solved if solved else float("nan"),
            "entered_repair_%": 100.0 * (d.iterations > 0).mean(),
            "mean_iterations": d.iterations.mean(),
            # of the programs the loop ACCEPTED (visible tests pass), how many fail hidden
            "false_accept_%": (100.0 * (~accepted).mean() if len(accepted) else float("nan")),
            "calls_" + "generator": d.calls_generator.mean(),
            "calls_test_writer": d.calls_test_writer.mean(),
            "calls_supervisor": d.calls_supervisor.mean(),
            "calls_repair": d.calls_repair.mean(),
        })

M1_METRICS = pd.DataFrame(rows)
M1_METRICS.to_csv(os.path.join(M1_OUTPUT_DIR, "metrics.csv"), index=False)
pd.set_option("display.width", 240, "display.max_columns", 60)
M1_METRICS[["benchmark", "system", "n_tasks", "pass@1", "ci_lo", "ci_hi", "calls_per_task",
            "calls_plus_exec_per_task", "tokens_per_task", "gpu_s_per_task", "joules_per_task",
            "gpu_s_per_solved", "entered_repair_%", "false_accept_%"]].round(2)
