rng_boot = np.random.default_rng(SEED)
FRESH = CALLS[~CALLS.cached & ~CALLS.remote].copy()   # local GPU calls only
FRESH["prompt_k"] = FRESH.prompt_tokens / 1000.0
FRESH["completion_k"] = FRESH.completion_tokens / 1000.0


def ols(X, y):
    """Least squares with intercept -> (coefs incl. intercept, R^2)."""
    X = np.column_stack([np.ones(len(X))] + [np.asarray(c, float) for c in X.T])
    y = np.asarray(y, float)
    ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[ok], y[ok]
    if len(y) < X.shape[1] + 2 or np.ptp(y) == 0:
        return np.full(X.shape[1], np.nan), np.nan
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    r2 = 1 - (resid ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return beta, float(r2)


# ---- (A) per-call cost model -------------------------------------------------
cm_rows = []
for tag in BENCH_TAGS + ["__pooled__"]:
    d = FRESH if tag == "__pooled__" else FRESH[FRESH.benchmark == tag]
    for target in ["gpu_seconds", "joules"]:
        beta2, r2_two = ols(d[["prompt_k", "completion_k"]].to_numpy(), d[target])
        _, r2_total = ols((d.prompt_k + d.completion_k).to_numpy()[:, None], d[target])
        cm_rows.append({
            "benchmark": "all (pooled)" if tag == "__pooled__" else M1_RESULTS[tag]["label"],
            "target": target, "calls": len(d),
            "intercept_per_call": beta2[0],
            "per_1k_prompt_tokens": beta2[1], "per_1k_completion_tokens": beta2[2],
            "completion_over_prompt": beta2[2] / beta2[1] if beta2[1] and beta2[1] > 0 else np.nan,
            "r2_prompt+completion": r2_two, "r2_total_tokens": r2_total})
COST_MODEL = pd.DataFrame(cm_rows)
COST_MODEL.to_csv(os.path.join(M1_OUTPUT_DIR, "cost_model_per_call.csv"), index=False)

# ---- (B) power by role -------------------------------------------------------
idle = M1_CONFIG["IDLE_WATTS"]
FRESH["power_w"] = FRESH.joules / FRESH.wall_seconds.replace(0, np.nan)
POWER = (FRESH.groupby(["bench_label", "role"])
         .agg(calls=("power_w", "size"), mean_power_w=("power_w", "mean"),
              p10=("power_w", lambda s: s.quantile(0.1)), p90=("power_w", lambda s: s.quantile(0.9)))
         .reset_index())
POWER["above_idle_w"] = POWER.mean_power_w - idle
POWER["per_card_w"] = POWER.mean_power_w / max(M1_CONFIG["N_GPUS"], 1)
POWER.to_csv(os.path.join(M1_OUTPUT_DIR, "power_by_role.csv"), index=False)

# ---- (C) per question, repair systems only -----------------------------------
try:
    from scipy.stats import rankdata as _rank
except Exception:
    _rank = lambda a: pd.Series(a).rank().to_numpy()

def spearman(x, y):
    x = _rank(x); y = _rank(y)
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])

def spearman_ci(x, y, B=500):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y); x, y = x[ok], y[ok]
    if len(x) < 8:
        return np.nan, np.nan, np.nan
    est = spearman(x, y)
    idx = rng_boot.integers(0, len(x), size=(B, len(x)))
    boots = [spearman(x[i], y[i]) for i in idx]
    boots = [b for b in boots if b == b]
    lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan))
    return est, float(lo), float(hi)

COUNTS = [("calls_total", "model calls"), ("calls_plus_exec", "calls + test runs"),
          ("prompt_tokens", "prompt tokens"), ("completion_tokens", "completion tokens"),
          ("total_tokens", "total tokens")]
COSTS = [("gpu_seconds", "GPU-seconds"), ("joules", "joules")]
REP = ALL[ALL.system != "baseline"]

rel_rows = []
for tag in BENCH_TAGS:
    for scope, d in [("all questions", REP[REP.benchmark == tag]),
                     ("entered repair", REP[(REP.benchmark == tag) & (REP.iterations > 0)])]:
        for cx, cxl in COUNTS:
            for cy, cyl in COSTS:
                rho, lo, hi = spearman_ci(d[cx], d[cy])
                _, r2 = ols(d[[cx]].to_numpy(float), d[cy])
                rel_rows.append({"benchmark": M1_RESULTS[tag]["label"], "tag": tag, "scope": scope,
                                 "n": len(d), "x": cx, "x_label": cxl, "y": cy, "y_label": cyl,
                                 "spearman": rho, "ci_lo": lo, "ci_hi": hi, "linear_r2": r2})
RELATIONS = pd.DataFrame(rel_rows)
RELATIONS.to_csv(os.path.join(M1_OUTPUT_DIR, "metric_relationships.csv"), index=False)

# ---- plain-language read-out -------------------------------------------------
print("(A) Per-call cost model, fresh calls only")
for r in COST_MODEL[COST_MODEL.target == "gpu_seconds"].to_dict("records"):
    print("  %-14s %5d calls: %.2f GPU-s per 1k prompt tokens, %.2f per 1k completion tokens "
          "(decode %.0fx prefill); R^2 %.2f with the split vs %.2f with total tokens"
          % (r["benchmark"], r["calls"], r["per_1k_prompt_tokens"], r["per_1k_completion_tokens"],
             r["completion_over_prompt"], r["r2_prompt+completion"], r["r2_total_tokens"]))
print("\n(B) Power while generating (all %d cards; idle %.0f W)" % (M1_CONFIG["N_GPUS"], idle))
for r in POWER.to_dict("records"):
    print("  %-14s %-26s %.0f W  (%.0f W above idle, 10-90%%: %.0f-%.0f W)"
          % (r["bench_label"], ROLE_LABEL.get(r["role"], r["role"]), r["mean_power_w"],
             r["above_idle_w"], r["p10"], r["p90"]))
print("\n(C) Per question, repair systems: what explains GPU-seconds?")
for tag in BENCH_TAGS:
    sub = RELATIONS[(RELATIONS.tag == tag) & (RELATIONS.scope == "all questions") &
                    (RELATIONS.y == "gpu_seconds")].set_index("x")
    print("  %-14s calls rho %.2f [%.2f, %.2f], R^2 %.2f | total tokens rho %.2f [%.2f, %.2f], R^2 %.2f"
          % (M1_RESULTS[tag]["label"],
             sub.loc["calls_total", "spearman"], sub.loc["calls_total", "ci_lo"],
             sub.loc["calls_total", "ci_hi"], sub.loc["calls_total", "linear_r2"],
             sub.loc["total_tokens", "spearman"], sub.loc["total_tokens", "ci_lo"],
             sub.loc["total_tokens", "ci_hi"], sub.loc["total_tokens", "linear_r2"]))
COST_MODEL.round(3)
