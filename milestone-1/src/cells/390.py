from math import comb

def mcnemar_exact(b01, b10):
    n = b01 + b10
    if n == 0:
        return 1.0
    k = min(b01, b10)
    return min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) * (0.5 ** n))


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, same order as the input."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj, running = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


COMPARISONS = [(a, b) for a, b in COMPARISONS if a in SYSTEMS and b in SYSTEMS]   # set in section 3

srows = []
for tag in BENCH_TAGS:
    wide = ALL[ALL.benchmark == tag].pivot(index="task_id", columns="system",
                                          values="final_hidden_pass").dropna()
    for a_sys, b_sys in COMPARISONS:
        a = wide[a_sys].astype(bool); b = wide[b_sys].astype(bool)
        b11 = int((a & b).sum()); b10 = int((a & ~b).sum())
        b01 = int((~a & b).sum()); b00 = int((~a & ~b).sum())
        n = len(wide)
        srows.append({"benchmark": M1_RESULTS[tag]["label"], "tag": tag,
                      "comparison": "%s -> %s" % (SYSTEM_LABEL[a_sys], SYSTEM_LABEL[b_sys]),
                      "a": a_sys, "b": b_sys, "n": n,
                      "a_pass": b11 + b10, "b_pass": b11 + b01,
                      "gains": b01, "regressions": b10, "b11": b11, "b00": b00,
                      "delta_pp": 100.0 * (b01 - b10) / n if n else float("nan"),
                      "p_exact": mcnemar_exact(b01, b10),
                      "do_no_harm_pass": b10 == 0, "powered_n45": n >= 45})

M1_STATS = pd.DataFrame(srows)
M1_STATS["p_holm"] = holm(list(M1_STATS["p_exact"])) if len(M1_STATS) else []
M1_STATS.to_csv(os.path.join(M1_OUTPUT_DIR, "stats.csv"), index=False)

print("Exact McNemar on the hidden-suite outcome; Holm correction across all %d tests.\n"
      % len(M1_STATS))
for r in M1_STATS.to_dict("records"):
    print("%-14s %-38s %3d/%-3d -> %3d/%-3d  %+5.1f pp  gains=%3d  regressions=%3d  "
          "p=%.4f  p_holm=%.4f  %s%s"
          % (r["benchmark"], r["comparison"], r["a_pass"], r["n"], r["b_pass"], r["n"],
             r["delta_pp"], r["gains"], r["regressions"], r["p_exact"], r["p_holm"],
             "significant" if r["p_holm"] < 0.05 else "n.s.",
             "" if r["do_no_harm_pass"] else "   [do-no-harm FAILED]"))
M1_STATS.round(4)
