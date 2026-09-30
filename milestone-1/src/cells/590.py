# --- Figure 11 : per question, repair systems - calls vs tokens as cost proxies
arms = [s for s in SYSTEMS if s != "baseline"]
rng_j = np.random.default_rng(SEED)
fig, axes = plt.subplots(len(BENCH_TAGS), 2, figsize=(11, 3.7 * len(BENCH_TAGS)), squeeze=False)
for i, tag in enumerate(BENCH_TAGS):
    d = REP[REP.benchmark == tag]
    sub = RELATIONS[(RELATIONS.tag == tag) & (RELATIONS.scope == "all questions") &
                    (RELATIONS.y == "gpu_seconds")].set_index("x")
    for j, (xcol, xl) in enumerate([("calls_total", "model calls"), ("total_tokens", "tokens")]):
        ax = axes[i][j]
        for s in arms:
            ds = d[d.system == s]
            xv = ds[xcol].to_numpy(float)
            if xcol == "calls_total":
                xv = xv + rng_j.uniform(-0.15, 0.15, len(xv))
            ax.scatter(xv, ds.gpu_seconds, s=20, color=SYS_COLOR[s], alpha=0.7,
                       marker="o" if s != "repair_tests" else "D",
                       edgecolor="white", linewidth=0.5, zorder=3)
        r = sub.loc[xcol]
        ax.text(0.03, 0.97, "Spearman rho %.2f [%.2f, %.2f]\nlinear R$^2$ %.2f"
                % (r["spearman"], r["ci_lo"], r["ci_hi"], r["linear_r2"]),
                transform=ax.transAxes, ha="left", va="top", fontsize=8.5, color=INK,
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor=GRID))
        ax.set_xlabel(xl); ax.set_xlim(left=0); ax.set_ylim(bottom=0)
        if j == 0:
            ax.set_ylabel("%s\nGPU-seconds per question" % M1_RESULTS[tag]["label"])
handles = [plt.Line2D([], [], marker="o" if s != "repair_tests" else "D", linestyle="none",
                      markerfacecolor=SYS_COLOR[s], markeredgecolor="white", markersize=8,
                      label=SYSTEM_LABEL[s]) for s in arms]
fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.995, 1.02), ncol=len(arms))
fig.suptitle("Per question: call count vs tokens as a proxy for GPU time",
             x=0.05, ha="left", fontsize=12.5, color=INK, y=1.02)
fig.text(0.05, 0.99 - 0.004 * len(BENCH_TAGS), "Repair systems only, one row per benchmark, "
         "95% bootstrap intervals. Call counts are jittered.", ha="left", fontsize=9, color=INK_2)
fig.subplots_adjust(top=0.93, hspace=0.35, wspace=0.2)
save(fig, "fig_11_calls_vs_tokens.png")
