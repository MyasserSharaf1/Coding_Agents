# --- Figure 4 : accuracy-cost frontier, one panel per benchmark ---------------
fig, axes = plt.subplots(1, len(BENCH_TAGS), figsize=(4.8 * len(BENCH_TAGS), 4.4), squeeze=False)
for ax, tag in zip(axes[0], BENCH_TAGS):
    xs = [mget(tag, s, "gpu_s_per_task") for s in SYSTEMS]
    ys = [mget(tag, s, "pass@1") for s in SYSTEMS]
    lo = [y - mget(tag, s, "ci_lo") for y, s in zip(ys, SYSTEMS)]
    hi = [mget(tag, s, "ci_hi") - y for y, s in zip(ys, SYSTEMS)]
    for s, x, y, l, h in zip(SYSTEMS, xs, ys, lo, hi):
        if s != "baseline":
            ax.plot([xs[0], x], [ys[0], y], "-", color=GRID, linewidth=2, zorder=1)
        ax.errorbar([x], [y], yerr=[[l], [h]], fmt="none", ecolor=INK_2, elinewidth=1,
                    capsize=3, zorder=2)
        ax.scatter([x], [y], s=110, color=SYS_COLOR[s], zorder=3, edgecolor="white",
                   linewidth=2, marker=SYS_MARKER[s])
    ax.set_xlim(0, max(xs) * 1.25)
    ax.text(0.98, 0.03, "\n".join("%s  %+.1f pp  %+.1f GPU-s" % (SYSTEM_SHORT[s], y - ys[0], x - xs[0])
                                  for s, x, y in zip(SYSTEMS, xs, ys) if s != "baseline"),
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=INK, family="monospace",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor=GRID))
    ax.set_title("%s (n = %d)" % (M1_RESULTS[tag]["label"], M1_RESULTS[tag]["df"].task_id.nunique()),
                 color=INK, loc="left")
    ax.set_xlabel("GPU-seconds per question")
    lows = [mget(tag, q, "ci_lo") for q in SYSTEMS]
    ax.set_ylim(max(0, min(lows) - 8), 105)
axes[0][0].set_ylabel("pass@1 (%)")
handles = [plt.Line2D([], [], marker=SYS_MARKER[s], linestyle="none",
                      markerfacecolor=SYS_COLOR[s], markeredgecolor="white", markersize=9,
                      label=SYSTEM_LABEL[s]) for s in SYSTEMS]
header(fig, "Accuracy against cost",
       "Up is better, left is cheaper. Boxes give each system's change against the baseline (%s); "
       "whiskers are 95%% Wilson intervals." % ", ".join("%s = %s" % (SYSTEM_SHORT[s], SYSTEM_LABEL[s])
                                                       for s in SYSTEMS if s != "baseline"), handles, x=0.04)
fig.subplots_adjust(wspace=0.25)
save(fig, "fig_04_frontier.png")
