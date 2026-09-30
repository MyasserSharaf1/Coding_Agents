# --- Figure 5 : paired outcomes (gains / regressions) per comparison ----------
comps = ["%s -> %s" % (SYSTEM_LABEL[a], SYSTEM_LABEL[b]) for a, b in COMPARISONS]
SHORT = SYSTEM_SHORT
short_comps = ["%s -> %s" % (SHORT[a], SHORT[b]) for a, b in COMPARISONS]
fig, axes = plt.subplots(1, len(BENCH_TAGS), figsize=(1.6 * len(COMPARISONS) * len(BENCH_TAGS) + 1.0, 4.6),
                         squeeze=False, sharey=True)
ymax = max(1, int(M1_STATS[["gains", "regressions"]].values.max()))
for ax, tag in zip(axes[0], BENCH_TAGS):
    st = M1_STATS[M1_STATS.tag == tag].set_index("comparison").reindex(comps)
    x = np.arange(len(comps))
    g = ax.bar(x - 0.19, st["gains"], width=0.36, color=C_GOOD, linewidth=0)
    r = ax.bar(x + 0.19, st["regressions"], width=0.36, color=C_CRITICAL, linewidth=0)
    ax.set_ylim(0, ymax * 1.45)
    label_bars(ax, g, "{:.0f}"); label_bars(ax, r, "{:.0f}")
    for xi, (_, row) in zip(x, st.iterrows()):
        ax.text(xi, ymax * 1.4, "p=%.3f\n%s" % (row["p_holm"],
                "no harm" if row["do_no_harm_pass"] else "harm %d" % row["regressions"]),
                ha="center", va="top", fontsize=7.5,
                color=INK_2 if row["do_no_harm_pass"] else C_CRITICAL)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace(" -> ", " ->\n") for c in short_comps], fontsize=8.5)
    ax.set_title("%s (n = %d)" % (M1_RESULTS[tag]["label"], int(st["n"].iloc[0])),
                 color=INK, loc="left")
    ax.grid(axis="x", visible=False)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
axes[0][0].set_ylabel("questions")
header(fig, "Paired outcomes on the hidden suite",
       "A -> B on the same questions (%s). Exact McNemar, Holm-corrected." % ", ".join(
           "%s = %s" % (SYSTEM_SHORT[s], SYSTEM_LABEL[s]) for s in SYSTEMS),
       [Patch(facecolor=C_GOOD, label="gained (A fails, B passes)"),
        Patch(facecolor=C_CRITICAL, label="regressed (A passes, B fails)")], x=0.04)
fig.subplots_adjust(wspace=0.12)
save(fig, "fig_05_paired_outcomes.png")
