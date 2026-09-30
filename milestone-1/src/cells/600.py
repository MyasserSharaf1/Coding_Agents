# --- Figure 12 : Spearman correlations per benchmark (repair systems) ---------
from matplotlib.colors import LinearSegmentedColormap

VARS = [("calls_total", "calls"), ("calls_plus_exec", "calls + runs"),
        ("prompt_tokens", "prompt tok"), ("completion_tokens", "completion tok"),
        ("gpu_seconds", "GPU-s"), ("joules", "joules")]
CORRS = {}
for tag in BENCH_TAGS:
    d = REP[REP.benchmark == tag][[v for v, _ in VARS]]
    CORRS[tag] = d.corr(method="spearman")
    CORRS[tag].to_csv(os.path.join(M1_OUTPUT_DIR, "spearman_%s.csv" % tag))

allv = np.concatenate([c.values[np.tril_indices(len(VARS), -1)] for c in CORRS.values()])
allv = allv[np.isfinite(allv)]
if len(allv) and allv.min() >= 0:           # one-sided data -> sequential blue ramp
    cmap = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#86b6ef", "#2a78d6", "#104281"])
    vmin, vmax, cbl = 0, 1, "Spearman rho"
else:                                       # signs present -> diverging, grey midpoint
    cmap = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])
    vmin, vmax, cbl = -1, 1, "Spearman rho"

k = len(VARS)
labs = [l for _, l in VARS]
fig, axes = plt.subplots(1, len(BENCH_TAGS), figsize=(4.9 * len(BENCH_TAGS), 4.9), squeeze=False)
for ax, tag in zip(axes[0], BENCH_TAGS):
    vals = CORRS[tag].values
    mask = np.triu(np.ones((k, k), dtype=bool))
    im = ax.imshow(np.ma.array(vals, mask=mask)[1:, :-1], cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(k - 1)); ax.set_xticklabels(labs[:-1], rotation=35, ha="right", fontsize=8.5)
    ax.set_yticks(range(k - 1)); ax.set_yticklabels(labs[1:], fontsize=8.5)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks(np.arange(-0.5, k - 1, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, k - 1, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2); ax.tick_params(which="minor", length=0)
    for i in range(1, k):
        for j in range(i):
            v = vals[i, j]
            if v == v:
                ax.text(j, i - 1, "%.2f" % v, ha="center", va="center", fontsize=8.5,
                        color="white" if v > 0.6 else INK)
    ax.set_title("%s (n = %d)" % (M1_RESULTS[tag]["label"], int((REP.benchmark == tag).sum())),
                 color=INK, loc="left")
cb = fig.colorbar(im, ax=list(axes[0]), fraction=0.02, pad=0.02)
cb.set_label(cbl, color=INK_2); cb.outline.set_visible(False)
fig.suptitle("Rank correlations between cost measures, repair systems", x=0.04, ha="left",
             fontsize=12.5, color=INK, y=1.03)
fig.text(0.04, 0.975, "GPU-s and joules come from the same measurement window, so their "
         "correlation is expected; compare the calls and token columns instead.",
         ha="left", fontsize=9, color=INK_2)
save(fig, "fig_12_spearman_matrix.png")
