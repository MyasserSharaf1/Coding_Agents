# --- Figure 6 : how many repair rounds each question used ---------------------
arms = [s for s in SYSTEMS if s != "baseline"]
fig, axes = plt.subplots(1, len(BENCH_TAGS), figsize=(4.6 * len(BENCH_TAGS), 4.2),
                         squeeze=False, sharey=True)
rounds = list(range(MAX_RETRIES + 1))
w = 0.8 / max(len(arms), 1)
for ax, tag in zip(axes[0], BENCH_TAGS):
    for k, s in enumerate(arms):
        it = ALL[(ALL.benchmark == tag) & (ALL.system == s)].iterations
        share = [100.0 * (it == r).mean() for r in rounds]
        bars = ax.bar(np.array(rounds) + (k - (len(arms) - 1) / 2) * w, share, width=w * 0.92,
                      color=SYS_COLOR[s], hatch=SYS_HATCH[s], edgecolor="white", linewidth=0.6)
    ax.set_ylim(0, 115)
    for c in ax.containers:
        label_bars(ax, c, "{:.0f}")
    ax.set_xticks(rounds)
    ax.set_xlabel("repair rounds used")
    ax.set_title(M1_RESULTS[tag]["label"], color=INK, loc="left")
    ax.grid(axis="x", visible=False)
axes[0][0].set_ylabel("% of questions")
header(fig, "Repair rounds per question",
       "0 = the draft passed its visible tests. Each round costs a review call and a repair call.",
       [h for h, s in zip(SYS_HANDLES, SYSTEMS) if s != "baseline"], x=0.04)
fig.subplots_adjust(wspace=0.1)
save(fig, "fig_06_repair_rounds.png")

# how each loop ended (table)
STOP_TABLE = (ALL[ALL.system != "baseline"]
              .groupby(["bench_label", "system", "stop_reason"]).size()
              .unstack(fill_value=0))
STOP_TABLE.to_csv(os.path.join(M1_OUTPUT_DIR, "stop_reasons.csv"))
STOP_TABLE
