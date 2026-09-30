# --- Figure 3 : what one solved question costs -------------------------------
fig, axes = plt.subplots(1, 2, figsize=(4 + 3.4 * len(BENCH_TAGS), 4.4))
grouped(axes[0], "gpu_s_per_solved", "GPU-seconds per solved question", "{:.1f}")
grouped(axes[1], "joules_per_solved", "Joules per solved question", "{:.0f}")
header(fig, "Cost per solved question",
       "Total cost divided by questions solved. Lower is a better exchange rate; this is not a "
       "matched-budget comparison.", SYS_HANDLES)
fig.subplots_adjust(wspace=0.22)
save(fig, "fig_03_cost_per_solved.png")
