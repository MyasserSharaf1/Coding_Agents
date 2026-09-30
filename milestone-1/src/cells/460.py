# --- Figure 2 : the five cost units, per question -----------------------------
panels = [("calls_per_task",           "Model calls / question",           "{:.2f}"),
          ("calls_plus_exec_per_task", "Calls + test runs / question",     "{:.2f}"),
          ("tokens_per_task",          "Tokens / question",                "{:.0f}"),
          ("gpu_s_per_task",           "GPU-seconds / question",           "{:.1f}"),
          ("joules_per_task",          "Joules / question (GPU cards)",    "{:.0f}")]
fig, axes = plt.subplots(2, 3, figsize=(17, 8.6))
axes = axes.ravel()
for ax, (col, title, fmt) in zip(axes, panels):
    grouped(ax, col, title, fmt)
    ax.tick_params(axis="x", labelsize=8.5)
for ax in axes[len(panels):]:
    ax.set_visible(False)
header(fig, "What each system costs, per question, in five units",
       "Same model, same first draft. The baseline runs no tests, so its calls + test runs equals its "
       "calls." + (" Kimi calls add to calls but not to tokens, GPU-s or joules."
                   if REVIEWER_BACKEND == "kimi" else ""), SYS_HANDLES, x=0.04)
fig.subplots_adjust(wspace=0.22, hspace=0.45)
save(fig, "fig_02_cost_units.png")
