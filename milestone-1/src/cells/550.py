# --- Figure 9 : what ONE call of each agent costs (fresh calls only) ----------
PER_CALL = (CALLS[~CALLS.cached & ~CALLS.remote].groupby(["benchmark", "role"])
            [["prompt_tokens", "completion_tokens", "gpu_seconds", "joules"]].mean().reset_index())
panels = [("prompt_tokens", "Prompt tokens / call", "{:.0f}"),
          ("completion_tokens", "Completion tokens / call", "{:.0f}"),
          ("gpu_seconds", "GPU-seconds / call", "{:.2f}"),
          ("joules", "Joules / call", "{:.0f}")]
fig, axes = plt.subplots(2, 2, figsize=(4 + 3.2 * len(BENCH_TAGS), 8.4))
nr = len(used_roles)
w = 0.8 / nr
x = np.arange(len(BENCH_TAGS))
for ax, (col, title, fmt) in zip(axes.ravel(), panels):
    for k, role in enumerate(used_roles):
        vals = []
        for tag in BENCH_TAGS:
            r = PER_CALL[(PER_CALL.benchmark == tag) & (PER_CALL.role == role)]
            vals.append(float(r[col].iloc[0]) if len(r) else np.nan)
        bars = ax.bar(x + (k - (nr - 1) / 2) * w, np.nan_to_num(vals), width=w * 0.92,
                      color=ROLE_COLOR[role], edgecolor="white", linewidth=0.6)
        bars._m1_vals = vals
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    for c in ax.containers:
        label_bars(ax, c, fmt, c._m1_vals)
    ax.set_xticks(x); ax.set_xticklabels(BENCH_LABELS)
    ax.set_title(title, color=INK, loc="left"); ax.grid(axis="x", visible=False)
fig.legend(handles=[Patch(facecolor=ROLE_COLOR[r], label=ROLE_LABEL[r]) for r in used_roles],
           loc="upper left", bbox_to_anchor=(0.045, 0.95), ncol=nr, fontsize=9)
fig.suptitle("What one call of each agent costs", x=0.05, ha="left", fontsize=12.5, color=INK, y=1.0)
fig.text(0.05, 0.957, "Averages over calls actually sent to the GPU; cached replays of the shared "
         "first draft are excluded.", ha="left", fontsize=9, color=INK_2)
fig.subplots_adjust(top=0.87, hspace=0.3, wspace=0.18)
save(fig, "fig_09_cost_per_call.png")
