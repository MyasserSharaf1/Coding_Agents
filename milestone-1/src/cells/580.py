# --- Figure 10 : the per-call cost model --------------------------------------
fig, axes = plt.subplots(2, len(BENCH_TAGS), figsize=(4.8 * len(BENCH_TAGS), 8.2), squeeze=False)
for j, tag in enumerate(BENCH_TAGS):
    d = FRESH[FRESH.benchmark == tag]
    cm = COST_MODEL[(COST_MODEL.benchmark == M1_RESULTS[tag]["label"]) &
                    (COST_MODEL.target == "gpu_seconds")].iloc[0]
    for i, (xcol, xl) in enumerate([("completion_tokens", "completion tokens per call"),
                                    ("prompt_tokens", "prompt tokens per call")]):
        ax = axes[i][j]
        for role in used_roles:
            s = d[d.role == role]
            ax.scatter(s[xcol], s.gpu_seconds, s=16, color=ROLE_COLOR[role], alpha=0.75,
                       edgecolor="white", linewidth=0.4, zorder=3)
        ax.set_xlabel(xl)
        ax.set_xlim(left=0); ax.set_ylim(bottom=0)
        if j == 0:
            ax.set_ylabel("GPU-seconds per call")
        if i == 0:
            ax.set_title("%s (%d calls)" % (M1_RESULTS[tag]["label"], len(d)), color=INK, loc="left")
            ax.text(0.03, 0.97, "per 1k completion tokens: %.2f GPU-s\nper 1k prompt tokens: %.2f GPU-s\n"
                    "R$^2$ = %.2f (split)  vs  %.2f (total)"
                    % (cm["per_1k_completion_tokens"], cm["per_1k_prompt_tokens"],
                       cm["r2_prompt+completion"], cm["r2_total_tokens"]),
                    transform=ax.transAxes, ha="left", va="top", fontsize=8.5, color=INK,
                    bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor=GRID))
fig.suptitle("What one call costs: prompt tokens vs completion tokens",
             x=0.04, ha="left", fontsize=12.5, color=INK, y=1.0)
fig.text(0.04, 0.957, "One point per call actually sent to the GPU. Coefficients come from one "
         "fit with both token counts, per benchmark.", ha="left", fontsize=9, color=INK_2)
fig.legend(handles=[Patch(facecolor=ROLE_COLOR[r], label=ROLE_LABEL[r]) for r in used_roles],
           loc="upper left", bbox_to_anchor=(0.035, 0.95), ncol=len(used_roles), fontsize=9)
fig.subplots_adjust(top=0.87, hspace=0.3, wspace=0.22)
save(fig, "fig_10_cost_model_per_call.png")
