# --- Figure 8 : where each system's cost goes, per question -------------------
panels = [("calls_per_question", "Model calls / question", "{:.2f}"),
          ("tokens_per_question", "Tokens / question", "{:.0f}"),
          ("gpu_s_per_question", "GPU-seconds / question", "{:.1f}"),
          ("joules_per_question", "Joules / question", "{:.0f}")]
used_roles = [r for r in ROLE_KEYS if AGENT_TABLE[AGENT_TABLE.role == r].calls.sum() > 0]
SHORT_SYS = SYSTEM_SHORT

fig, axes = plt.subplots(len(panels), 1, figsize=(2.2 + 2.4 * len(BENCH_TAGS) * N_SYS / 3, 3.0 * len(panels)))
pos, ticks, centers = [], [], []
for bi, tag in enumerate(BENCH_TAGS):
    base = bi * (N_SYS + 1)
    for k, s in enumerate(SYSTEMS):
        pos.append((tag, s, base + k)); ticks.append(SHORT_SYS[s])
    centers.append(base + (N_SYS - 1) / 2)
for ax, (col, title, fmt) in zip(axes, panels):
    bottoms = {p: 0.0 for p in pos}
    for role in used_roles:
        vals = []
        for tag, s, xpos in pos:
            r = AGENT_TABLE[(AGENT_TABLE.tag == tag) & (AGENT_TABLE.system == s) &
                            (AGENT_TABLE.role == role)]
            v = float(r[col].iloc[0]) if len(r) else 0.0
            vals.append(0.0 if v != v else v)
        ax.bar([p[2] for p in pos], vals, bottom=[bottoms[p] for p in pos], width=0.8,
               color=ROLE_COLOR[role], edgecolor="white", linewidth=1.0)
        for p, v in zip(pos, vals):
            bottoms[p] += v
    top = max(bottoms.values()) or 1.0
    ax.set_ylim(0, top * 1.2)
    for p in pos:
        ax.text(p[2], bottoms[p] + 0.02 * top, fmt.format(bottoms[p]), ha="center",
                va="bottom", fontsize=7.5, color=INK_2)
    ax.set_xticks([p[2] for p in pos]); ax.set_xticklabels(ticks, fontsize=8)
    for c, tag in zip(centers, BENCH_TAGS):
        ax.text(c, -0.2, M1_RESULTS[tag]["label"], transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=9, color=INK)
    ax.set_title(title, color=INK, loc="left"); ax.grid(axis="x", visible=False)
fig.suptitle("Where each system's cost goes", x=0.05, ha="left", fontsize=12.5, color=INK, y=0.998)
fig.text(0.05, 0.968, ", ".join("%s = %s" % (SYSTEM_SHORT[s], SYSTEM_LABEL[s]) for s in SYSTEMS) +
         (". Kimi calls count as calls but use no local GPU, so they add no GPU-s or joules."
          if REVIEWER_BACKEND == "kimi" else ". The pink segment is the review, by whichever model wrote it."), ha="left", fontsize=9, color=INK_2)
fig.legend(handles=[Patch(facecolor=ROLE_COLOR[r], label=ROLE_LABEL[r]) for r in used_roles],
           loc="upper left", bbox_to_anchor=(0.045, 0.962), ncol=len(used_roles), fontsize=9)
fig.subplots_adjust(top=0.91, hspace=0.62)
save(fig, "fig_08_cost_by_agent.png")
