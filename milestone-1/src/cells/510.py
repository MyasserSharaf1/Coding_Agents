# --- Figure 7 : what the test agent actually did ------------------------------
if len(TA_TABLE):
    fig, axes = plt.subplots(1, 2, figsize=(4 + 3.2 * len(TA_TABLE), 4.3))
    x = np.arange(len(TA_TABLE))
    ticks = ["%s\n(%d with tests)" % (r["benchmark"], r["questions_with_generated_tests"])
             for r in TA_TABLE.to_dict("records")]

    ax = axes[0]
    g = ax.bar(x - 0.19, TA_TABLE["caught_real_bug"], width=0.36, color=C_GOOD, linewidth=0)
    b = ax.bar(x + 0.19, TA_TABLE["false_alarm_on_correct_draft"], width=0.36,
               color=C_CRITICAL, linewidth=0)
    top = max(1, int(TA_TABLE[["caught_real_bug", "false_alarm_on_correct_draft"]].values.max()))
    ax.set_ylim(0, top * 1.3)
    label_bars(ax, g, "{:.0f}"); label_bars(ax, b, "{:.0f}")
    ax.set_xticks(x); ax.set_xticklabels(ticks, fontsize=8.5)
    ax.set_ylabel("questions"); ax.grid(axis="x", visible=False)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title("Generated tests flagged the first draft", color=INK, loc="left")
    ax.legend(handles=[Patch(facecolor=C_GOOD, label="caught a bug the public tests missed"),
                       Patch(facecolor=C_CRITICAL, label="false alarm on a correct draft")],
              loc="upper left", fontsize=8.5)

    ax = axes[1]
    v = TA_TABLE["valid_tests_%"].to_numpy(float)
    bars = ax.bar(x, np.nan_to_num(v), width=0.5, color=SYS_COLOR[TA_TABLE["system"].iloc[0]],
                  hatch="///", edgecolor="white", linewidth=0.6)
    ax.set_ylim(0, 110)
    for xi, val, n in zip(x, v, TA_TABLE["tests_checked_vs_reference"]):
        ax.text(xi, (0 if val != val else val) + 2,
                "no reference" if val != val else "%.0f%%\nof %d" % (val, n),
                ha="center", va="bottom", fontsize=8, color=INK_2)
    ax.set_xticks(x); ax.set_xticklabels(ticks, fontsize=8.5)
    ax.set_ylabel("% of generated tests"); ax.grid(axis="x", visible=False)
    ax.set_title("Generated tests that are correct", color=INK, loc="left")

    fig.suptitle("Test agent diagnostics", x=0.05, ha="left", fontsize=12.5, color=INK, y=1.08)
    fig.text(0.05, 1.005, "A generated test is correct when the benchmark's reference solution "
             "passes it. LiveCodeBench has no reference, and its stdin problems get no tests.",
             ha="left", fontsize=9, color=INK_2)
    fig.subplots_adjust(wspace=0.25)
    save(fig, "fig_07_test_agent.png")
