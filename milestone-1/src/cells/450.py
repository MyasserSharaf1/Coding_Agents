# --- Figure 1 : correctness on the hidden suite, with 95% Wilson intervals ----
fig, ax = plt.subplots(figsize=(3.2 + 2.6 * len(BENCH_TAGS), 4.6))
grouped(ax, "pass@1", "Functional correctness (hidden tests)", "{:.1f}", "pass@1 (%)", ci=True)
ax.set_ylim(0, 105)
header(fig, "pass@1 - %s" % MODEL_NAME,
       "Whiskers: 95%% Wilson interval. Repair feedback: %s tests." % FEEDBACK_TESTS, SYS_HANDLES, x=0.07)
save(fig, "fig_01_pass_at_1.png")
