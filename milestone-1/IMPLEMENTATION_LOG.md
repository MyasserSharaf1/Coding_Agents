# M1 implementation log: repair, reviewers and the test agent (qwen3-coder:30b on Kaggle 2x T4)

Last updated: 2026-09-30. This is the running record of every design decision, change, run and result in the
Milestone 1 notebook. Add to it; do not rewrite history.

## 1. What M1 measures

Four systems answer the same questions with the same code model (`qwen3-coder:30b`, Ollama, Q4) and the same first
draft. Every answer is graded on the **hidden** test suite. Cost is recorded per call in five units.

| System (key) | What it does |
|---|---|
| `baseline` | One generation. No tests, no repair. |
| `repair` | Draft, run visible tests, then (review, repair) up to 3 rounds. The 30B reviews its own failures (self-review). |
| `repair_rev` | As `repair`, but a separate reviewer model writes the review (`qwen2.5-coder:7b`, on the same GPUs). |
| `repair_tests_rev` | As `repair_rev`, plus a test agent (the 30B) whose asserts join the visible tests. |

Paired comparisons (exact McNemar, Holm-corrected across all tests): baseline -> repair, baseline -> repair_rev,
repair -> repair_rev (does a separate reviewer help?), repair_rev -> repair_tests_rev (does the test agent help?).

Cost units: model calls; calls + test executions; tokens (prompt / completion split); GPU-seconds (server compute
time x GPUs holding the model); joules (NVML hardware energy counter over each call's wall window; idle 54.9 W for
both cards).

## 2. Current configuration (notebook v3, file `m1-run-mbpp120-he164-lcb100.ipynb`)

| Setting | Value | Why |
|---|---|---|
| Questions | MBPP+ 120 (seeded sample of 378), HumanEval+ all 164, LiveCodeBench 100 (seeded sample of 182 from 2025-01-01 on) | Run-time budget; seed 42 so every sample is a subset of the full shuffle |
| `FEEDBACK_TESTS` | `"public"` | The loop only sees public tests: MBPP's 3 original asserts, HumanEval docstring examples, LCB public cases. `"hidden"` reproduces the first M1 run |
| Reviewer | `REVIEWER_BACKEND = "local"`, `qwen2.5-coder:7b` | Free, no API limits, measured in GPU-s and joules. `"kimi"` switches to Kimi K2.6 over an API |
| `OLLAMA_MAX_LOADED_MODELS` | 2 when the local reviewer is used | 30B (~18.6 GB) + 7B (~5 GB) stay resident, so reviews never trigger reloads |
| Temperature / seed | 0.1 / 42 | Shared, reproducible first draft (prompt cache + seed) |
| `MAX_RETRIES` | 3 | Review + repair rounds |
| `EARLY_STOP_NO_PROGRESS` | True | Stop when a repair adds no passing visible tests |
| `NUM_PREDICT` / `NUM_PREDICT_LCB` / `NUM_CTX` | 512 / 2048 / 8192 | One `NUM_CTX` for all benchmarks: changing it forces a model reload |
| Timeouts | 15 s per suite (MBPP+/HE+); LCB 10 s per stdin case, 900 s suite safety cap | The old 60 s LCB suite cap failed slow-but-correct programs |
| `SESSION_BUDGET_HOURS` | 8.0 | Stop cleanly before Kaggle kills the session; rerun to resume |
| `RESUME` | True | Questions already in `rows.jsonl` are skipped |

## 3. Notebook structure (sections)

1 Environment, 2 Dependencies, 3 Configuration, 4 Kaggle preflight, 5 Ollama server, 6 Model download (+ reviewer
check that both models are resident), 7 Benchmark helpers (HumanEval public tests from docstrings), 8 Sandbox
(assert mode + stdin mode, memoised grading), 9 LLM interface (+ 9b remote reviewer client), 10 Agents, 11 Repair
orchestrator, 12 Baseline, evaluation and checkpointing, 13 NVML cost meter, 14 Metered calls, 15 Load benchmarks,
16 Run, 17 Metrics, 18 Paired statistics, 19 Test-agent diagnostics, 19b Regressions + keep-the-best re-score,
19c Walkthrough of single questions, 19d LiveCodeBench sanity check and re-grade, 20 Charts (Figs 1-7),
21 Per-agent cost (Figs 8-9), 22 Cost relationships (Figs 10-12), 23 Summary report, 24 Download.

Figures: 1 pass@1 with Wilson 95% CIs; 2 five cost units; 3 cost per solved question; 4 accuracy vs GPU-s per
benchmark; 5 paired gains/regressions with Holm p; 6 repair rounds; 7 test-agent diagnostics; 8 cost by agent;
9 cost per call; 10 per-call cost model (prompt vs completion tokens); 11 per-question calls vs tokens (Spearman
+ bootstrap CI); 12 Spearman matrices per benchmark.

Output files (in `outputs_milestone1/`): per benchmark `rows.jsonl`, `calls.jsonl`, `traces.jsonl`,
`results_long.csv`, `m1_config.json`; top level `metrics.csv`, `stats.csv`, `test_agent_diagnostics.csv`,
`regressions.csv`, `keep_best_rescore.csv`, `livecodebench_regrade.csv`, `per_agent_costs.csv`,
`cost_model_per_call.csv`, `power_by_role.csv`, `metric_relationships.csv`, `spearman_<bench>.csv`,
`stop_reasons.csv`, `summary.md`, all figures; zipped as `milestone1_results.zip`.

## 4. Change history

| Date | Version | Change | Reason |
|---|---|---|---|
| 2026-09-20 | M1 v1 | Baseline vs supervisor+repair, n = 20 per benchmark, 4-axis metering | First smoke test |
| 2026-09-25 | v2 | Full MBPP+/HumanEval+; LiveCodeBench added; checkpoint + resume; test agent as a third system; public/hidden split; calls + executions as a fifth unit | User request; v1's repair saw the grading tests |
| 2026-09-25 | v2 | Correlation section rebuilt: per-call cost model, per-benchmark Spearman with bootstrap CI, cached replays excluded, mechanical pairs reported as power; warm-up call | v1 pooled benchmarks and systems, was driven by 8 high-leverage points, summed prompt and completion tokens |
| 2026-09-25 | v2 | Resume made robust: `run_uid` ties rows to calls; rows written last as commit marker; newline repair on truncated JSONL | Tested killed-session cases in a dry run |
| 2026-09-25 | v2 | Section 19b: regressions + keep-the-best re-score from saved traces | Explain test-agent losses without GPU reruns |
| 2026-09-26 | v3 | 120 per benchmark; Kimi K2.6 reviewer arm (OpenRouter / Moonshot / NVIDIA); remote calls reported separately, not in GPU-s or J | User request |
| 2026-09-26 | v3 | Walkthrough cell (19c) | User request |
| 2026-09-26 | v3 | Default reviewer switched to local `qwen2.5-coder:7b`; Kimi kept as a switch | Free tiers cap at 50 requests/day; user wants a run that never stops |
| 2026-09-27 | v3 | Section 19d: LCB breakdown + re-grade without the suite cap | LCB baseline only 33.3% |
| 2026-09-28 | v3 | Full benchmarks; LCB suite cap 60 s -> 900 s with 10 s per case | Match the official LCB harness |
| 2026-09-30 | v3 | MBPP+ 120, HumanEval+ 164, LCB 100 | Run-time budget |

## 5. Results so far

All with `qwen3-coder:30b`, dual T4, feedback from public tests unless noted.

**M1 v1 (2026-09-20, n = 20, feedback from the grading suite: optimistic)**

| Benchmark | Baseline | Repair | Calls/q | GPU-s/q | J/q |
|---|---|---|---|---|---|
| MBPP+ | 65.0% | 80.0% | 1.0 -> 2.8 | 2.0 -> 12.8 | 152 -> 833 |
| HumanEval+ | 85.0% | 85.0% | 1.0 -> 1.9 | 6.9 -> 17.2 | 440 -> 1096 |

One repair call cost 3.0x a first-draft call in GPU-s on MBPP+ (1.6x on HumanEval+). Tokens explained ~97% of
GPU-s variance vs ~77% for call count (pooled fit; later rebuilt per benchmark).

**v2 full run (2026-09-25): baseline / repair (self-review) / repair + test agent**

| Benchmark | n | Baseline | Repair | Repair + tests |
|---|---|---|---|---|
| MBPP+ | 378 | 73.5% | 78.6% | 72.5% |
| HumanEval+ | 164 | 89.6% | 90.2% | 88.4% |

**v3, 120 per benchmark (2026-09-27): four systems**

| Benchmark | Baseline | Self-review | 7B review | 7B + tests |
|---|---|---|---|---|
| MBPP+ | 70.8% | 76.7% | 77.5% | 70.8% |
| HumanEval+ | 87.5% | 87.5% | 88.3% | 86.7% |
| LiveCodeBench | 33.3% | 37.5% | 34.2% | 33.3% (graded with the 60 s suite cap) |

**v3 full HumanEval+ (2026-09-28)**: 89.6% / 89.6% / 90.2% / 86.0% (147 / 147 / 148 / 141 of 164).

## 6. Findings and interpretations

- Repair with public-test feedback gains about 5-6 points on MBPP+; with grading-suite feedback (v1) it looked like +15.
- A separate 7B reviewer is about as good as 30B self-review on MBPP+/HumanEval+ and worse on LiveCodeBench.
- The test agent lowers accuracy on every benchmark. Mechanism: generated tests with wrong expected values reject
  correct drafts (false alarms), the loop "repairs" working code and returns the last, broken program.
- HumanEval+ is at a ceiling (17 failures left) and repair rarely triggers: docstring examples pass while the hidden
  edge cases fail (false accept, C4). 38 of 164 problems have no parseable public tests.
- Run-to-run noise is about one question (self-review 90.2% vs 89.6% on identical settings). Treat 0-1 question
  differences as ties; measure the noise floor by rerunning the baseline once.
- Thesis argument: where accuracy differences fall within noise, systems must be compared on cost; because a repair
  call costs up to 3x a first-draft call in GPU time and tokens explain GPU time far better than call counts,
  hardware-native units are needed. The noise itself is not evidence for energy measurement.

## 7. Known issues and open options (not yet applied)

- **Keep the best program** instead of the last one (19b re-scores this offline). Pre-register before using it.
- **Only repair when a public test fails**; use test-agent tests only to choose between versions.
- **Swap roles: 7B writes, 30B reviews** (strong reviewer, cheap drafts; links to C5 adaptive allocation).
- **30B checks the test agent's tests** before they can trigger a repair (targets false alarms).
- **Kimi K2.6 reviewer** via NVIDIA NIM (free trial credits) or paid OpenRouter; not in GPU-s/J.
- LCB contamination: check the printed contest-date range against the model's training cutoff.
- LCB re-grade (19d) not yet run on the 60 s-cap results.

## 8. How to run on Kaggle

1. Accelerator **GPU T4 x2**, Internet **On**. No secrets needed for the local reviewer.
2. Start from a clean `outputs_milestone1/` when the system list or grading rule changes.
3. Run all cells. If the 8 h budget stops the run: Save Version, start a new session with the saved output
   attached, run all cells again (it resumes).
4. Download `milestone1_results.zip`; send `summary.md` and the 19b / 19d outputs for review.
5. "Error displaying widget: model not found" is the progress-bar widget failing to render. Harmless.
6. Stay on T4: TPUs cannot run Ollama and have no NVML energy counter.

## 9. Claude Code on Kaggle / Colab (checked against code.claude.com docs, 2026-09-30)

- Install: `curl -fsSL https://claude.ai/install.sh | bash` (or `npm install -g @anthropic-ai/claude-code`, Node 22+).
- Needs a Pro, Max, Team, Enterprise or Console account; the free claude.ai plan does not include Claude Code.
- No browser on the VM: on your own computer run `claude setup-token`, copy the token, and in the notebook set it as
  a secret, then `export CLAUDE_CODE_OAUTH_TOKEN=...`. Or use `ANTHROPIC_API_KEY` (API billing).
- Colab has a terminal for interactive `claude`. Kaggle notebooks have no interactive terminal (as far as known); use non-interactive
  `claude -p "..."` from a cell.
