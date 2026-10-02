# M1 implementation log: repair, reviewers and the test agent (qwen3-coder:30b on Kaggle 2x T4)

Last updated: 2026-10-01. This is the running record of every design decision, change, run and result in the
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

## 2. Current configuration (v3e: two notebooks, `07_m1-v3e-part-a-mbpp-humaneval.ipynb` and `08_m1-v3e-part-b-livecodebench.ipynb`)

M1 is split into two notebooks built from the same cells; `M1_PART` is the only difference. Part A
(`"evalplus"`) runs MBPP+ and HumanEval+ into `outputs_m1_evalplus/`; Part B (`"livecodebench"`) runs LiveCodeBench
into `outputs_m1_livecodebench/`. Each zips to `m1_<part>_results.zip`. Section 19d is in Part B only.

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
| `RESUME_FROM_INPUT` | True | Section 3b copies results of an unfinished run, attached as a Kaggle input, into the output folder first; runs with different settings are ignored |

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

Output files (in `outputs_m1_<part>/`): per benchmark `rows.jsonl`, `calls.jsonl`, `traces.jsonl`,
`results_long.csv`, `m1_config.json`; top level `metrics.csv`, `stats.csv`, `test_agent_diagnostics.csv`,
`regressions.csv`, `keep_best_rescore.csv`, `livecodebench_regrade.csv`, `per_agent_costs.csv`,
`cost_model_per_call.csv`, `power_by_role.csv`, `metric_relationships.csv`, `spearman_<bench>.csv`,
`stop_reasons.csv`, `summary.md`, all figures; zipped as `m1_<part>_results.zip`.

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
| 2026-10-01 | v3e | Split into Part A (MBPP+ + HumanEval+) and Part B (LiveCodeBench); 3b restore from attached input; results limited to the sampled questions; time-left estimate per benchmark; 19d re-runs only failed programs | The single v3d run did not finish in one session, and a new Kaggle session starts with an empty /kaggle/working, so the old resume could not see earlier results |
| 2026-10-01 | v3e | Section 2b: optional Claude Code install (off by default; token from a Kaggle/Colab secret); Installation section in the README | User request |

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

**v3e Part A (2026-10-01, `m1_evalplus_results.zip`; MBPP+ 120, HumanEval+ 164; idle 57.5 W)**

| Benchmark | Baseline | Self-review | 7B review | 7B + tests |
|---|---|---|---|---|
| MBPP+ pass@1 | 70.8% (85) | 76.7% (92) | 77.5% (93) | 70.8% (85) |
| MBPP+ GPU-s/q | 2.8 | 4.5 | 4.2 | 14.0 |
| HumanEval+ pass@1 | 89.6% (147) | 90.2% (148) | 89.6% (147) | 87.8% (144) |
| HumanEval+ GPU-s/q | 5.9 | 6.8 | 6.7 | 12.7 |

- MBPP+ numbers are identical to v3 (same seeded 120, shared cached drafts, temp 0.1): a replication, not independent evidence.
  HumanEval+ self-review vs 7B review swapped by one question vs v3 full; test agent 141 -> 144.
- Paired tests: baseline -> 7B review MBPP+ 8 gains / 0 regressions, raw p = 0.0078, Holm p = 0.0625; baseline -> self-review
  7 / 0, raw 0.0156, Holm 0.109. Nothing survives Holm over the 8-test family. 7B review -> + tests MBPP+ 1 / 9, raw 0.021, Holm 0.129.
- Repair headroom (MBPP+, 7B review): 35 baseline failures; 20 questions entered repair, 8 fixed (40%); 15 final answers pass
  the public tests but fail hidden ones (invisible to the loop). 25 questions unsolved by any system.
- HumanEval+: 38 / 164 questions have no parseable public tests; 120 drafts pass public tests; only 6 (4%) entered repair.
- Test agent: 5 tests/question; valid 72% (MBPP+) / 95% (HE+); flagged drafts 56 / 30, real bugs caught 9 / 5, false alarms
  on correct drafts 33 / 21 (flag precision ~21% / ~19%); +9.8 / +6.0 GPU-s per question over 7B review.
  regressions.csv lists 8 MBPP+ broken drafts; the 9th McNemar regression (task 301) is a fix 7B review found and the TA arm missed.
- Keep-the-best re-score (TA arm only differs): MBPP+ 85 -> 89 (+6 -2), still below 7B review's 93 (1 gain / 5 regressions);
  HumanEval+ 144 -> 147 (= baseline). Keep-best does not rescue the test agent.
- Per-call cost model (fresh calls): GPU-s = 0.44 + 1.85 per 1k prompt tok + 29.5 per 1k completion tok, R^2 0.99 (split) vs 0.56
  (total tokens); completion token ~15x a prompt token. Within repaired questions, Spearman(calls, GPU-s) = 0.47 MBPP+, 0.25
  HE+ (CI crosses 0); Spearman(total tokens, GPU-s) 0.97 / 0.88.
- GPU-s / wall time = 1.99 for every role (both cards charged on every call, including the 7B reviewer). Mean power ~125 W for
  every role; Spearman(GPU-s, J) = 0.998-0.999. On this setup GPU-s, joules and wall time are effectively one axis.
- Decode speed: qwen3-coder:30b ~54-59 tok/s; 7B reviewer ~38-41 tok/s (30B is MoE with ~3B active). The 7B review is cheaper
  per call only because it writes fewer tokens (~68-81 vs ~150-200 completion tokens).
- Calls vs GPU-s ranking flip: self-review 1.33 calls / 4.46 GPU-s vs 7B review 1.35 calls / 4.19 GPU-s on MBPP+ (same direction
  on HE+). Tokens give the same order as GPU-s, and the accuracy gap is 1 question, so this is illustrative only.
- No ollama errors, no model reloads during measured calls (load_flag 0); 30B 49/49 layers on GPU.
- Full write-up with every figure explained: M1_evalplus_results_report.pdf (24 pages, sent 2026-10-01).

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
- (v3e) The cost-axis result is "calls are a poor proxy; split tokens are a near-perfect one". GPU-s and joules add almost nothing
  beyond split tokens on this hardware (constant ~125 W, GPU-s = 2 x wall). The hardware-native argument has to rest on
  cases where tokens stop predicting time/energy (different models, MoE vs dense, prompt-heavy calls, other GPUs), not on this run.
- (v3e) The +6.7 pt repair gain is not significant under the current 8-test Holm family. The thesis question needs a matched-budget
  comparison (repair vs Best-of-N at equal GPU-s); M1 has no BoN arm, so it cannot say whether repair beats spending the same
  1.5x budget on more samples.
- (v3e) The test agent as a repair trigger is a negative result: ~1 in 5 flags is a real bug, and keep-best does not recover it.

## 7. Known issues and open options (not yet applied)

- **Keep the best program** instead of the last one (19b re-scores this offline). Pre-register before using it.
  (v3e: re-score shows it does not fix the test agent; still worth it as a no-regression safeguard for the repair arms.)
- **Only repair when a public test fails**; use test-agent tests only to choose between versions.
- **Swap roles: 7B writes, 30B reviews** (strong reviewer, cheap drafts; links to C5 adaptive allocation).
  (v3e: the 7B decodes slower than the 30B MoE on T4, so a 7B writer may not be cheaper in GPU-s.)
- **30B checks the test agent's tests** before they can trigger a repair (targets false alarms).
- **Kimi K2.6 reviewer** via NVIDIA NIM (free trial credits) or paid OpenRouter; not in GPU-s/J.
- LCB contamination: check the printed contest-date range against the model's training cutoff.
- LCB re-grade (19d) not yet run on the 60 s-cap results.
- (v3e) Checked in ollama_serve.log: the 7B reviewer is split across both T4s (1.87 GB + 2.30 GB, 29/29 layers on GPU), so
  charging 2 cards per reviewer call is consistent with the GPU-s definition. (The first 7B load had only 26/29 layers on GPU
  while the 30B held a large probe KV cache; the warm-up reload fixed this before any measured call.)
- (v3e) NUM_PREDICT = 512 truncates programs: 33 fresh calls hit the cap (16 MBPP+, 17 HE+); all 3 HumanEval+ baseline
  SyntaxErrors (HumanEval/130, /32, /81) are truncated drafts. Raise to 1024 for MBPP+/HE+.
- (v3e) Early stop is inactive with a single public test (22 HE+ questions); HumanEval/130 ran 3 rounds (7 calls, ~77 GPU-s).
- (v3e) Pre-register one primary comparison (e.g. baseline -> repair_rev on MBPP+) before the LCB / next run instead of an
  8-test Holm family.
- (v3e) Add a Best-of-N / majority-vote arm at matched GPU-s (from cached candidate pools) so repair is compared at equal budget.

## 8. How to run on Kaggle

1. Two notebooks: Part A (`07_...part-a...`) and Part B (`08_...part-b...`). Import each as its own Kaggle notebook;
   they can run at the same time if your account allows two GPU sessions, otherwise one after the other.
2. Accelerator **GPU T4 x2**, Internet **On**. No secrets needed for the local reviewer.
3. Best: *Save Version -> Save & Run All (Commit)*. It runs with the browser closed (up to 12 h) and saves the
   output. The run stops starting new questions after `SESSION_BUDGET_HOURS` (8 h) and still writes the analysis.
4. If it printed `PARTIAL - resume to finish`: open the notebook, *Add Input -> Your Work -> Notebooks ->* this
   notebook's last version, and run again. Section 3b prints `restored N result rows`; the run continues.
   After each benchmark it prints minutes per question and the hours still needed.
5. Download `m1_evalplus_results.zip` / `m1_livecodebench_results.zip`; send `summary.md` (and 19b / 19d output).
6. "Error displaying widget: model not found" is the progress-bar widget failing to render. Harmless.
7. Stay on T4: TPUs cannot run Ollama and have no NVML energy counter.

## 9. Claude Code on Kaggle / Colab (section 2b; checked against code.claude.com docs, 2026-10-01)

- Install: `curl -fsSL https://claude.ai/install.sh | bash` (or `npm install -g @anthropic-ai/claude-code`, Node 22+).
- Needs a Pro, Max, Team, Enterprise or Console account; the free claude.ai plan does not include Claude Code.
- No browser on the VM: on your own computer run `claude setup-token`, copy the token, and in the notebook set it as
  a secret, then `export CLAUDE_CODE_OAUTH_TOKEN=...`. Or use `ANTHROPIC_API_KEY` (API billing).
- Colab has a terminal for interactive `claude`. Kaggle notebooks have no interactive terminal (as far as known); use non-interactive
  `claude -p "..."` from a cell.