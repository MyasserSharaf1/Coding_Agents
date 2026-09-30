# M1 notebooks: every version

Each notebook version sits in `notebooks/`, numbered in order, and each one was added in its own commit
(`git log` shows the history, `git diff` between commits shows what changed). `m1-latest.ipynb` is always
a copy of the newest version. The full change history and results are in `IMPLEMENTATION_LOG.md`.

| # | File | Date | What changed |
|---|---|---|---|
| v1 | `notebooks/01_m1-v1-baseline-vs-repair_n20.ipynb` | 2026-09-20 | Baseline vs supervisor+repair, n=20 per benchmark, 4-axis metering (with run outputs) |
| v2 | `notebooks/02_m1-v2-full-benchmarks-test-agent-lcb.ipynb` | 2026-09-25 | Full MBPP+/HumanEval+, LiveCodeBench, test agent, public/hidden split, resume, rebuilt correlations |
| v3a | `notebooks/03_m1-v3-kimi-reviewer-120.ipynb` | 2026-09-26 | 120 per benchmark, Kimi K2.6 reviewer arm, walkthrough cell (19c) |
| v3b | `notebooks/04_m1-v3-local-7b-reviewer-120.ipynb` | 2026-09-27 | Local qwen2.5-coder:7b reviewer by default; 19d LiveCodeBench check + re-grade |
| v3c | `notebooks/05_m1-v3-full-benchmarks-lcb900s.ipynb` | 2026-09-28 | Full benchmarks; LCB suite cap 60 s -> 900 s with 10 s per case |
| v3d | `notebooks/06_m1-v3-run-mbpp120-he164-lcb100.ipynb` | 2026-09-30 | MBPP+ 120, HumanEval+ all 164, LiveCodeBench 100 |

## Find a version

Each version is one commit on this branch. The commit page shows exactly what changed; *browse files* shows the whole folder as it was at that version.

| Version | Date | Commit | Snapshot | Message |
|---|---|---|---|---|
| m1-v1 | 2026-09-20 | [`32bfb87`](https://github.com/MyasserSharaf1/coding_agents/commit/32bfb8704b0d6d0f935ac2ee7e6556c8a4cb9a69) | [browse files](https://github.com/MyasserSharaf1/coding_agents/tree/32bfb8704b0d6d0f935ac2ee7e6556c8a4cb9a69/milestone-1) | v1: Baseline vs supervisor+repair, n=20 per benchmark, 4-axis metering (with run outputs) |
| m1-v2 | 2026-09-25 | [`cd99fbe`](https://github.com/MyasserSharaf1/coding_agents/commit/cd99fbe4e2b463dbadf6fae8b24297671dbbbf72) | [browse files](https://github.com/MyasserSharaf1/coding_agents/tree/cd99fbe4e2b463dbadf6fae8b24297671dbbbf72/milestone-1) | v2: Full MBPP+/HumanEval+, LiveCodeBench, test agent, public/hidden split, resume, rebuilt correlations |
| m1-v3a | 2026-09-26 | [`5b85af9`](https://github.com/MyasserSharaf1/coding_agents/commit/5b85af9936d4bd6b684d58bea55caea34022a632) | [browse files](https://github.com/MyasserSharaf1/coding_agents/tree/5b85af9936d4bd6b684d58bea55caea34022a632/milestone-1) | v3a: 120 per benchmark, Kimi K2.6 reviewer arm, walkthrough cell (19c) |
| m1-v3b | 2026-09-27 | [`b607484`](https://github.com/MyasserSharaf1/coding_agents/commit/b607484012c3333ecdf481a7ca2a24c803bdfe12) | [browse files](https://github.com/MyasserSharaf1/coding_agents/tree/b607484012c3333ecdf481a7ca2a24c803bdfe12/milestone-1) | v3b: Local qwen2.5-coder:7b reviewer by default; 19d LiveCodeBench check + re-grade |
| m1-v3c | 2026-09-28 | [`024f6eb`](https://github.com/MyasserSharaf1/coding_agents/commit/024f6eb530b3d837641dbb90f363e073c96d4116) | [browse files](https://github.com/MyasserSharaf1/coding_agents/tree/024f6eb530b3d837641dbb90f363e073c96d4116/milestone-1) | v3c: Full benchmarks; LCB suite cap 60 s -> 900 s with 10 s per case |
| m1-v3d | 2026-09-30 | [`987fb85`](https://github.com/MyasserSharaf1/coding_agents/commit/987fb8570d41d8bd928e94aceb0e340b83c51f34) | [browse files](https://github.com/MyasserSharaf1/coding_agents/tree/987fb8570d41d8bd928e94aceb0e340b83c51f34/milestone-1) | v3d: MBPP+ 120, HumanEval+ all 164, LiveCodeBench 100 |
| m1-src | 2026-09-30 | [`21d5d71`](https://github.com/MyasserSharaf1/coding_agents/commit/21d5d713b4090641f715e057501ac28717f01a08) | [browse files](https://github.com/MyasserSharaf1/coding_agents/tree/21d5d713b4090641f715e057501ac28717f01a08/milestone-1) | Add cell sources, build/dry-run scripts and implementation log |

Tags with the same names (`m1-v1`, ...) exist in the local copy. To add them on GitHub, run this once on your own computer after `git fetch`:

```
git tag m1-v1 32bfb87
git tag m1-v2 cd99fbe
git tag m1-v3a 5b85af9
git tag m1-v3b b607484
git tag m1-v3c 024f6eb
git tag m1-v3d 987fb85
git push origin --tags
```

## Other folders

- `src/cells/` one file per notebook cell (source of truth), `src/build.py` assembles them into the notebook
  (`python src/build.py notebooks/07_....ipynb`), `src/dryrun.py` runs it end to end with a mocked model.
- `src/*_cell.py` stand-alone cells you can paste into an already-finished run.

## Adding the next version

1. Change cells in `src/cells/`, build to `notebooks/NN_<name>.ipynb`, copy it to `m1-latest.ipynb`.
2. Add a row above and a line to `IMPLEMENTATION_LOG.md`.
3. `git add -A && git commit -m "vX: what changed"` (message = the reason, results in the body).
4. After a Kaggle run, add `results/<version>/summary.md` in the same way so results sit next to the notebook that produced them.
