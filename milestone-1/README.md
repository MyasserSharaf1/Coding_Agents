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
