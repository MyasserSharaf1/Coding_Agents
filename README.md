# Coding Agents: thesis experiments

Efficient local code generation with small LLMs: repair loops, reviewers and test agents, measured in
accuracy and in hardware cost (calls, tokens, GPU-seconds, joules).

## How this repository is organised

- **One branch per milestone**: `milestone-1`, `milestone-2`, ... All work for a milestone happens on its branch.
- **One commit per notebook version.** The commit message says what changed, why, and the results that version
  produced. Every version stays as its own file in `milestone-N/notebooks/`, numbered in order.
- **Tags** mark each version (`m1-v1`, `m1-v2`, `m1-v3a`, ...): `git checkout m1-v2` or the GitHub *Tags* page.
- **`main` only receives a milestone when it is done**: its final run finished, results checked and recorded
  in the milestone's `IMPLEMENTATION_LOG.md`. Then the milestone branch is merged into `main` through a pull request.

## Milestones

| Milestone | Branch | Status |
|---|---|---|
| M1: repair vs baseline, reviewers, test agent (MBPP+, HumanEval+, LiveCodeBench) | `milestone-1` | In progress: final run pending |
