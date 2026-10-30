---
name: cosmos3-ci-smoke
description: >
  CI smoke skill for the skill-eval pipeline: two trivial, CPU-only file-writing
  tasks that finish in a couple of minutes. Use only to exercise the eval CI end
  to end (trigger → runner → sequential cells → PR comment → dashboard). Not a
  real Cosmos skill.
---

# Skill-eval CI smoke

Two tiny tasks that prove the evaluation pipeline without touching a model:

1. **smoke-hello** — write `outputs/hello.txt` containing exactly the line
   `cosmos skill eval smoke OK` and print it back.
2. **smoke-python** — with `python3`, compute the sum of the integers 1..100 and
   write `{"sum": 5050}` to `outputs/sum.json`.

Record what you did in `result_summary.md`. No GPU, network or credentials are
needed; everything runs in the task container's working directory.
