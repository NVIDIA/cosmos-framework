# Start the Bimanual workflow

1. For generation, use the [inference tutorial](../inference/SKILL.md).
   It prepares the locked environment, selects and validates one manifest row,
   checks media access, generates video, and inspects the result.
2. For a fresh four-stage build, use the
   [post-training tutorial](../post-training/SKILL.md).
   It prepares Nano-to-DCP weights and a completed local bundle, launches the
   native Framework TOMLs, checks save/resume, and follows Stage 0 through
   Stage 3 handoffs.
3. For a failure, use the
   [troubleshooting guide](../troubleshooting/SKILL.md).
   It routes symptoms to diagnosis, the smallest supported recovery, and
   matched verification.
4. For held-out scoring, use the
   [evaluation guide](../inference/references/evaluation.md).
   Keep checkpoint identity, preprocessing, population, and generation recipe
   matched.

Shared action and timing invariants live in the [model contract](model-contract.md).
The [environment guide](release-workflow.md) covers conditional credential and
allocation details. The [prompt guide](prompt-guide.md) explains grounded
plain-text conditioning.
