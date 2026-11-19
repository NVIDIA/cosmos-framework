# Notebook validation — October 15, 2026

This report records validation of source GitLab revision
`563ead66b769a7462c42067cbfa971cbe622bcef` before its presentation changes were
merged into this recipe. The measurements below apply to that source revision.

**Passed:** the published notebook completed end to end on cs-001 with four
NVIDIA GB200 GPUs. Slurm job **2503107** exited `COMPLETED / 0:0`.

| Check              | Result                                                                                                                                     |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------ |
| Notebook execution | All eight code cells passed, in order, without modifying their source.                                                                     |
| Model and data     | Pinned HF bundle downloaded; prepared EEF data, normalizer, and checkpoint contract checks passed.                                         |
| Source previews    | All three donor/host pairs prepared and six video players displayed before generation.                                                     |
| Generation         | Handover drop, pickup miss, and pickup slip examples rendered with seed 0 and the original 30 sampling steps.                              |
| Actions            | Each example used 16 finite actions with 20 values per action.                                                                             |
| Video decoding     | All nine unique source/generated videos decoded: 17 frames, 5 FPS, 960×960, with changing frames.                                          |
| Browser playback   | All 15 embedded players played in Chrome; workflow diagram and author line loaded. Desktop and narrower comparison layouts were inspected. |
| CPU suite          | 108 passed, two skipped.                                                                                                                   |

The successful job took **2m 31s**, including approximately **98s** in the
generation cell. It reused the model download from the first attempt; the first
download took approximately 64s. These are measurements from this run, not a
runtime guarantee. Seventeen frames at 5 FPS cover timestamps 0–3.2s and have a
3.4s encoded playback duration.

## Required framework revision

The first attempt, job **2503049**, failed during model initialization with
`Missing key flex_attention`. It used a different framework checkout. The second
attempt used the revision recorded in the bundle's training contract and passed.
No notebook code, model weights, checkpoint configuration, recipes, or seeds were
changed between attempts.

- Framework: `cf5d68c00d97ccd2480a2320ed652b92dec63102`.
- HF bundle: `eb236695a05af0c4977554baf004162159a7bb99`.
- Tested cookbook commit: `b9d9f9dfed124e79095bd1965678743c14739314`.
- Notebook SHA-256: `1dba8fe43144b3f6180834d1cec672db4ccdea77b4e94a26829464636ec776a6`.

## Evidence

Both attempts, Slurm receipts, logs, source snapshots, and generated outputs
were preserved in the authors' internal validation archive; that cluster path
is not distributed with this recipe. The successful executed notebook,
standalone HTML, machine-readable checks, and browser review were retained
there. Generated videos remain outside Git. This validates execution and
playback for the three examples; it does not establish a failure yield or
downstream model gain.
