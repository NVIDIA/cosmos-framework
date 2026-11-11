# Two-node full-data campaign

This is a new qualification campaign. The historical single-host adapter results remain attached to their original revisions. No local CPU mesh, source download, or infrastructure allocation is a two-host training pass.

The working inventory and machine-readable evidence live at `/localhome/local-rarunachalam/cosmos-adapter-all-family-2node-20261111`. PR #342 remains closed; this work is local.

## Finite training coverage

`SFTConfig.training_epochs` opts into finite, whole-corpus passes. `FiniteSFTCursor` uses the existing `IndexedCursor` order and dataset identity. Every real row contributes once per epoch; the final global optimizer batch has explicit zero-weight padding. Padding does not consume or wrap the source. Real and padded rows both carry the weight field, so mixed microbatches are well-defined. The number of updates is `epochs * ceil(rows / (DP world size * micro_batch_size * grad_accum))`; configuration must match it.

The causal objective accepts a globally padding-only accumulation chunk only in this mode. Its forward/backward graph still participates in collectives. The trainer rejects a whole optimizer update containing only padding. Positive denominator windows retain the existing TOKEN_SUM accumulation and global weighting. Router auxiliary loss must be disabled because that separate objective does not yet mask training padding.

Finite exact resume preflights complete rank-local RNG state, the existing trainer update counters, the complete data cursor and its relation to committed updates inside the checkpoint transaction, before model loading. This mode requires completed updates without skipped attempts; it does not claim mid-accumulation recovery. A fresh factory-owned cursor may be restored during preflight, but refusal leaves model, optimizer and global RNG untouched.

Scheduler construction identity is now saved and checked, including cosine cycle length and linear warmup length. Legacy checkpoints made by a bound scheduler without this identity must resume using their original runtime/configuration or use a model/adapter warm start. This is an explicit compatibility boundary, including adapter-disabled legacy resume. Fresh adapter-disabled behavior and unbound legacy scheduler behavior are preserved.

Flow collation failures converge across ranks before a collective-bearing model forward. This does not attempt to recover a worker that fails inside a distributed forward; scheduler-level failure and checkpoint recovery remain separate campaign gates.

## Validation so far

Focused CPU tests cover uneven sequence lengths, real/padded mixed batches, globally empty accumulated chunks, exact ID coverage, optimizer parameters and moments, and checkpoint pre-mutation refusal. An independent reviewer additionally exercised actual CPU FSDP2/HSDP meshes (one physical host, Gloo, no CUDA). The report and executable probes are in `review/distributed-change-review.{md,json}` in the campaign directory. These checks do not qualify any GPU, model binding, full dataset, or two-node layout.
