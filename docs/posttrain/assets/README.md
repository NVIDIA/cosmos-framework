# Adapter validation artifacts

These files record executed checks, not a promise that every checkpoint in an
architecture family is supported. Start with
[the final review](../adapter-final-review.md) and the fixed 48-row
[checkpoint/method table](family-adapter-method-results.csv).

- `family-adapter-qualification.json.gz` is the **unchanged historical snapshot**
  of 182 successful correctness executions. Repeated host/Slurm runs are distinct
  executions. Its old pending-work notes are superseded by the final review.
  Decompressed SHA-256:
  `dbfea5cbc1a72a8956b61465d399f941f002d28228dd1ac662b1e84506bb1293`.
- `adapter-profile-contracts.json.gz` binds all 48 fixed profiles to their actual
  saved target manifests, base/processor/media identities, method settings,
  precision and dataset hashes. Physical input media hashes are included. A
  failed MoE/PiSSA merge remains a failed profile, even though its portable
  artifact is complete.
- `completed-host-summary.json` reconciles the additional four full WTS epochs,
  recovery and evaluation, separately from the correctness subsets. Its raw
  source reports remain in the local review package with SHA-256 digests.
- `moe-pissa-fp64-summary.json` records local FP64 diagnostics of the original
  failing MoE probe. FP64 local calculations are not a full FP64-model run.
- The WTS PNGs and JSON summaries retain measured results. The 576-row subset is
  a **reused scene-disjoint benchmark**, not an untouched final test. Historical
  Cosmos3 figures use a different model/preprocessing and are not baselines for
  Qwen3-VL-2B. Duplicate SVG renderings are retained in the local review package
  and in the previous commits, rather than repeated in the cumulative patch.

- `adapter-effective-execution.json.gz` indexes the versioned stock-compatible
  MoE PiSSA qualification, three physical 2×2 HSDP profiles, repeated-prefix
  correction and expanded BF16 failures. It separates the `e5d975d` trained/exported
  artifact from `da4d98b` public admission and installed reload. Historical 48/47
  counts remain unchanged. Decompressed SHA-256:
  `4b888ba3e06902390f9e4c975af354b246bce8114a583b37b12ccd002351bf73`.
  See [effective execution](../adapter-effective-execution.md) for scope and costs.

Inspect the compressed evidence without installing the training dependencies:

```bash
gzip -dc docs/posttrain/assets/family-adapter-qualification.json.gz > /tmp/adapter-cases.json
sha256sum /tmp/adapter-cases.json
gzip -dc docs/posttrain/assets/adapter-profile-contracts.json.gz > /tmp/adapter-contracts.json
python -m json.tool /tmp/adapter-contracts.json
```

All compression is lossless, with fixed gzip timestamps. Checkpoint weights,
datasets, downloaded third-party runtimes and generated environments are outside
the repository. The pinned runtime installer and preparation/qualification
programs stay in the patch.
