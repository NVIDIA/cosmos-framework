# Sim-Bimanual inference troubleshooting

> **Skill note (not in the source):** this file holds the setup, input, generation and evaluation parts of the Sim-Bimanual troubleshooting sub-skill, moved verbatim (cosmos-framework `main` 45fecab): the opening guidance, the symptom rows, procedures 1, 2, 6 and 7 and "Return to the workflow" of its SKILL.md, then the matching sections of its failure playbook. Section numbers are the source's. Only link targets were re-pointed. Launch, training-progress and checkpoint symptoms (procedures 3 to 5) are in the [troubleshooting reference](../../cosmos3-post-train-simulation-bimanual/references/troubleshooting.md) of the Sim-Bimanual post-training skill.

Find the first broken boundary in setup, data preparation, training,
checkpoint loading, generation, or scoring. Start with the cheapest check that
can distinguish the bounded causes. Preserve the
[model contract](model-contract.md) during recovery.

## Find the symptom

| Symptom                                                             | Likely boundary                     | Procedure                                               |
| ------------------------------------------------------------------- | ----------------------------------- | ------------------------------------------------------- |
| Import, CUDA, or model access fails                                 | Environment or allocation           | [Check runtime and access](#1-check-runtime-and-access) |
| Width, mask, timing, caption, or split is wrong                     | Data materialization                | [Validate one input](#2-validate-one-input)             |
| Missing input media, mutated prompt, static video, or mesh mismatch | Input access or generation contract | [Check generation](#6-check-generation)                 |
| Missing metric rows or incomparable PSNR                            | Population and target alignment     | [Check evaluation](#7-check-evaluation)                 |

## 1. Check runtime and access

1. Preserve the exact command, code revision, resolved config, immutable
   checkpoint, selected variant, first traceback, and failing input identity.
2. From the checkout root with the locked environment active, inspect the
   package and CLI. A container package outside the staged checkout is a
   different runtime.

   ```bash
   : "${FRAMEWORK_DIR:?set the Framework checkout root}"
   cd "$FRAMEWORK_DIR"
   source .venv/bin/activate
   python -c 'import cosmos_framework, torch; print(cosmos_framework.__file__, torch.__version__, torch.version.cuda, torch.cuda.is_available())'
   python -m cosmos_framework.scripts.inference --help
   df -h /dev/shm
   ```

   Keep `COSMOS_TRAINING=0` for an inference-only installation; training uses
   `COSMOS_TRAINING=1` and the training extra.
3. **If `uv sync --locked` reports a stale lock before installation:** verify
   it with the required resolver from the checkout root:

   ```bash
   uvx --from 'uv==0.11.29' uv lock --check
   ```

   If this passes, the original command used an incompatible `uv`; repeat setup
   through the documented `uvx` command in the environment guide. Do not regenerate
   the lock with the incompatible resolver. If it fails in a clean checkout at
   the selected Framework commit, preserve the output and report the revision.
4. **If imports fail:** repair the locked environment using the
   [setup guide](../SKILL.md#prerequisites). **If CUDA fails:** repeat GPU
   checks inside the target allocation before replacing dependencies.
5. Check local checkpoint, tokenizer, data and media readability. HF downloads
   may need model read access. Require provider credentials only for resources
   that are actually remote. Never copy or print credentials.
   Continue when imports, help, allocation CUDA, and selected access checks
   pass. Then rerun the original boundary with one sample.

## 2. Validate one input

1. For a schema-5 input, use Framework's checker against the actual export and
   the smallest failing manifest:

   ```bash
   : "${CHECKPOINT_DIR:?set the complete local schema-5 HF export}"
   : "${INPUT_JSONL:?set the failing schema-5 manifest path}"
   python tools/check_bimanual_mixed_inputs.py \
     --checkpoint-path "$CHECKPOINT_DIR" --input "$INPUT_JSONL"
   ```

2. **If width, mask, or domain fails:** trace native calibration, geometry,
   59-value mapping, normalization, and 64-value invalid padding through the
   [action schema](unified-v1-action-schema.md).
   Repair the actual mapping. Do not reshape values or relabel an embodiment.
3. **If timing fails:** inspect timestamps and resampling. Require `T`
   actions for `T+1` observations and `4k+1` output frames. Do not trim one
   modality silently.
4. **If captions or splits fail:** match captions to the selected interval
   and isolate entire episodes. Remove future-outcome leakage when the task
   conditions only on the initial scene.
5. Repeat JSON validation, the schema round trip, and loader materialization.
   Continue with finite tensors, aligned times, grounded captions, and disjoint
   episodes.
   For a local training bundle, follow the native
   [data preparation and short training check](../../cosmos3-post-train-simulation-bimanual/SKILL.md#1-prepare-and-validate-training-data).

## 6. Check generation

1. Run the native [checkpoint and input check](../SKILL.md#1-choose-the-checkpoint-path).
   The checker verifies the export, image, and action sidecar before loading
   weights. Check decoding in the target environment as well. For user-owned remote media,
   establish access separately; checkpoint credentials do not authenticate it.
2. **If an object is confirmed absent:** choose another supported row or
   user-owned media. Preserve relative action paths. Do not cancel an
   independent healthy training run.
3. For field rejection, compare the imported package origin with the staged
   source before editing valid input fields. For a short, static, or corrupt
   video, inspect the actual checkpoint, schema, normalized actions, frame
   count, domain, prompt, seed, and decode properties.
4. For a schedule mismatch, select the matched checkpoint and command. The
   current released schema-5 model and canonical user-trained Stage 3 student
   both use four steps, but each export owns its exact ladder. Preserve the
   selected export's saved schedule and matching CLI step count.
5. For changed punctuation or resolution text, check the export's prompt mode
   and input manifest. The matched schema-5 export selects plain text. Keep the
   exact intended prompt and follow the matching native example.
6. For process-group mismatches, require the resolved world size to equal the
   launched count. Verify divisibility of DP and CP/CFG products before loading
   weights. Retest one row before scaling.
7. Continue only with one playable output per input row after the bundled
   validator fully decodes it and confirms exact saved frame count, dimensions,
   frame rate, and spatial/temporal change. Source, padded-canvas, and saved
   dimensions may differ after spatial-compression alignment. Keep unchanged
   intended conditioning, a complete metadata record, and Cosmos3-Nano, released, and
   user-trained outputs labeled with their actual checkpoint identities.

## 7. Check evaluation

1. Compare planned sample IDs with output and metric rows before aggregation.
   Keep failures, skips, duplicates, corruption, and missing targets in the
   denominator.
2. For PSNR discrepancies, match membership, target alignment, RGB/action
   preprocessing, checkpoint variant, seed, generation recipe, and metric
   implementation. Do not resize, truncate, or drop samples to force a score.
3. Recompute aggregate counts and PSNR from the saved rows. Continue only when
   accounting reconciles. Report a recipe comparison if an invariant changes.
   Follow the [evaluation guide](evaluation.md) for the
   saved results and comparison procedure.

## Return to the workflow

1. Record the symptom, root cause, correction, and checks rerun.
2. Return training to [post-training](../../cosmos3-post-train-simulation-bimanual/SKILL.md),
   generation to [inference](../SKILL.md),
   and scoring to [evaluation](evaluation.md).
3. State whether existing outputs remain comparable.

> **Skill note (not in the source):** the sections below are the matching sections of the source's failure playbook.

## Environment or model access

Activate the locked Framework environment from its checkout root and check
`cosmos_framework.__file__`, the native CLI and CUDA inside the target allocation.
If a pre-install lock check reports stale state, rerun it with the resolver
required by `pyproject.toml`:
`uvx --from 'uv==0.11.29' uv lock --check`. A passing check identifies a local
resolver-version mismatch; use the required setup command rather than rewriting
the lockfile.
Keep `COSMOS_TRAINING=0` for inference-only installs. Training and the metric
entry point need their documented dependencies. Follow
[Framework setup](../SKILL.md#prerequisites) for FFmpeg
and NPP libraries; changing locked packages does not repair a missing native
library. HF access is needed only to download gated assets. The local training
TOMLs use local inputs and disable W&B by default.

## Data failures

For wrong width, mask or domain, trace calibration, coordinate conversion,
59-value mapping, normalization and 64-value padding through the
[action schema](unified-v1-action-schema.md).
Do not reshape an action or relabel its domain to make loading succeed.

For off-by-one timing, require `T` actions and `T+1` observations after
resampling. Check timestamps and interpolation rather than trimming one stream.
For captions or splits, keep wording grounded and source episodes disjoint.

## Inference failures

Run `tools/check_bimanual_mixed_inputs.py` against the downloaded HF snapshot
and selected manifest before loading weights. Resolve missing local media or
provider access before generation.

Match input and checkpoint: the current schema-5 model uses normalized 59D
mixed-domain actions, plain text, and four steps. The user's canonical Stage 3
student uses its own matched action metadata and exact four-step ladder. A width
change is not a semantic conversion.

For short, static or corrupt output, check actions, frame count, domain, prompt,
seed and full video decode. Distinguish source dimensions, padded model canvas
and saved dimensions: a 640x360 input can yield 640x352 after spatial alignment.

For distributed inference, match launched process count and resolved model world
size and divisibility. Test one output before scaling parallelism.

## Evaluation failures

Compare planned sample IDs, prediction/target pairs, successful metric rows,
missing targets and failed/skipped outputs. The native vision scorer can trim
unequal lengths or skip missing targets, so reconcile its aggregate count with
the planned population. Follow the [evaluation guide](evaluation.md)
for matched timing, geometry and complete reporting.
