# Evaluate your labeled video multiple-choice data

This is a local **customer-video MCQ** evaluation, not an official benchmark protocol, published Cosmos score, or leaderboard submission. It does not convert arbitrary datasets automatically, build benchmarks, use an external judge, or obtain private labels. Each question has 2–26 options, one known correct answer, and optional category and temporal bounds. There is no requirement to use MVBench's tasks or three options.

## 1. Labels and permissions first — before provisioning a GPU

Obtain the selected videos and their actual ground-truth answers, confirm permission to process them, and confirm that every selected question has exactly one labeled answer. An unlabeled test split, a caption-only collection, or a dataset whose answers require inaccessible credentials is **not ready**. Stop before GPU provisioning if labels or videos are missing. Never infer, invent, or ask the evaluated model to generate missing ground truth. Record the split, source revision, license, inclusion rules, and any deliberate subset. A small subset is a customer subset, not the full source benchmark.

Choose an external persistent directory, for example `/mnt/customer-eval/inspection-v1/`. Keep inputs and outputs outside the source checkout and outside source control. Do not paste private data or credentials into shared notebooks, commits, or conversion-agent prompts without authorization. Conversion agents should work with authorized local files only.

Use exactly this layout (video filenames are examples, not supplied assets):

```text
/mnt/customer-eval/inspection-v1/
  rows.jsonl
  config-nano.json
  config-qwen.json
  media/
    invented_clip_001.mp4
    invented_clip_002.mp4
```

No annotations or videos are included in this recipe. The examples below are invented metadata to explain the schema, not evaluation data or labels for any real collection.

## 2. Map your source columns into canonical `rows.jsonl`

Have an existing Codex or Claude coding agent inspect the source schema, source documentation, and a few authorized rows. Give it the schema and mapping instructions below. Ask it to write canonical JSONL directly into the external directory; do not create a reusable converter product or alter the public benchmark loaders.

Before conversion, require the agent to state a concrete mapping table, including these decisions:

| Canonical field | Mapping decision to document |
| --- | --- |
| `id` | Stable source question ID, including split/video/question identity if source IDs are not globally unique. Never a model-dependent ID. |
| `video` | Existing local video path relative to `media_root`; map source video IDs/filenames explicitly. Do not guess by caption similarity. |
| `question` | Actual question text, not an answer-bearing explanation, caption, rationale, or task label. |
| `candidates` | Ordered source options as an array of full strings. Preserve order and exact text; 2–26 unique nonempty entries. |
| `answer` | Full text of the one correct candidate. Resolve a source letter/index/option-column through the documented option order and index base. Never treat `0` and `1` index conventions interchangeably. |
| `category` | Optional existing source task/category or a documented, fixed grouping. Do not invent a new taxonomy per model. |
| `start`, `end` | Optional paired numeric seconds into the source video, with source units converted explicitly. If the source already provides trimmed clips, use the clip's own timeline, not original-video timestamps. |

Suitable conversion-agent instruction:

> Read CUSTOM_DATA_RECIPE.md and inspect the authorized local dataset. Report the source-to-canonical mapping, answer index base, split, selected count, video mapping, timestamp units and category meanings first. Produce only rows.jsonl and a config JSON in my external data directory using the exact schema below. Preserve source question IDs, option order, actual labels and video content. Resolve each source label to exactly one complete candidate string. Do not generate missing labels, download untrusted replacements, change the public benchmark loaders, add a converter tool, or request a GPU. Validate every selected row and list missing/ambiguous inputs rather than silently skipping them. Run custom_eval.py validate and resolve all validation failures before inference. Record the inclusion rules and source revision in dataset.source/revision. Do not include credentials in any artifact.

Canonical `rows.jsonl` is UTF-8, one complete JSON object per line, no blank lines. These two invented rows illustrate different option counts and category/temporal settings:

```jsonl
{"id":"inspection-v1:invented-001:q1","video":"invented_clip_001.mp4","question":"Which direction does the box move?","candidates":["Left","Right","It stays still","It leaves the frame"],"answer":"Right","category":"object-motion","start":0.0,"end":2.0}
{"id":"inspection-v1:invented-002:q1","video":"invented_clip_002.mp4","question":"Does the person place the cup on the table?","candidates":["Yes","No"],"answer":"Yes","category":"action-completion"}
```

### Content and temporal semantics

- All rows are selected; there is no hidden sampling or automatic row dropping. Make a separate fixed manifest and dataset revision for an intentional subset.
- IDs are nonempty strings, globally unique within the manifest. Multiple questions can refer to one video but must have different IDs.
- `video` uses a safe relative POSIX path within `media_root`; absolute paths, URLs, `..`, backslashes, colon-containing paths and symlink escapes are rejected. Copy authorized media into the root or use paths that resolve within it. Missing or empty files fail validation.
- `question`, candidate strings, ID and category cannot be blank or have surrounding whitespace. Unknown fields are rejected so source labels/rationales are not accidentally sent to inference. Candidate duplicates are rejected. Keep semantically duplicate options out of the source selection too; validation cannot determine semantic equivalence.
- `answer` must match exactly one candidate string, including case and punctuation. The runner sends *all candidate texts* but never sends which one is correct, the answer field, IDs, categories or source labels. The source text itself must not disclose the answer in a rationale or annotation; inspect this during mapping.
- `start` and `end` must both be omitted for a full clip, or both supplied as finite numeric seconds with `0 <= start < end <= duration`. Booleans, NaN and infinity are invalid. Questions must be answerable from the selected segment; validation checks bounds, not semantic relevance. Native decoding performs temporal selection before sampling.
- Categories describe the question, not the predicted answer. Missing category is reported as `uncategorized`. Category accuracy is a question-weighted percentage within that category; overall accuracy is question-weighted across every selected row, not an average of category percentages. If a category is literally named `uncategorized`, it shares the missing-category bucket.

## 3. Write `config-nano.json`

Paths resolve relative to the **config file**, never the current working directory. Absolute external input paths also work. Adapt the invented dataset identity, source, model revision and pixel/frame settings to your data and deployed model:

```json
{
  "dataset": {
    "name": "customer-inspection-mcq",
    "revision": "inspection-v1-labeled-fixed-selection",
    "source": "Authorized local inspection export; source revision example-r1; fixed selected split; mapping documented by data owner"
  },
  "manifest": "rows.jsonl",
  "media_root": "media",
  "nframes": 32,
  "min_pixels": 50176,
  "max_pixels": 50176,
  "image_patch_size": 16,
  "model": "nvidia/Cosmos3-Nano",
  "model_revision": "unknown",
  "generation": {
    "temperature": 0,
    "seed": 42,
    "max_tokens": 32,
    "enable_thinking": false
  }
}
```

Customization points:

- `dataset.name/revision/source` are required nonempty strings. Set revision to the real fixed export/selection identity; record transformations and licensing/source context in `source`. Do not put secrets or credential-bearing URLs here.
- `nframes` is an even integer at least 4, requested of the existing native video utility. Choose it to fit the segment durations and model capacity. Native decoding can reject clips that cannot satisfy the request; the runner pre-decodes every selected clip before any endpoint request.
- `min_pixels` and `max_pixels` are positive per-frame spatial pixel budgets (`min <= max`), not a total video token limit. `image_patch_size` must match the deployed processor; the example is not a universal model profile. Native shape rounding may change actual dimensions. Keep budgets fixed for like-for-like comparisons unless explicitly studying budgets.
- `model` and `model_revision` identify the deployed model. If omitted, use `COSMOS_MODEL` and `COSMOS_MODEL_REVISION`; model must be supplied in config or environment, and absent revision is honestly `unknown`. A config value takes precedence. The runner does not select/deploy a different endpoint based on the model field; point the endpoint to that actual deployment.
- Optional `processor_profile` is an evidence object supplied only from verified deployment/processor provenance, with `patch_size`, `spatial_merge_size`, `temporal_patch_size`, `image_token_id`, `video_token_id`, `provenance`, and the bounded-group markers `vision_start_token_id`/`vision_end_token_id` as supported by `vlmeval.execution_evidence`. When omitted for the exact `nvidia/Cosmos3-Nano` model name, the shared helper uses its pinned, provenance-recorded Nano processor profile; other models do not inherit that profile. Do not copy Nano values to another model or invent a profile to obtain a verified status. Missing runtime observability remains unknown; requested budgets alone are not proof of effective execution. Even verified bounded token-group counts do not attest decoded frame count, decoder FPS, generation settings or effective model revision.
- `generation` supports only `temperature`, nonnegative integer `seed`, positive integer `max_tokens`, and boolean `enable_thinking`. Defaults are the example values. The prompt requires one uppercase letter; a very small token cap or thinking mode can cause truncation, which counts wrong. Do not retry malformed answers selectively.
- No judge, scoring URL, label URL, automatic converter, output overwrite, or resume settings exist. Scoring is always local strict MCQ accuracy. Output location is passed separately with a fresh `--work-dir`.

## 4. Validate all inputs before using an endpoint

The toolkit is `.agents/skills/evaluation/cosmos3/reasoner/vlmevalkit` in this repository. In a launchable installation, use its toolkit root containing `vlmeval/` and `launchable/`. Use the installed launchable Python environment (including decord, ffmpeg and the native Cosmos video dependencies). From that root:

```bash
python launchable/custom_eval.py validate --config /mnt/customer-eval/inspection-v1/config-nano.json
```

Validation is CPU-side and sends no endpoint requests. It validates all labels, IDs, candidate counts, safe media paths, media duration and temporal ranges, probes media frames, and hashes every selected media file. Resolve every failure; never remove inconvenient rows just to obtain a completed score. An intentional subset needs a new documented identity before model comparison. Keep the returned `dataset_identity`, expected count and `manifest_sha256`. The semantic data identity includes dataset metadata, canonical rows and video hashes/metadata; reformatting JSON does not change it, but changing labels, options, selection, categories, temporal bounds or media bytes does. The raw manifest hash can change on harmless JSON reformatting.

## 5. Run from notebook or CLI

Open `launchable/Customer_Data.ipynb` from the `launchable/Evaluate_Cosmos3.ipynb` overview. Set its visible `CONFIG_FILE` to your absolute external config path; the notebook invokes the same validate/run CLI and renders `summary.json`. Use the overview/launchable endpoint workflow for deployment. The same Python API can also be called from a notebook cell after making the toolkit root and `launchable/` importable:

```python
from pathlib import Path
import sys
sys.path.insert(0, str(Path("launchable").resolve()))
import custom_eval

config = Path("/mnt/customer-eval/inspection-v1/config-nano.json")
validated = custom_eval.validate(config)
summary = custom_eval.run(config, Path("/mnt/customer-eval/results/nano-inspection-v1-run01"), api_nproc=1)
summary
```

This cell assumes the notebook working directory is the toolkit root; adjust the import path to the actual toolkit installation if not. Endpoint startup and model deployment remain the notebook/launchable's responsibility. Do not provision an endpoint until validation succeeds.

For CLI, export the actual complete chat-completions URL; keep a real API key only in the environment:

```bash
export COSMOS_API_BASE=http://cosmos-endpoint:8080/v1/chat/completions
export COSMOS_MODEL=nvidia/Cosmos3-Nano
export COSMOS_MODEL_REVISION=unknown
export COSMOS_API_KEY=EMPTY
python launchable/custom_eval.py run \
  --config /mnt/customer-eval/inspection-v1/config-nano.json \
  --work-dir /mnt/customer-eval/results/nano-inspection-v1-run01 \
  --api-nproc 1
```

`EMPTY` is only for a local endpoint that does not require authentication. Use your authorized secret environment for authenticated endpoints; never store keys in config. `COSMOS_TIMEOUT` optionally sets positive finite request timeout seconds (default 300). Concurrency is 1–4; use 1 first. No endpoint URL or key is deliberately persisted. Dataset/source strings are user metadata, so keep them credential-free.

Choose an external output parent that the evaluation process owns or can write to. Inputs may remain read-only. If an operator-provided output directory has the wrong owner, correct that directory's ownership or choose a writable output location before running; do not broaden raw-data permissions or alter the mapping to bypass a filesystem error.

A result directory **must not already exist**, even if empty. Each run validates inputs again, pre-decodes every selected clip with the existing native video processor, and only then issues requests. Temporary transport files can contain encoded customer video while running and are removed when the run exits normally or through handled exceptions; keep work storage private and remove abandoned temporary files after an unclean process kill.

## 6. Inspect local scoring and compare models

Outputs under the fresh work directory:

- `validated.json`: fixed dataset identity, source metadata, media SHA256/durations, selected count, requested sampling/model/generation.
- `requested-execution.json`: requested model/generation/processor budget/profile, not proof of runtime execution.
- `responses/000000.json`, etc.: per-request HTTP status, shared requested/observed execution evidence, strict parsed answer outcome, or redacted exception type. No inference prompt or video base64 is retained in response evidence. Arbitrary malformed model text is not retained.
- `execution-summary.json`: shared aggregate of recorded requested settings, observed bounded visual-group shapes, verification counts and unknown fields. Also embedded in `summary.json` as `execution_evidence`; incomplete evidence coverage or matching token geometry does not establish endpoint/model attestation or all effective media settings.
- `results.jsonl`: one result per processed row, stable ID, category, parsed single-letter prediction (or null), validity, correctness, finish reason and response-evidence pointer; transport errors have an `error` instead of a correctness value.
- `summary.json`: `benchmark=customer-video-mcq`, `scope=custom`, dataset/model identity, `status`, expected/scored coverage, elapsed seconds, execution pointers, overall accuracy and category breakdown.

A valid answer is exactly one uppercase candidate letter after surrounding whitespace is removed. Full option text, lowercase letters, multiple choices, explanations, blanks and truncated (`finish_reason=length`) answers are **wrong** and remain in the denominator. Ground-truth labels are stored locally in your input manifest, never sent as an answer field. HTTP/connection/JSON transport failures are **not** wrong answers: the run fails, exits nonzero, and does not report a completed accuracy. Successful HTTP JSON responses with missing/malformed answer choices count as invalid wrong answers, while their execution evidence remains unverified. Inspect `status=completed`, `coverage.expected == coverage.scored`, and evidence availability before quoting any score. A preflight failure also fails the run without issuing requests. Never compare a failed/partial result as if complete.

For Qwen or another compatible deployment, copy `config-nano.json` to `config-qwen.json` and change only the actual model/revision and provenance-grounded processor settings needed for that deployment. Point `COSMOS_API_BASE` to its endpoint. Do not regenerate the manifest, relabel options, alter selection, or replace media between runs. Use a different fresh work directory, for example `/mnt/customer-eval/results/qwen-inspection-v1-run01`. Use the exact model-supported patch size/profile; document any unavoidable processor budget differences.

Before reporting comparison, require both runs completed, equal expected/scored counts, and identical `dataset_identity`; verify manifest/media identities if transferring between machines. Compare `primary_metric.value` (percent) with `correct/total`, per-category `accuracy_percent`, elapsed seconds and requested/observed execution evidence. Identify model revision, any budget differences, unknown runtime facts and transport/deployment environment. Preserve first-run results rather than overwriting or selectively regenerating wrong answers. The result measures this fixed customer selection under these settings, not official benchmark parity or leaderboard rank.

### Fields reported as unknown

Every run lists the same five fields in `execution_evidence.unknown_fields`. A standard chat-completions response does not report these values, so they are left unknown rather than filled in from the requested settings.

| Field | Meaning |
| --- | --- |
| `decoded_video_frames` | Frames actually decoded and the source frame indices used. Frame slots are reported separately and can include repeated or padded frames. |
| `processor_width_height` | Width, height and crop used by the processor. Grid area is reported, but area alone does not determine dimensions. |
| `effective_decoder_fps` | Sampling FPS the decoder actually used, as opposed to the requested or source FPS. |
| `effective_generation_settings` | Generation parameters the server actually applied, as opposed to the requested configuration. |
| `effective_model_revision` | Checkpoint revision the server actually loaded, as opposed to the configured revision or returned model name. |

For image inputs, the three video-decoder fields do not apply. What is reported: requested settings, the model name returned by the server, token usage, and visual-token counts. Frame slots and grid areas are derived from those token counts using the processor profile.

