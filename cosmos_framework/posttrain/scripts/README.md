# Offline posttrain scripts

## Merge experts with task arithmetic

`merge_experts.py` is the posttrain-local copy of the Cosmos3 Nano Reasoner
separate-then-merge ablation script. It merges full-weight Hugging Face
safetensors experts with task arithmetic:

```text
W_merged = W_base + sum(lambda_t * (W_expert_t - W_base))
```

The base checkpoint remains the original hard-coded path in the script:

```text
/path/to/shared-storage
```

Every base and expert directory must contain `model.safetensors.index.json` and
the referenced safetensors shards. All checkpoints must have identical weight
keys. The script runs on CPU, writes BF16 weights using the base checkpoint's
shard layout, and copies JSON config/tokenizer files from the base directory.

Run from the cosmos-framework repository root with one shared lambda:

```bash
PYTHONPATH=. python -m cosmos_framework.posttrain.scripts.merge_experts \
  --experts /path/to/ctrl_ifeval,/path/to/ctrl_chartqa \
  --lam 0.5 \
  --out /path/to/merged_lam0p5
```

Use `--lams` for one lambda per expert. It overrides `--lam`:

```bash
PYTHONPATH=. python -m cosmos_framework.posttrain.scripts.merge_experts \
  --experts /path/to/ctrl_ifeval,/path/to/ctrl_chartqa \
  --lams 0.25,0.75 \
  --out /path/to/merged_per_task
```

The existing script semantics are preserved: `--experts` and `--lams` are
comma-separated, the output directory may already exist, and matching output
shards and JSON sidecars are overwritten.

Experts saved as consolidated posttrain `model.pt` files (rather than Hugging Face
safetensors directories) are merged by `python -m cosmos_framework.posttrain.checkpoint.tools.merge`,
which emits a native checkpoint artifact; `merge_experts.py` remains the safetensors variant.

## Pre-launch SFT row validator

`validate_sft_rows.py` runs every row of the configured SFT sources (`rl.data`, and `rl.validation.data` when
enabled) through the trainer's own `build_sft_dataset` -> `build_sft_packer` -> `sft_process_sample` path on CPU,
before any model exists. `--validate-manifest` checks wiring and file availability only; a row that the packer
refuses, or whose mask disagrees with its assistant turns, otherwise dies at step 0 inside `algorithm/sft.py`.

Per row it reports total tokens, loss tokens (`labels != -100`), an independent assistant-token count (each
rendered assistant block the row's `supervise` field supervises: its text fragments plus its `<|im_end|>`,
tokenized on their own, never through offsets or labels, one token fewer per block when the row's `opener` is
`prefilled`), compared per assistant block with the loss tokens the labels carry between supervised `<|im_end|>`
tokens; image and video counts with their visual tokens (`*_grid_thw` / merge^2) against the `<|image_pad|>` and
`<|video_pad|>` tokens in the stream, each modality on its own (the model scatters image and video features
separately, so an image surplus never cancels a video deficit); the row's length against `rl.max_length`; and the
source role grammar (tool results follow tool calls one-to-one; by id only when every call and every result of
the turn carry string ids, since the Qwen template renders no ids; the conversation ends with an assistant turn;
consecutive assistant turns are accepted, as the normalizer and the packer accept them and supervise each as its
own block). Rows are numbered from 1 in source order.

Source availability is not a configuration error for this script: a selected split whose JSONL is missing or
whose factory does not import is reported as that split's load refusal (FAIL line, gate line, `load_error`, path
redacted), and an unavailable split never prevents the other split from being inspected. The trainer's
`--validate-manifest` preflight still refuses the whole configuration in that case.

Terms in its output:

- **FAIL line**: `FAIL <split> row <n>: <reason>; <reason>` for a refused row, or `FAIL <split>: <reason>` for a
  split whose source is refused as a whole (a JSONL that cannot be opened, a factory that does not import, a
  missing media file at load, no rows, a held-out source identical to the training source) and for the validation
  split's held-out census when it refuses the pair or could not read a source row (`census_error` below).
- **gate line**: one per split, exactly
  `<split>: rows <N> | loss_tokens == assistant_tokens on <K>/<N> | longest <L> <= <max_length> | failed <F>`
  (`>` when the longest row is over-long; `| rows % dp_world(<W>) = <R>` appended with `--dp-world`); for a
  split refused at load: `<split>: rows 0 | load refused: <reason> | failed 1`.
- **summary line**: the last JSONL line of a split, `{"summary": {...}}` with `split`, `rows`, `failed`,
  `failed_rows`, `unmeasured_rows`, `matched_rows`, `longest_tokens`, `max_length`, `over_long_rows`,
  `total_tokens`, `loss_tokens`, `loss_share`, `images`, `videos`, `visual_tokens`, `held_out_census` (the
  validation split's census against the training split: the setup's `require_disjoint_datasets`; `null` elsewhere),
  `census_error`, `dp_world`, `dp_remainder`,
  `load_error` (the refusal when the source could not be loaded, else null) and `ok`. `failed` counts the refused
  rows (`len(failed_rows)`) plus one when the split carries a `census_error`, so on a validation split the setup
  refuses, `failed` exceeds `len(failed_rows)` by one and the gate line carries `held-out census refused: ...`.
  `census_error` is the setup's refusal of the pair, or
  `held-out census could not read <side> source {row <n>[ (item <id>)] | (<call>)}: <class>: <message>` when the
  census failed while it asked a source for something: `row <n>[ (item <id>)]` is the row it was reading, or whose
  media it was hashing right after (a lazy factory's missing backing record, unreadable media, a corrupt record, a
  malformed record the factory's own code trips on), and `(<call>)` is the source call that failed while no row was
  being read: `(len())`, the length the census asks for when it starts a side's walk and for its row counts;
  `(.rows)` and `(.media_hashes)`, a JSONL source's in-memory rows and its media digests. The side and the call are
  the ones the census asked for last, never a guess from the walk order. Every `Exception` the call raises is this
  sentence, exactly the classes the per-row reader files as `source refused the row`, so the verdict never depends
  on which class a factory happens to raise (both readers read a row through one helper, so a `StopIteration` a
  factory raises is `StopIteration: ...` for both and a subclass of it is named by its own class, never the
  `RuntimeError: generator raised StopIteration` a generator frame makes of it); a `KeyboardInterrupt` propagates,
  and so does a failure of the census's own code before it asked either source for anything (it is not a source's);
  the class is named, the item id and every path are redacted and the whole sentence is bounded, as in a row reason;
  the shape is the module constant `CENSUS_READ_ERROR`, held equal to this line by a test. In both cases the
  validation rows are still inspected, the summary is written and the exit status is 1. A split refused at load
  still gets its summary line (`failed 1`, `ok false`).
- **straddle**: one token whose character span covers both a supervised assistant-body character and an ignored
  header or media character (the tokenizer forms `\n\n` from the header's newline and a body that starts with a
  newline); the packer refuses such a row, and the validator reports the refusal as the row's reason.
- **redaction**: reasons that quote an exception keep only the last two components of each absolute path
  (`/data/items/item_7/image.png` -> `item_7/image.png`) and are bounded in length as a whole sentence (an item id
  quoted in a census reason is cut shorter, so the class and the message after it survive), so the report carries
  counts and reasons only, never row content and never a path with its tree.
- **supervise / opener**: the row's own fields, read exactly as the packer reads them (`SFTTarget`).
  `supervise` is `all_assistant` (the default: every assistant block carries loss) or `last_assistant` (the last
  assistant block alone; every earlier one stays at -100 like the observations around it, so `loss_tokens_per_block`
  has one entry and the independent count covers that block only, while `assistant_blocks` and `body_tokens` still
  describe every assistant block of the row). `opener` is `self` (the default: the `<think>` that opens a supervised
  body is the student's token) or `prefilled` (the serve prefills that token, the packer leaves it at -100 and the
  independent count subtracts one per supervised block). So the loss count a passing row reports means: under
  `all_assistant` / `self`, every assistant body plus its `<|im_end|>`; under `last_assistant`, the last body plus
  its `<|im_end|>`; under `prefilled`, the same minus one `<think>` per supervised block. A `prefilled` row with a
  supervised turn that renders no `<think>` opener is refused by the packer
  (`QwenSFTDataPacker._require_one_opener_per_supervised_block`) and the refusal is the row's FAIL line; a row whose
  `meta.supervise` or `meta.opener` disagrees with its field is refused by the source loader (`normalize_row`): the
  row's FAIL line for a factory source, the split's load refusal for a JSONL source (every JSONL row is normalized at
  load). An unknown value is the row's reason, not a crash. Each row's report line carries `supervise`, `opener` and
  `supervised_blocks`. The run-level knob `rl.mask_think_opener` (the recipe's acknowledged variant) applies the same
  rule to every row of the run: the validator builds its packer with the run's value, exactly as the trainer does, and
  the independent count takes one token off every supervised block that opens with `<think>` when the knob or the
  row's `prefilled` asks for it, once, never twice; a block that opens with anything else loses nothing under the knob
  (the packer masks nothing there and refuses nothing), and under `prefilled` the packer has refused such a block
  already.

Exit status is 1 on any failed row or split, 2 when `--out` names an existing file and `--force` is absent. The
JSONL report (`--out`; default a fresh `<job out_dir>/validate_sft_rows_<timestamp>.jsonl`, never reused) holds
one line of counts and redacted reasons per row plus the summary line per split.

```bash
PYTHONPATH=. python -m cosmos_framework.posttrain.scripts.validate_sft_rows \
  --config cosmos_framework/posttrain/configs/sft_qwen3p8_27b_multimodal.py --dp-world 4 -- <overrides>
# equivalently, from the entrypoint, before any trainer is built:
PYTHONPATH=. python -m cosmos_framework.posttrain.entrypoints.sft --validate-rows -- <overrides>
```

`--dp-world N` additionally refuses a training source whose row count is not divisible by N; the entry's
`--validate-rows` takes N from `WORLD_SIZE` under torchrun and otherwise prints `train: dp_world not checked` after
the train gate line (the SFT source
refuses it at setup; no rows are dropped). torchrun is the only route to that check from the entry (it has no valued
`--dp-world` flag), and under torchrun `main()` runs on EVERY rank: each rank validates every row and writes its own
timestamped report under the job output dir, so run `--validate-rows` on one rank (a one-process torchrun, or the
script above with `--dp-world`) rather than on the training allocation. `--allow-empty-assistant` accepts assistant
turns whose body is empty (only `<|im_end|>` supervised), which the validator refuses by default.

The held-out census (the setup's `require_disjoint_datasets`) needs both splits in one run:
`--split all` (the default, and what the entry's `--validate-rows` uses) applies it; `--split validation` alone prints
`validation: held-out census not run (the train split was not loaded in this run; use --split all ...)`, leaves
`held_out_census` at `null` in the summary and can exit 0 for a split the setup would refuse. Use `--split all`
whenever the exit status is meant to stand for the setup's verdict.

## GR00T-N1.7 checkpoint export

`export_gr00t_checkpoint.py` turns a posttrain checkpoint (the `action_head.*` wire scope only) back into
a whole checkpoint Isaac-GR00T and vLLM-Omni can load
(for an operator-provisioned GR00T runtime). The two GR00T parity gates are test
launchers, `tests/gr00t_parity_cli.py` and `tests/gr00t_train_parity_cli.py`, not scripts.

```bash
python -m cosmos_framework.posttrain.scripts.export_gr00t_checkpoint \
  --base nvidia/GR00T-N1.7-DROID --posttrain-ckpt $CKPT/iter_000100 --out $OUT/export
```
