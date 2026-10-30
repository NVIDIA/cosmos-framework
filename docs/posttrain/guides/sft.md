# SFT

## 1. When to choose it

Full-parameter supervised fine-tuning of Qwen conversations through the existing `LLMTrainer`.
The [multimodal recipe](../../../cosmos_framework/posttrain/configs/sft_qwen3p8_27b_multimodal.py) targets `Qwen/Qwen3.8-27B`
(the dense `qwen3_5` architecture), with text, images, videos and multiple assistant turns.
Local/torchrun FSDP is supported; Ray execution, LoRA, sequence packing and MTP training are not.
There is no rollout engine, reference model or reward function.

SFT opts into Transformers 5.8.1; existing RL/GR00T defaults stay on 4.57.6. The SFT builder
supports Qwen text models, dense Qwen3.5/3.8 multimodal models and the Qwen3-VL / Qwen3-VL-MoE
families (all three ship `Qwen3VLProcessor`). A "seat" is the model construction path the SFT
objective sits on. Qwen3.5/3.8 sits on the HF backend (`model_backend='hf'`,
`model/hf.py::HFModel`). Qwen3-VL and Qwen3-VL-MoE sit on the native policy builder
(`model_backend='i4'` with `config.model = NativePolicyBuildSpec(builder=model/qwen3_vl.py::
build_qwen3_vl_native_policy, ...)`, see [the native recipe](../../../cosmos_framework/posttrain/configs/sft_qwen3_vl_moe_native.py)):
the HF backend refuses MoE VLMs because the general weight loader has no fused-expert handling and
Transformers 5 stores the MoE expert tensors transposed relative to a Transformers 4 checkpoint.
On the native seat the parameters are stored in bfloat16 (the MoE expert kernel requires it), so the
recipe uses `attn_impl='sdpa'` and `OptimizerConfig(name='fusedadam')` (apex fp32 master weights and
moments inside the optimizer); `validate_config` refuses `adamw` there because its bfloat16 moments
round most updates to zero. The seat's masking contract is pinned against the official Qwen3-VL
processor template in `tests/test_data_packer_sft.py` (the collated `attention_mask` is `torch.int64`
0/1, the contract `main` pins in `test_model_qwen3_vl.py`; the native eager mask casts it to bool); its
GPU training path reuses the online adapters.

## 2. Quick start

Start the container and install uv 0.11.29 as in
[sections 1 and 2 of the run guide](../howto/run-posttrain.md#1-start-the-container), then run
from the repository root. SFT uses its own venv with the `sft` group
([locked environment project](../../../docker/posttrain/environment/pyproject.toml)):

```bash
# `python3 -m uv` runs the pip-installed uv 0.11.29: a bare `uv` can resolve to an older release
# that ships with the container, which the environment project refuses.
# cu130-runtime matches the run guide's CUDA 13.0 image; use cu128-runtime on a CUDA 12.8 stack.
UV_PROJECT_ENVIRONMENT="$PWD/docker/posttrain/environment/.venv-sft" \
  python3 -m uv sync --project docker/posttrain/environment --frozen \
  --no-group transformers4 --group sft --group cu130-runtime
export SFT_PYTHON="$PWD/docker/posttrain/environment/.venv-sft/bin/python"
export PYTHONPATH=.

# The reviewed Qwen3.8-27B snapshot (about 56 GB).
uvx hf@latest download Qwen/Qwen3.8-27B \
  --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
  --local-dir artifacts/models/Qwen3.8-27B
export SFT_MODEL_PATH="$PWD/artifacts/models/Qwen3.8-27B"
export SFT_DATA_ROOT="$PWD/.ene/scratch/sft-example"
"$SFT_PYTHON" -m cosmos_framework.posttrain.scripts.prepare_sft_example --output "$SFT_DATA_ROOT"
"$SFT_PYTHON" -m cosmos_framework.posttrain.entrypoints.sft --validate-manifest
"$SFT_PYTHON" -m cosmos_framework.posttrain.entrypoints.sft --validate-rows

# N is the allocated GPU count; this command does not allocate resources.
"$SFT_PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$N" \
  -m cosmos_framework.posttrain.entrypoints.sft
```

The generator creates 32 training and 8 validation conversations with local PNG/MP4 media;
it refuses to overwrite a nonempty directory. Video decoding requires TorchCodec 0.11.0
and compatible FFmpeg shared libraries. This is a functional example, not a quality benchmark.
Preflight checks configuration, local file availability and role wiring, not GPU memory or
remote access. `--validate-rows` then runs every row through the real packer on CPU and refuses
over-long rows, masks that disagree with the assistant turns, and mislabelled tool turns before any
trainer is built ([scripts/README.md](../../../cosmos_framework/posttrain/scripts/README.md#pre-launch-sft-row-validator)). Use torchrun even for one GPU; 27B FP32 training does not fit a single ordinary GPU.

Use a complete immutable snapshot, not a mutable Hub branch or metadata-only test fixture.
The reviewed Qwen3.8 revision is `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`; its config selects
`Qwen3_5ForConditionalGeneration` and its processor is `Qwen3VLProcessor`. Preserve the
processor/tokenizer/template sidecars. The loader requires every dense model parameter and
ignores extra `mtp.*` weights; SFT does not train MTP.

## 3. Data contract

One complete conversation per JSONL record:

```json
{"messages":[{"role":"user","content":[{"type":"image","image":"media/frame.png"},{"type":"text","text":"What color is the object?"}]},{"role":"assistant","content":"Red."},{"role":"user","content":"Repeat in uppercase."},{"role":"assistant","content":"RED."}]}
```

- Roles: `system` (first only), `user`, `assistant`, and `tool`. Text is a string or a
  `{"type":"text","text":"..."}` part. Videos use `{"type":"video","video":"..."}`.
- Optional `tools`, assistant `tool_calls`, and `reasoning_content` are retained. Tool-only
  assistant content may be absent/null; it normalizes to empty text.
- Optional `supervise`, a string: `all_assistant` (the rule when the key is absent) or `last_assistant`
  (section 4). Any other value, `null` included, is refused at load with the row's label. The key is
  never added to a row that lacks it, so the fingerprint of an existing file does not change.
- Optional `opener`, a string: `self` (the rule when the key is absent) or `prefilled` (section 4);
  validated and never added the same way. A `meta.opener` (the exporter's record of its prefilled slice)
  must agree with the `opener` field, which is what the packer reads.
- `rl.data.path` and `rl.validation.data.path` accept local JSONL or `s3://bucket/key.jsonl`.
  Remote reads use each source's `credentials` path and the existing S3/GCS client.
- JSONL is read once per rank into memory. Fingerprints cover normalized content and local
  media bytes. Text-only identity excludes URI/credentials and survives cross-cluster moves.
- Local media resolves relative to the JSONL file unless `media_root` is set. S3 JSONL with
  media requires an explicit local `media_root`; automatic remote media fetching is not supported.
- Alternatively, `rl.data.factory` plus `factory_kwargs` selects a map-style source returning
  this schema with an immutable `cursor_fingerprint()` covering rows and media. Additional
  dataset cursor hooks must be supplied as a save/load pair.
- Training row count must divide DP world size. Rows shuffle and repeat by epoch; no training
  records are silently dropped or padded. Validation requires a separate source.

## 4. Objective and masking

The packer renders with the checkpoint's own chat template and passes `preserve_thinking=True`
only to a template that declares that variable (the toy test template does). The official
Qwen3-VL template has no such variable: it renders `reasoning_content` for every assistant turn
and passes inline `<think>` text through as content, so nothing is passed to it; a kwarg the
template does not declare is one Transformers 5 reclassifies as a processor kwarg and warns about
on every render. Every assistant body and turn-ending token is supervised; headers,
user/system/tool content, media tokens, generated video timestamps and padding are ignored.

**Row-level supervision (`supervise`).** A row's `supervise` field selects the assistant
turns that carry loss: `all_assistant` (the default and the behaviour before the field existed: every
assistant body and its `<|im_end|>`) or `last_assistant` (the LAST assistant turn's body through its
`<|im_end|>` only; every earlier assistant turn stays in the sequence as context at -100, like the tool
results and user turns around it). The packer records the row's contract string on
`SFTPolicyInput.masking` (`qwen_chatml_all_assistant_bodies_and_end_v2` or
`qwen_chatml_last_assistant_body_and_end_v1`); the packer-level resume identity keeps the all-assistant
string because the field is part of the row and therefore of the source fingerprint.
`rl.mask_think_opener` applies to the supervised turn only (the per-row `opener` field below applies the
same rule to one row). The rule exists for **prefix rows** (a term
defined here): one row per model call of an agent-harness episode, whose prompt is the served prefix of
that call -- the tokenizer template's rendering of the history, every earlier assistant turn
reasoning-free, exactly what the default serve renders when the request carries a tools array (measured
equal to the serving deployment's tokenize endpoint on 146 of 146 calls of one agent-harness episode) --
and whose completion is the student's own turn at that call, reasoning included.
A prefix row therefore matches the **default serve** of a default export token for token; a
whole-episode row with reasoning in every assistant turn does not (it is exact only under a serve that
renders the processor template with tools: a **unified export**, defined in section 7 as an export built
with the materializer's `--unified-chat-template`, whose two template files carry the canonical processor
template). Fixtures:
`tests/test_data_packer_sft.py::test_supervise_last_assistant_three_turn_row_exact_loss_counts`
(a three-turn row through the toy tokenizer: 96 = [91, 5] loss tokens under the default, 5 = [5] under
`last_assistant`, the ids identical) and
`test_supervise_last_assistant_official_template_keeps_history_at_minus_100` (the official template
with an image and a tool call; the opener mask then touches the last turn's opener only). Under
`last_assistant` the history turns must carry no reasoning: the packer refuses a history assistant turn
with a non-empty `reasoning_content` or a content opening with `<think>` (the processor template renders
a history `reasoning_content` as a think block, 250 vs 236 ids on one probe, while the default serve's
prefix has none, and it strips an exact `<think></think>` content prefix the tokenizer template keeps).
Fixture:
`test_last_assistant_row_prefix_is_the_served_prefix_under_both_templates_and_history_reasoning_is_refused`
(the base's two templates as repo fixtures: the row's text and ids begin with the tokenizer template's
served prefix, the two templates agree byte for byte when no turn carries reasoning, a history
`reasoning_content` or `<think></think>` prefix diverges inside the history and is refused, an
`all_assistant` row with the same history still packs).

**Row-level opener (`opener`).** A row's `opener` field says whose token the
`<think>` opener of a supervised assistant turn is: `self` (the default and the behaviour before the
field: the opener is the student's first supervised token, as the agent harness's serve prefills nothing)
or `prefilled` (the serve prefills the opener token itself, `<|im_start|>assistant\n<think>`, id 151667, no
newline after it, as a thinking-on client without tools does: the packer masks the first `<think>` token
of every supervised assistant block of that row, exactly the rule the run-level `rl.mask_think_opener=True`
applies to every row, and the supervised target begins with the newline that opens the reasoning text;
under `last_assistant` that is the last block only). One rule: a row with `opener: prefilled` in a run
with `mask_think_opener=True` is masked once, the count unchanged. Why per row: the exporter writes the
prefilled slice (20 percent of the supervised reasoning turns, a fixed seed) into the same row files
as the self rows, so the mask must be the row's, not the run's; the exporter records its choice as
`meta.opener` and must write the same value in the row's `opener` field, which is what the packer reads
(the loader refuses a `meta.opener` that disagrees with the field, a `prefilled` in `meta` without the
field included). The per-row contract string gains `_opener_prefilled_v1`
(`qwen_chatml_all_assistant_bodies_and_end_v2_opener_prefilled_v1`,
`qwen_chatml_last_assistant_body_and_end_v1_opener_prefilled_v1`); the packer-level identity keeps
recording the run-level knob. A `prefilled` row on a tokenizer without `<think>` is refused. A `prefilled`
row needs a reasoning block in every supervised assistant turn: the packer
refuses the row unless its mask removed exactly one `<think>` id in every supervised block (the refusal
names the bare blocks and the per-block counts), and the loader refuses a `prefilled` row whose supervised
turns (all of them under `all_assistant`, the last under `last_assistant`) lack a non-empty
`reasoning_content.strip()`, so the string `_opener_prefilled_v1` is only ever recorded when the masking
happened; the run-level knob on a bare turn masks nothing and refuses nothing (it is the run's variant,
recorded in the packer identity, not the row's contract). The loader also refuses a `meta.supervise` that
disagrees with the row's `supervise` field, as it does for `meta.opener`. Fixtures:
`tests/test_data_packer_sft.py::test_opener_prefilled_three_turn_row_exact_loss_counts_under_the_four_combinations`
(a three-turn row through the toy tokenizer: with reasoning in both assistant turns under `all_assistant`,
`self` 118 = [102, 16] and `prefilled` 116 = [101, 15]; under `last_assistant`, with the history turn
reasoning-free as that mode requires, 16 and 15; the difference exactly the opener token of each
supervised block), `test_opener_prefilled_and_the_global_mask_are_one_rule`,
`test_opener_refuses_unknown_values_and_a_tokenizer_without_think`,
`test_opener_prefilled_refuses_a_supervised_block_without_an_opener` (B1 / B2 / B3 refused, self rows and the
run-level knob unchanged),
`tests/test_roles_env_sft.py::test_opener_field_is_validated_by_the_loader_and_reaches_the_target` and
`::test_opener_prefilled_rows_require_reasoning_on_every_supervised_turn_and_meta_supervise_must_agree`. A row
validator that counts assistant tokens independently of the labels must count over
`data/packer/sft.py::supervised_assistant_blocks(n_assistant_blocks, supervise)`, the indices the packer
paints, or it reports every `last_assistant` row as a mask mismatch.
`rl.mask_think_opener=True` additionally unsupervises the `<think>` token that opens an assistant
body, and only that token. Keep the default `False`: on the serving path of record (Codex
`--thinking high` sends `reasoning.effort` and the harness's tools array on every call) vLLM 0.24
prefills no opener -- with tools present the served prefix is `<|im_start|>assistant\n` and nothing
more (the effort field adds 0 input tokens; `/tokenize` with tools and `enable_thinking` ends in
`assistant\n`; measured live; the ids are the fixture
`tests/fixtures/served_prefix_with_tools.json`), so the student must emit `<think>` itself as the
first token of every turn and then the reasoning, or the closer `\n\n</think>\n\n` of the empty
block a turn without reasoning (a no-think turn) is rendered with; all of it is the student's and
the opener's loss is the only training signal for that token. `True` would leave it unsupervised;
what a `True`-trained student would emit there is UNMEASURED with tools present (no such student
exists, and the first token after `assistant\n` with tools present was never sampled), so the
consequence is a RISK, not a measurement. The failure it could reopen has a measured trigger: the
qwen3 parser treats an output that carries no `</think>` as reasoning in full, so a no-think TEXT
turn of the student never reaches Codex as an agent message, while a no-think TOOL-CALL turn is
unaffected (vLLM 0.24 `reasoning/basic_parsers.py` 173-181: without `</think>` the whole output is
reasoning; with `</think>` it is split even without `<think>`); under the exporter's rendering the
closer stays supervised under `True`, so a turn that dropped only the opener would still carry
`</think>`. The decision does not rest on the risk: `False` stays the default because the opener's
loss is the only training signal for a token the student must emit on the path of record. Selecting
the `prefilled_opener` variant sets the typed field `rl.serve_contract=prefills_opener` beside the
mask: the declaration that the student will be served by a path that prefills `<think>` for it,
recorded in the resolved config and, being non-default, in the objective identity. `SFTConfig`
refuses the VALUE `rl.mask_think_opener=True` under the default contract `tools_no_prefill` from any
source, a `--` override on the `default` variant included, and the resolved variant name is stamped
into the attempt's provenance as `recipe_variant`. The prefill exists only for requests without
tools: there a present effort sets `enable_thinking` and the template appends `<think>` (10 input
tokens instead of 9) -- not a path of record; under such a serve the supervised opener costs one
deterministic token per turn and the student continues after it, so `False` is right under both
conditions. (The Nemo-Chat processor rule also unsupervises an empty block, because its serve
supplies the whole block; that rule, applied here, would remove the closer's loss under a single-newline
rendering.) The native recipe's table (its `<think>` opener row) records the decision with its
measurements. Templates that drop supplied reasoning or reorder messages are rejected, as are
reserved chat/media delimiters in source text.

Masks use the **final processor-expanded token stream**, requiring an exact tokenizer round
trip; no prefix-length approximation or pre-expansion assistant mask is used. Labels are
unshifted at collation; the objective shifts exactly once. With `rl.ce_exponent=0` (the framework
default) the loss is the global supervised-token mean across ranks and accumulation; with
`ce_exponent=e` every row contributes its token sum divided by `n^e` over the DP-mean of
`sum_rows n^(1-e)` (`0.5` = the Cosmos post-training recipe's square-root weighting, `1` = every row
weighs the same). `rl.moe_lbl_coeff>0` adds the MoE load-balancing term
`coeff * E * mean_layers(sum_e f_e * P_e)` from the sparse blocks' routing statistics of the same
forward (`f_e` the share of routed tokens, summed over the DP ranks under `moe_lbl_method='global'`;
`P_e` the mean router probability); `train/lbl_loss`, `train/moe_load_max_rel` (the largest
expert's share relative to uniform, 1.0 = balanced) and `train/objective` (the value descended)
then appear beside `train/loss_avg`, which stays the plain token-mean CE. Overlength conversations
fail rather than truncating targets.

## 5. Main knobs

Global conversations/update are `rl.micro_batch_size × rl.grad_accum × DP world size`.
`rl.max_steps` counts optimizer updates; `rl.max_length` limits the full expanded sequence.
`rl.optimizer.*` controls the optimizer, clipping and warmup (`warmup_steps` must be `>= 0`);
`rl.lr_schedule.kind` selects what follows the warmup: `constant` (the pre-existing path) or
`cosine` (the Cosmos post-training recipe's `LambdaWarmUpCosineScheduler`: lr × `f_start` at the
first update, linear to lr at `warmup_steps`, cosine down to lr × `f_min` at `cycle_steps` or
`max_steps`). `rl.optimizer.lr_multipliers` scales the learning rate of the parameters whose name
contains a key (`{"model.visual": 0.1}` = the vision tower at a tenth). Precision differs by seat.
The dense HF recipe keeps FP32 parameters and optimizer state (`master_dtype='float32'`) under BF16
autocast. The native Qwen3-VL-MoE recipe stores the parameters in bfloat16
(`master_dtype='bfloat16'`, `param_dtype='bfloat16'`); its fp32 master weights and fp32 moments
live inside the apex fused Adam (`OptimizerConfig(name='fusedadam')`), which refreshes the bf16
parameters from the master on every step, and the saved artifact is that bf16 copy, never the
master: an update below half a bf16 ulp of a weight is carried in the master during the run and
reaches neither the artifact nor the training forward, which reads the same bf16 copy. Known limit,
measured over 1,401 training steps at lr 1e-5: `model.language_model.norm.weight`
(median |w| 3.08, bf16 ulp 0.0156 > 1,401 × 1e-5 = 0.014) changed in 0 of 2,048 elements while
`layers.0.input_layernorm.weight` (median 0.0036) changed in 1,985 of 2,048; 36 of the 882 tensors
(34 vision norms, `layers.47.input_layernorm.weight`, the final norm) were bitwise the base's, i.e.
those 36 norm scales did not train. Trade-off: the served model is bf16 either way, and storing the
1-D parameters (the 426 norm scales and biases, a few MB in fp32) in fp32 would un-freeze them at
negligible memory cost, the export rounding each once (at most half an ulp lost). The materializer
records per export how many tensors equal the base (`tensors_equal_to_base` in `.materialized`).
`tests/test_config_sft.py::test_bf16_storage_freezes_large_scales_below_half_an_ulp` is a pin of the
mechanism on a tiny fixture (it records pure bf16 arithmetic, which the framework does not change), as
is `test_deterministic_knob_reaches_setup` (pre-existing setup behaviour); neither tests a change to
the framework. Both recipes use activation checkpointing and SDPA. Weight decay is decoupled;
`decay_groups=''` builds one parameter group (every trainable parameter decays, norms, biases,
embeddings and routers included -- the Cosmos post-training recipe's own grouping:
`cosmos3/utils/optimizer.py::_build_params_with_metadata` decays all 882 tensors of the backbone),
`decay_groups='hf'` reproduces the Transformers Trainer's [decay, no-decay] split (LayerNorm modules,
biases and `*norm*` names undecayed; the embedding, the head and the router gates decay), and
`decay_groups='qwen_moe'` (the native recipe) leaves every 1-D parameter (norm scales, biases) AND
the token embedding, the output head and the MoE router gates undecayed -- the seat's own choice,
whose protection and cost the recipe's table states in its weight-decay row. The native recipe's
docstring carries the per-knob decision table (Cosmos post-training recipe vs the A1 arms vs its
defaults).

`rl.processor_kwargs` accepts HF `images_kwargs` and `videos_kwargs`. Pixel budgets use
`size.shortest_edge`/`longest_edge` (pixel area), not legacy `min_pixels`/`max_pixels`.
The example samples 8 video frames with `fps=None`; do not also request FPS-based sampling.
The processor's media tensors and `mm_token_type_ids` reach the model unchanged. The Qwen3.5
forward adapter orders image/video vision calls and backward graphs on every rank, including
dummy work for absent modalities; dummy grids never enter token positions or M-RoPE.

## 6. Metrics and validation

`train/loss_avg` and `val/loss_avg` are token-weighted means; valid token/sequence counts
are sums. Every committed frame is written to `<run dir>/metrics.jsonl` (one JSON line per step:
`train/grad_norm` = the pre-clip global norm, `train/learning_rate`, `train/valid_token_count`,
`timer/e2e_step_s`, `train/grad_clip_coef` = the factor `clip_grad_norm_` applied,
`min(1, grad_clip_norm / (norm + 1e-6))`, and `train/grad_clipped` = 1.0 exactly when that factor is
below 1 (one rule; both recorded by `LLMTrainer` from the norm it clipped),
`train/grad_clip_fraction` = the share of the run's updates clipped so far (the loop's running count),
plus the validation keys of that step) and the console line carries the headline subset. The file
holds one row per step across relaunches: an attempt that resumes at step `s` keeps the rows with
`step <= s` and drops the later ones, which it re-emits (a crash past the last save leaves committed
rows the resumed attempt recomputes), and the clip counters continue from the kept rows, so the
fraction is the run's, not the attempt's; a fresh attempt starts the file empty. Held-out validation
uses finite batches with zero-weight tail padding and restores module modes, source cursor and RNGs.
Final validation is mandatory when enabled. The validation source must be held out row by row: setup
refuses it when any validation row has the content hash (messages + tools, media by file bytes so a
copy of the training media under another root is the same row) of a training row or the item id
(`meta.item`) of one, and when validation rows carry no item id while the training rows do (a row
without an item has only its content hash as identity, which one changed character defeats); a
whole-source fingerprint check is not enough, since it passes a strict subset of the training file.

On the native Qwen3-VL-MoE seat, `train/grad_norm` varies from run to run because routing in the bf16
MoE forward is not deterministic. Read it as a per-run diagnostic, not as a quantity to compare
between two runs.

## 7. Checkpoints, resume and export

The generic example saves DCP every 10 updates and at its final step. Obj1k instead selects
consolidated `model.pt` for HF export. `rl.resume=auto` resumes the latest checkpoint for the
same job identity; an explicit iteration directory also works. Exact resume binds every rank's
data cursor, tokenizer/processor/masking semantics, precision and batching. Preemption saves
only at complete optimizer boundaries.

For storage and retention see [checkpointing](../reference/checkpoint.md). The
[HF materializer](../howto/checkpoint-artifacts.md#materialize-an-opd-teacher) consumes consolidated
`model.pt`, not DCP; it preserves tokenizer/processor sidecars, writes the base's `config.json` into the
export (the transformers-written one is kept unpublished as `config.json.trainer.bak`, the export's own file
before the base overlay), and verifies every tensor of `model.pt` bitwise against the shards before it
returns, keeping a sha256 per tensor and the list of tensors bitwise equal to the base in `.materialized`. A
`processor_config.json` in the export is the base's own file or none (the reference base carries none);
`conversion.hf.service_processor_config` composes one from the base's sidecars but has no production caller,
and the file served beside one deployed export (placed by hand) differs from it in the image processor class
and 17 explicit defaults, so connecting the helper is a separate change: a contract bump, `verify_sidecars`
accepting the composed file and a re-anchored serve. Select `rl.checkpoint_format=consolidated` if export is
required and budget rank-0 host memory. `rl.export_*` is not wired and is rejected.

**Unified chat template (`--unified-chat-template`).** The base ships two chat templates: `chat_template.json`
(the **processor template**, 5,614 chars, text sha256 `97de86bf6ddb765a...`) and `tokenizer_config.json`'s
`chat_template` (the **tokenizer template**, 5,292 chars, `3636d0f0bd6bef02...`). They differ in what they
render for an assistant turn in the HISTORY: the processor template renders `reasoning_content`
(`<think>...</think>`) in every assistant turn, the tokenizer template renders none. vLLM 0.24 renders a
request that carries a tools array with the tokenizer template (measured: a captured agent-harness request,
req_012, tokenizes to 8,946 ids on the default serve and to 8,969 with the processor template forced, the 23
extra ids an earlier turn's think block; the first differing id sits at position 7,247, `<think>` vs
`<tool_call>`). The materializer option writes BOTH template files of the export, `tokenizer_config.json`'s
`chat_template` and `chat_template.json`'s `chat_template`, as the processor template in its CANONICAL form,
every other key of each file unchanged (every other line byte-equal when the base's file is a plain
`json.dumps` rendering, which both of the reference base's are). The **canonical form**
(`template_mode: processor-canonical`) is the processor template with ONE change in its reasoning branch,
`'<think>\n' ~ (message.reasoning_content | trim) ~ '\n</think>\n'` instead of the untrimmed rendering: a
replayed reasoning string that arrives with a newline at both ends (the agent harness's form) then renders the
canonical block `<think>\n<text>\n</think>\n`, the student's own emission form, instead of the padded
`<think>\n\n<text>\n\n</think>\n` (`conversion.hf.canonicalize_processor_template`; a base whose processor
template lacks the exact line is refused). The marker records the processor template's sha, the shipped sha of
each file (`shipped`) and the one-hunk diff under `chat_template` (`template_mode`, `files`, `canonical_diff`;
also `options.unified_chat_template` and `sidecars.chat_template`), makes the option part of the output's
identity (a request without it on such an output is refused until `--force`, and the reverse) and verifies
both files by rule instead of byte identity (`verify_sidecars`: each file's mapping equals the base's on every
key but `chat_template`, whose value is the canonical text; `verify_with_torch`: the loaded tokenizer's and
the loaded processor's templates are that text). Default off: the default export stays byte-identical to the
base's sidecars. The rows for this template write `reasoning_content` as the model's TRIMMED reasoning text
(the `str.strip()` of the model's reasoning bytes; the packer renders the base's own processor template, which
yields the same canonical block for a stripped string, so training bytes are canonical with or without the
option), and no-reasoning turns stay BARE, no think block (matching the student's measured emission and the
serve's history form). Serving condition: **unified rows** (a term defined here: whole-episode rows in which
every assistant turn is supervised and the history turns carry their reasoning, the row shape of a unified
export's training set) are exact under ANY serve of a unified export and without a serve-side template flag:
both template files of a unified export carry the canonical text, so a request with or without tools renders
the same reasoning form (a tokenizer-only loader reads `tokenizer_config.json`, a processor loader reads
`chat_template.json`). Two row conditions carry this claim, both measured on the captured request req_012: (a)
tool-call `arguments` travel in the row as the PARSED OBJECT, which the template renders through `tojson`
(`{"query": "value"}`, with `": "` and `", "`), exactly as the serve re-serialises the parsed arguments; a row
that keeps the model's compact JSON string (`{"query":"value"}`) renders it verbatim and differs from the
served prefix at the first argument (3 of 3 argument strings of req_012 differ from their re-serialised form);
the packer also accepts `arguments` as a string and renders it verbatim, and such a string matches only when
it equals `json.dumps(obj, ensure_ascii=False)` of its parsed object (the `tojson` form the serve emits): the
model's compact string and `json.dumps(obj)` with the default `ensure_ascii=True` (non-ASCII escaped as
`\uXXXX`) both differ; (b) `reasoning_content` is the `str.strip()` of the model's reasoning bytes (Jinja's
`trim`), not the text with one newline removed at each end: a trailing space or a second newline survives the
latter and breaks the match. Pins: the string-vs-object and whitespace cases of
`tests/test_data_packer_sft.py::test_processor_canonical_template_trims_the_replayed_reasoning_where_the_verbatim_template_pads_it`
(the `ensure_ascii=True` case included). Supported-base contract under the option: a base ships its templates
in the two JSON files, `chat_template.json` and `tokenizer_config.json`'s `chat_template`, as the evidenced
Cosmos base does, and carries NO template sidecar a loader prefers over them; the materializer's preflight
enforces it. The materializer's preflight is the torch-free set of checks in
`checkpoint/tools/materialize_hf.py` that read the base alone before anything is removed or written (not the
trainer's preflight of section 2, which checks the training configuration). Read on transformers 4.57.6 and
5.8.1, the two pinned versions, those are `chat_template.jinja` (the tokenizer loader takes it over
`tokenizer_config.json`'s key; the processor loader reads it only when `chat_template.json` is absent), the
`additional_chat_templates/` directory (named templates the tokenizer loader prefers over the key; beside
`chat_template.json` the processor loader refuses the combination; the directory is refused whatever it holds,
empty or with nested directories only included, by the one helper the materializer's preflight and
`verify_sidecars` share: both loaders discover named templates with `glob("*.jinja")` and ignore an empty
directory, and the rule still refuses it, fail closed) and a `chat_template` key in `processor_config.json`
(the processor loader prefers it on 5.8.1). With such a sidecar the export would ship both canonical JSON
files while `AutoTokenizer` renders the sidecar, so the materializer REFUSES the base in unified mode before
anything is published, with or without `--no-verify`, naming the file. That refusal is part of the
materializer's preflight, which runs before any destructive step: under `--force` the existing output is
removed, `<out>.tmp` created and the overlay run only after it has passed, so a refused request leaves the
existing output byte-identical. The same preflight also parses every JSON sidecar of the base
(`tokenizer_config.json` and `chat_template.json` as objects; the overlay and the template record read them
after the overlay), refuses in the default path an `additional_chat_templates/` whose entries are not all
regular files (the default path copies the directory verbatim and its verifier refuses a sub-directory), and
turns an unreadable base into a named refusal instead of a traceback (fixtures
`tests/test_checkpoint_materialize_hf.py::test_force_rebuild_keeps_the_existing_output_when_a_base_json_sidecar_does_not_parse`,
`::test_default_preflight_refuses_a_named_template_directory_with_entries_that_are_not_regular_files` and
`::test_preflight_refuses_an_unreadable_base_with_a_named_reason_instead_of_a_traceback`). The same preflight
also runs on every base shard the index lists the safetensors header check the output gets
(`conversion.hf.shard_status`: present, readable, header parsed, size = 8 + header + data, header keys = the
tensors the index maps to it), because the overlay's `from_pretrained` opens every shard in the staged build
and a shard truncated by one byte, with a garbage header or unreadable would otherwise pass (the identity
records sizes only); requires every JSON sidecar of the base to be a JSON object and `chat_template.json`,
when present, to carry a non-empty `chat_template` string (the loaders index these after the overlay: a `[]`
sidecar fails `AutoProcessor` / `GenerationConfig` and a `{}` template file is the processor loader's
KeyError, read on the container's transformers 5.13.0); validates the index's `weight_map` (a non-empty object
of tensor name -> shard name, so no index shape escapes as AttributeError / TypeError); and raises every
refusal in the one message family
`REFUSED before anything is published: <what was found; what the staged build would have done with it>`
(fixtures
`::test_preflight_refuses_a_base_shard_the_safetensors_header_check_rejects_before_anything_is_removed`,
`::test_preflight_refuses_a_json_sidecar_the_loaders_reject_although_it_parses`,
`::test_preflight_refuses_an_index_whose_weight_map_is_not_tensor_to_shard_names_without_a_traceback` and
`::test_every_preflight_refusal_is_in_the_one_message_family_and_names_what_it_found`). ONE JSON decoder
serves the whole preflight and the materializer's own readers (`conversion.hf.decode_json_bytes`: strict UTF-8
text, what `model_identity`, `_read_json_file` and transformers' loaders decode in the staged build), so a
UTF-8 byte-order mark in front of valid JSON (config.json, any JSON sidecar, the index) is refused by name
instead of passing through `json.loads(bytes)`, which strips it (on the reference base's files a leading mark
fails AutoConfig, AutoProcessor, AutoTokenizer and GenerationConfig: `Unexpected UTF-8 BOM` /
`not a valid JSON file`); and `shard_status` validates the safetensors header's shape (a JSON object of tensor
entries with a dtype string, a shape list and `data_offsets` of two non-negative integers begin <= end, an
optional `__metadata__` object), so a malformed header is a named refusal in the preflight and a named failure
of the output verifier, never a traceback (fixtures
`::test_preflight_refuses_a_utf8_byte_order_mark_in_front_of_valid_json_as_the_loaders_do` and
`::test_shard_status_refuses_a_header_without_the_safetensors_shape_and_the_preflight_names_it`). The
construction is STAGED. The whole build happens in `<out>.tmp` beside an existing output: the overlay (the
base read by its real readers: the pinned safetensors reader on every shard, transformers on the sidecars),
the sidecar sync, `config.json <- base`, the verification (index, sidecars and, unless `--no-verify`, the
torch probe, all on the staged copy) and the `.materialized` marker, the last file written there (its
semantics unchanged); only then the swap replaces the existing output by renames (`<out>` -> `<out>.previous`,
`<out>.tmp` -> `<out>`, `<out>.previous` removed; `materialize_hf.publish_staging`). On any failure before the
swap the staging directory is removed, the error names its stage and the existing output is byte-identical,
under `--force --no-verify` too. Transient disk: two exports side by side from the overlay to the swap (the
staged shards are written by `save_pretrained`, not linked). A leftover `<out>.previous` is refused before the
build unless `--force`, which discards it. Before the staging, the same preflight asks the pinned safetensors
reader itself (`conversion.hf.safetensors_reader_error`: `safe_open` on the header of every base shard the
index lists; dtype names, string metadata values, contiguous offsets, byte count per tensor, file length: a
BF16 `[3]` tensor over four bytes, an unknown dtype, a non-string metadata value and an offset gap all passed
the shape check) and decodes, strictly as UTF-8, every Jinja template the tokenizer and processor loaders read
(`chat_template.jinja` and each `additional_chat_templates/*.jinja` named template;
`conversion.hf.loader_template_files` cites the loader lines), refusing by file and reader message / decode
position in the one family (fixtures
`::test_preflight_refuses_a_base_shard_the_pinned_safetensors_reader_rejects_and_keeps_the_export`,
`::test_preflight_decodes_the_jinja_templates_the_loaders_read_as_strict_utf8_and_keeps_the_export`,
`::test_a_failure_after_the_preflight_keeps_the_existing_export_and_removes_the_staging_directory`,
`::test_the_swap_publishes_by_rename_and_a_leftover_previous_output_is_refused_without_force` and
`::test_preflight_refuses_when_the_pinned_safetensors_reader_cannot_be_imported`). Under `--force` no complete
marked tree is discarded while it is the only one: in the crash state between the swap's two renames
(`<out>.previous` = the old export, `<out>.tmp` = the new one complete and marked, no `<out>`)
`<out>.previous` is renamed back to `<out>` first and the build happens beside it; a leftover `<out>.previous`
beside an existing `<out>` is removed at the swap, after the new tree is verified. `--force` rebuilds from the
source; a death-left `<out>.tmp` is discarded once `<out>` is complete again whatever its marker says (a
marker match covers the identity, the shard sizes and tensor counts, not the shard bytes, and the base digest
excludes the base shards' bytes), a complete marked `<out>.tmp` that is the only complete tree is refused by
name rather than discarded, and a kept staged copy is published only by a request without `--force`, after the
same verification as a fresh build (index, sidecars, the torch probe unless `--no-verify`, which trusts the
kept copy's bytes on size and tensor count). One materializer per output: `<out>.lock` beside the output, held
with `fcntl.flock` from before the staging to after the swap; a second materializer on the same `--out` is
refused by name (pid, host, start time of the holder) without touching anything; the lock file stays beside
the output (it is not part of the export) and the file system must support flock (Lustre: the `flock` mount
option; a mount that refuses it is a named refusal). Fixtures
`::test_force_after_a_death_between_the_swap_renames_restores_the_old_export_before_building`,
`::test_two_materializers_on_one_output_exclude_each_other_through_the_per_output_lock`,
`::test_the_per_output_lock_refuses_by_name_in_process_and_fails_closed_without_flock`,
`::test_force_rebuilds_from_the_source_instead_of_publishing_a_death_left_copy` and
`::test_a_death_left_copy_is_published_only_by_a_request_without_force_after_verification`. An inherited sole
complete marked `<out>.tmp` (a first publication that died after `.materialized` and before the rename) is
never deleted: the unforced recovery checks its marker against the request (else refused by name), repairs a
missing staged `config.json` from the base through a temporary file and a rename, re-syncs the sidecars the
same way (never a shard), then verifies, re-marks and swaps; a failed recovery check refuses by name and KEEPS
the tree (its shards and its marker untouched, its sidecars possibly repaired from the base before the check:
config.json rewritten, a differing one kept as `config.json.trainer.bak`, a stray sidecar removed, a
`.copying` or `.writing` temporary left by an interrupted copy or unified-template rewrite, and under
`--unified-chat-template` the two template files back in the base's form when that rewrite did not reach its
rename; the refusal says so and names the state at `<out>`: absent, standing but not a complete marked export,
or a complete marked export), and the overlay is never run while the copy is the sole complete tree. The
marker / skip / mismatch decision is taken under the per-output lock, so a materializer that paused before the
lock while another published a different source is refused on that export's marker instead of overwriting it;
a held lock is waited for up to 120 s (`--lock-wait-seconds`, polling), so two simultaneous identical requests
of a published source both end with 0 (the second skips once the holder has finished), a differing request is
refused on the marker, and a holder still there after the bound is refused by name with the bound stated.
Fixtures: `::test_a_failed_recovery_check_leaves_the_inherited_sole_marked_copy_as_found`,
`::test_the_marker_decision_is_taken_under_the_output_lock_so_a_paused_materializer_cannot_overwrite_a_newer_export`,
`::test_a_repair_then_failed_recovery_check_keeps_the_shards_and_the_marker_and_says_what_it_changed`,
`::test_an_identical_request_waits_for_the_lock_holder_and_skips_with_0`,
`::test_the_lock_wait_is_bounded_and_the_refusal_names_the_bound` and
`::test_a_failed_unified_rewrite_during_recovery_leaves_the_writing_temporary_and_the_refusal_names_it`; the
lock bound must be a finite number >= 0 (`inf` and `nan` are refused by name:
`::test_lock_wait_seconds_refuses_non_finite_values`). The source's identity of record (sha256 + bytes of
`model.pt`) is read UNDER the output lock, not before the wait, through a file object opened once there, and
the overlay memory-maps that SAME open file (through `/proc/self/fd`; where that reference does not exist the
loader reads the payload whole through the file object, no mmap, with the whole payload in RAM), so by
construction the skip / mismatch decision, the source record and the marker name the bytes the overlay
consumed, and the default verification (the torch probe) compares the staged shards against that same open
file (a path replaced during the overlay can neither fail a correct build nor pass a wrong one), whatever the
path names meanwhile: the merge tool (tmp + os.replace over `model.pt`) is not excluded by the lock and may
replace the path while the request waits (adopted with a WARNING naming both hashes) or while the overlay runs
(logged; nothing of what was built changes). The opened inode is fstat'd before the hash and after the
overlay: a change of its size or mtime refuses the build by name, with no "same bytes" acceptance, since the
digest of the file after the fact does not say what a reader consumed; that check is best effort against a
writer into the opened inode (a client may report an open inode's attributes late), which the merge tool never
is: the guarantee of record is against a replaced path, by construction (fixtures
`::test_the_marker_records_the_source_the_overlay_read_when_the_source_changes_during_the_lock_wait`,
`::test_a_source_replaced_during_the_overlay_does_not_change_what_was_built_or_recorded`,
`::test_the_marker_names_the_bytes_the_overlay_consumed_when_forced_merges_race_the_overlay` with the real
merge tool, `::test_the_default_verification_receives_the_held_payload`, the verification suite's
`test_checkpoint_hf_verification.py::test_default_verification_consumes_the_held_payload_when_the_path_is_replaced`
with the real merge tool and the real torch verifier, and the loader's
`test_checkpoint_hf_overlay.py::test_load_consolidated_state_dict_reads_the_open_file_not_the_path`). The
weights-free loader calls themselves (`AutoProcessor` / `AutoTokenizer` / `GenerationConfig` `from_pretrained`
on the base) are NOT part of the preflight, by design: they are not torch-free (the processor instantiates the
image and video processors, which import torch), their verdicts depend on the installed transformers rather
than on the files, and the synthetic bases of the tests are not loadable by them; a base that passes every
file check but fails a loader on a schema fact the files do not state (`tokenizer.json` = `{}`, a
`preprocessor_config.json` without `image_processor_type`) is still refused by the overlay, in the staged
build, with the existing output kept. A `chat_template.jinja` (or a `processor_config.json` key) whose text
equals the canonical text is allowed, synced verbatim and recorded under `chat_template.preferred_sidecars`;
`verify_sidecars` applies the same rule to the export. Fixtures:
`tests/test_checkpoint_materialize_hf.py::test_unified_chat_template_refuses_a_base_whose_preferred_template_sidecar_diverges`
and `tests/test_data_packer_sft.py::test_unified_export_tokenizer_loads_the_canonical_text_and_a_divergent_jinja_sidecar_is_refused`
(a real tokenizer load from a unified export of a toy base: canonical; with the Jinja beside it: the Jinja);
the directory rule and the preflight order:
`tests/test_checkpoint_materialize_hf.py::test_unified_template_directory_rule_is_one_helper_for_preflight_and_verifier`
and `tests/test_checkpoint_materialize_hf.py::test_force_rebuild_keeps_the_existing_output_when_a_torch_free_preflight_refuses`.
Prefix rows (section 4) are exact under the default serve of a default export. Offline proof on the real
base's sidecars: the tokenizer loaded from a unified scratch export renders req_012 (messages normalised as
vLLM does: `developer` to `system`, `content: null` to `""`, `reasoning` as `reasoning_content`, argument
strings parsed, the image placeholder expanded to its 209 tokens; its two replayed reasoning strings begin and
end with a newline) to 8,969 ids with positions 7248, 7254, 8016, 8025 reading 198, 624, 198, 624 (the
stripped form: `\n` and `.\n`); the verbatim processor template renders 8,969 ids with 271, 382, 271, 382 at
the same positions (the padded form: `\n\n` and `.\n\n`), pinned beside it; the base's tokenizer renders
8,946. Consequence: the shared-vocabulary identity of `config/build.py` hashes the tokenizer's
`chat_template`, so a unified export is not an OPD teacher for a student loaded from the default sidecars (and
the reverse); pair teacher and student on the same choice. Serve assumption, recorded in `.materialized`
beside the shas (`chat_template.serve_assumption`): the serve sets no `enable_thinking`. Under the unified
template a request that sets `enable_thinking=true` is prefilled with the opener token itself,
`<|im_start|>assistant\n<think>` (id 151667, no newline after it: 251 vs 250 ids; the same tail without tools,
`\n`, `<|im_start|>`, `assistant`, `\n`, `<think>`, 10 ids against 9), while the rows supervise the opener as
the student's first token; under such a serve `rl.mask_think_opener=True` masks exactly that one token and the
supervised target begins with the newline that opens the reasoning text (the rule of an `opener: prefilled`
row, section 4). The agent harness sends no `enable_thinking`, and the student's serve recipe must not set it.
Fixtures:
`tests/test_checkpoint_materialize_hf.py::test_unified_chat_template_rewrites_the_tokenizer_template_only_and_is_verified_by_rule`
and `test_main_unified_chat_template_option_identity_and_skip`; the padded and stripped forms side by side on
the official template through the toy tokenizer (same count, `\n\n` tokens become `\n` tokens; a trimmed
string renders identically under both):
`tests/test_data_packer_sft.py::test_processor_canonical_template_trims_the_replayed_reasoning_where_the_verbatim_template_pads_it`.

## 8. Extension points

Keep tensor shaping in `data/packer/sft.py`, objective math in `algorithm/sft.py`, model
adaptation in `model/qwen3_5.py`, and lifecycle orchestration in `entrypoints/sft.py`.
