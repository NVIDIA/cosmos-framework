# DataPacker — the raw → tensor shaping seam

**Come here when you are bringing up a model architecture the framework has no packer for.** A
DataPacker owns the raw-sample to model-ready-tensor step, and is the one place tokenizer, chat
template, placeholder and mask semantics are defined for a model.

You do *not* need this to change datasets — that is [`custom-data.md`](custom-data.md). Every
capability guide depends on it through the rollout and policy paths — see
[`../guides/grpo.md`](../guides/grpo.md) for the most common case.

One object owns *all* raw-sample → model-ready-tensor shaping for a model:
tokenization, prompt templating, packing, collation. Rollout bodies and objectives
call it instead of tokenizing/tensorizing inline, so "how this model eats data" lives
in exactly one place and a new architecture (VLM/VLA) drops in as one new file
instead of edits scattered across the rollout body and every objective.

The seam is deliberately narrow:

- **Weight-free and CPU-side.** The packer never runs a model. It stops at
  model-ready inputs — `DecoderOnlyLLMDataPacker.policy_collate_fn` returns **CPU**
  tensors and the trainer moves them to device. The weight-dependent VAE
  pixels→latents encode stays in the model's forward (`model/cosmos3.py::encode_and_condition`),
  which is why the packer stays parallelizable off the GPU.
- **It does not own heavy-payload transport.** `comm/codec.py`'s metadata/bulk split
  already externalizes big tensors (DESIGN §3.4); `get_rollout_output` is an
  output-*shaping* hook, not an offload hook.

Mechanism-level truth (invariants, alignment arguments) lives in the docstrings of
`data/packer/base.py` and `data/packer/decoder_only_llm.py`; this doc is the user/ops
view. Design rationale and history: internal design record.
Run-local factory configuration: [`custom-data.md`](custom-data.md).

## The contract

`BaseDataPacker` (`data/packer/base.py`) is the root contract plus the
`model_type → packer class` registry. `DataPacker` adds the tokenizer:
`setup(name_or_path, trust_remote_code=False)` builds `self.tokenizer` (`AutoTokenizer`) once
and falls back `pad_token_id → eos_token_id` when unset. **`trust_remote_code` defaults to
`False`** and is an explicit opt-in: set `model_hf.trust_remote_code=true` for a model whose
tokenizer, processor, or config needs custom code. Online entrypoints propagate that single
policy-owned decision to every actor-local packer and to the rollout engine.

| method                                                                                   | side            | today (`DecoderOnlyLLMDataPacker`)                                                                                                                                                                                                                                          |
| ---------------------------------------------------------------------------------------- | --------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `get_rollout_input(item)`                                                                | rollout         | abstract — the generation prompt. A raw `representations["prompt"]` **str** passes through verbatim; a conversation (`list[dict]`) is rendered with the tokenizer's chat template (`add_generation_prompt=True`); anything else raises                                      |
| `get_pretokenized_rollout_input(item, token_ids)`                                        | rollout/teacher | base returns exact `prompt_token_ids`; multimodal packers attach original media without re-rendering; native and HF Qwen3-VL packers share prompt-only image-run collapse (vLLM re-expands placeholders on token prompts) and keep the completion tail verbatim             |
| `rollout_collate_fn(items)`                                                              | rollout         | base default: passthrough                                                                                                                                                                                                                                                   |
| `get_rollout_output(completions, ...)`                                                   | rollout         | base default: passthrough (optional decode/logprob-attach hook)                                                                                                                                                                                                             |
| `get_policy_input(sample, rollout_output, n_ignore_prefix_tokens=0, *, prompt_ids=None)` | objective       | abstract — exact rollout `prompt_ids` are authoritative when present; the packer still restores model-specific media/tensors and accepts the completion as text or token IDs                                                                                                |
| `policy_compute_max_len(processed)`                                                      | objective       | abstract — max `len(input_ids)`                                                                                                                                                                                                                                             |
| `policy_collate_fn(processed, max_len)`                                                  | objective       | abstract — `{"input_ids": LongTensor[B,L], "logprob_masks": BoolTensor[B,L]}`, padded with `pad_token_id`, **on CPU**; plus `forward_replay_routing` (integer `[B,L,blocks,top_k]`) when the samples carry the rollout's recorded expert routing, via `add_forward_routing` |
| `sft_process_sample` / `sft_compute_max_len` / `sft_collate_fn`                          | SFT             | Existing RL packers raise; use `data.packer.sft.build_sft_packer` for Qwen text/image/video conversation SFT                                                                                                                                                                |
| `batch_size` / `slice_batch`                                                             | misc            | base defaults (`len`, python slice)                                                                                                                                                                                                                                         |
| `save_state(dest_path)`                                                                  | misc            | `tokenizer.save_pretrained(dest_path)`                                                                                                                                                                                                                                      |

The RL triad (`get_policy_input` → `policy_compute_max_len` → `policy_collate_fn`) is
what the GRPO stack uses today. The separate `QwenSFTDataPacker` implements the SFT triad
with unshifted `labels` and CPU media tensors; see the [SFT guide](../guides/sft.md).
Calling SFT methods on an RL-only packer still fails loudly.

`RLPolicyInput.logprob_masks` is `[0]*(len(prompt)-1) + [1]*len(completion) + [0]`
(with the default `n_ignore_prefix_tokens=0`; length == `len(input_ids)`), so `mask[:-1]`
indexes the length-`L-1` next-token logprob vector and the objective slices exactly the
completion positions. The full alignment argument — including what
`n_ignore_prefix_tokens` shifts — is the `get_policy_input` docstring; that is the
authoritative statement, not this paragraph.

`RLPolicyInput.forward_routing` (and `Qwen3VLPolicyInput.forward_routing`) is the rollout's
recorded expert routing for the sequence (`policy_info["rollout_routed_experts"]`,
`[T-1, layers, top_k]`, one row per input position), attached by `shape_policy_input` once the row
count has been checked against `input_ids`, and collated by `add_forward_routing`
(`data/packer/forward_routing.py`) into the model batch under `FORWARD_REPLAY_BATCH_KEY`; both keys
and the record layouts are declared once in `schema/routing.py`. A custom policy input
carries it by satisfying the `RoutedPolicyInput` protocol (a dataclass with `input_ids` and
`forward_routing`), and a custom packer's `policy_collate_fn` calls
`add_forward_routing(batch, processed, max_len)` after building its tensors to support
`rl.routing_replay=rollout`; without that call a replay run fails loud at the trainer, never
silently natural.

## Using it

```python
from cosmos_framework.posttrain.data.packer import init_data_packer

packer = init_data_packer("Qwen/Qwen2.5-0.5B-Instruct")   # setup-complete: tokenizer built
```

Low-level selection order in
`init_data_packer(name_or_path, packer=None, trust_remote_code=False, model_backend=None)`:

1. an explicit `packer=` instance wins (it is still `setup(name_or_path)`-ed);
2. otherwise `AutoConfig.from_pretrained(...)` resolves the model family and modality;
3. `model_backend="hf"` selects `HFVLMDataPacker` for any config with `vision_config`, or
   `DecoderOnlyLLMDataPacker` for a recognized causal LM;
4. `model_backend="i4"` (and the compatibility default `None`) first selects a registered native
   packer, then permits the decoder-only packer for causal-only model families.

There is deliberately no broad text fallback: types also registered as masked or seq2seq models
are excluded. An unregistered VLM on the native path raises
and directs the caller to `model_backend="hf"` or an explicit packer; an architecture that is
neither a declared VLM nor a recognized causal LM also raises. This lets unsupported native VLMs
use the generic HF model/processor path for image-conditioned clipped GRPO without allowing an
accidental backend choice to discard media tensors.

Production GRPO/PPO/OPD entrypoints resolve an immutable `DataPackerBuildSpec`. GRPO/OPD actors
build fresh packers from the same model/backend/factory/trust identity; PPO builds the rollout-side
packer once per rollout actor and carries its exact policy/reference/critic inputs on trajectories:

1. `data_packer.factory` for the student model;
2. `data_packer.teacher_model_factory` for a distinct cross-token multimodal OPD teacher model;
3. the low-level built-in `model_type` registry above when the applicable factory is empty.

Factories are zero-argument dotted callables returning fresh `DataPacker` instances; setup remains
centralized in `init_data_packer`. Rollout and policy actors construct separate objects from the same
student factory because one DataPacker class owns both API surfaces and their shared tokenizer and
processor semantics. See [`custom-data.md`](custom-data.md) for complete examples and Hydra dotlists.

## Declared identity

Two `runtime_checkable` protocols in `data/packer/base.py` make a packer's semantics part of the
exact-resume identity; the framework checks them with `isinstance`, never by probing attribute
names:

- `PackerSemanticsIdentity.validation_semantics_identity()` is required by held-out validation
  (`rl.validation`) from any configured packer instance; the framework additionally binds the
  packer class source. `Qwen3VLDataPacker` and `HFVLMDataPacker` implement it.
- `PackerProcessorIdentity.processor_semantics_identity()` is the immutable processor contract of
  a multimodal packer, consumed by the OPD cross-token teacher identity. `Qwen3VLDataPacker`
  implements it.

`Qwen3VLDataPacker.processor_semantics_identity()` binds the processor and tokenizer types, the
resolved chat template, the image processor config, the **video processor type and config** (the
Qwen processor exposes frame sampling and resizing as a component separate from the image
processor), the model type, and the image/vision placeholder ids. Both `rl.validation` and the OPD
cross-token teacher identity consume it, so `rl.validation` and OPD identities persisted on Qwen3-VL
before the video processor was bound refuse to resume. This is intended: two live processors that
differ only in video shaping must not share an identity. Start a fresh run instead of resuming.

Registered native packers today (`data/packer/__init__.py`):

| `model_type`                                         | packer                     | notes                                                                       |
| ---------------------------------------------------- | -------------------------- | --------------------------------------------------------------------------- |
| `qwen2`, `qwen3`, `gpt2`, `llama`                    | `DecoderOnlyLLMDataPacker` | causal-LM text shaping; `llama` may trigger the BOS warning described below |
| `qwen3_vl`, `qwen3_vl_moe`, `qwen3_5`, `qwen3_5_moe` | `Qwen3VLDataPacker`        | native image/video shaping                                                  |

### Qwen3-VL video rows

`Qwen3VLDataPacker` shapes video rows through the HF processor's text path, and that path disagrees
with the vLLM engine on two points that matter for training. The engine tokenizes the rendered
template with the video placeholder as three special tokens and then splices, per temporal group,
`encode("<t seconds>")` + `<|vision_start|>` + `<|video_pad|>` x tokens_per_frame + `<|vision_end|>`;
the HF processor writes the timestamps into the text and tokenizes once, so the last text character
BPE-merges with the `<` of the first timestamp (`".<"`), the stored rollout ids are one token longer
than the processor's, and `get_policy_input` refuses the row. The engine also needs one content UUID
per video at TP > 1, which a frame-list handle does not carry.

`Qwen3VLVideoDataPacker` (`data/packer/qwen3_vl_video.py`) is the same packer for text and image
rows and, on a video row, builds the policy ids the engine's way while keeping the HF processor's
media tensors; materialises the frame list into one `uint8 [T,H,W,3]` array named by a content UUID
over the ordered frame digests and the fps / frames_indices metadata; cross-checks the splice against
the HF ids in either transformers text form (4.x replaces the whole placeholder triple, 5.x replaces
only `<|video_pad|>` and keeps the outer marker pair); and refuses identically on the request side and
the policy side an odd frame count, a literal placeholder or stray vision marker in the text, a
missing `fps`, a non-list video handle, unequal frame sizes and a pretokenized video request. It is
not the registry default; select it per run:

```bash
data_packer.factory=cosmos_framework.posttrain.data.packer.qwen3_vl_video.make_packer
```

Video blocks must be `{"type": "video", "video": [<frame handles>], "fps": <float>,
"total_num_frames": <int>, "frames_indices": [<int>, ...]}` (`"type": "video_frames"` is accepted
as an alias and carried as `video` on the packer's copy of the messages) with a frame count that is
a multiple of the video processor's `temporal_patch_size`; the same block feeds the rollout request
and the policy input, and the policy side hands the HF processor the rollout's own decoded,
EXIF-transposed RGB frames rather than the row's original handles.

### Generic HF VLM backend

Use the generic HF path when a VLM has no native model adapter and packer:

```python
cfg.model_backend = "hf"
cfg.model_hf.name_or_path = "org/model-with-vision-config"
cfg.rollout.name_or_path = cfg.model_hf.name_or_path
```

GRPO and OPD propagate the backend identity and policy-owned `trust_remote_code` decision to
rollout, reward, trainer, and OPD student-packer construction. `HFVLMDataPacker` owns the Hugging
Face processor contract and emits both the vLLM `multi_modal_data` payload and ordered, content-addressed
`multi_modal_uuids`; `HFModel` exposes `forward_policy_batch` so the clipped GRPO objective forwards
the collated media tensors instead of reducing the batch to token IDs. The legacy policy-gradient
objective rejects a shaped input with non-empty `mm_kwargs` rather than relying on a packer class
flag. The implementation is imported
lazily, so selecting a native model does not import or instantiate the generic HF VLM packer.

Conversation images are passed directly to `AutoProcessor`; processor outputs such as
`image_sizes` are preserved through collation and model forwarding. The verified generic contract
covers image inputs with clipped GRPO and same-token OPD. Unsupported SFT, DPO, video, and base64
helpers are intentionally absent rather than carried as untested code.

A batch must also use a consistent processor schema. In particular, do not mix samples where only
some processor outputs contain `mm_token_type_ids`: the current collator requires that optional
field on every sample or none, because its architecture-specific padding semantics are not defined
for a mixed batch.

Each process that needs shaping builds its own packer through the factory resolver. The packer is
plain CPU state — it is not shipped across the wire. The policy packer is constructed eagerly during
online trainer assembly because every objective enters through that seam. Reward and teacher-model
packers are constructed only when their enabled feature consumes one.

## Adding a packer for a new architecture

Drop a sibling module in `data/packer/` that subclasses `DataPacker`:

```python
# cosmos_framework/posttrain/data/packer/my_vlm.py
from cosmos_framework.posttrain.data.packer.base import DataPacker

class MyVLMDataPacker(DataPacker):
    def setup(self, name_or_path: str) -> None:
        super().setup(name_or_path)          # builds self.tokenizer (+ pad fallback)
        self.processor = ...                 # any extra per-model shaping state

    def get_rollout_input(self, item): ...
    def get_policy_input(self, sample, rollout_output, n_ignore_prefix_tokens=0, *, prompt_ids=None): ...
    def policy_compute_max_len(self, processed_samples): ...
    def policy_collate_fn(self, processed_samples, computed_max_len): ...
```

then register it for its HF `model_type`(s). The registration must actually be *executed*
before `init_data_packer` runs, so put the call either in `data/packer/__init__.py` next
to the shipped one (always imported — the safe default), or at the bottom of your module
if you can guarantee it is imported first:

```python
# in a sibling module: import the base directly (no package-__init__ cycle)
from cosmos_framework.posttrain.data.packer.base import BaseDataPacker
BaseDataPacker.register(["my_vlm"], MyVLMDataPacker)   # allow_override=True to replace one

# from outside the package, the same function is re-exported:
from cosmos_framework.posttrain.data.packer import register
```

Rules that bite:

- `setup()` must leave `self.tokenizer` usable — either call `super().setup(...)` or set
  it yourself; `pad_token_id` and `assert_no_special_tokens()` both read it. An empty
  `name_or_path` raises.
- Registering a *different* class for an already-claimed `model_type` raises unless you
  pass `allow_override=True`.
- Four methods are `@abstractmethod` (`get_rollout_input` + the RL triad): a subclass that
  omits one cannot be instantiated.
- Keep it CPU-side and weight-free. If your shaping needs model weights, it belongs in
  the model adapter, not here.

## Where it plugs in

| consumer                                         | how the packer arrives                                                                                                                           | what it uses                                                                                                                                                   |
| ------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `roles/rollout/hf.py::HFRollout`                 | `HFRollout(model, packer=..., sampling=...)` — **keyword-only, required**                                                                        | `get_rollout_input` for the prompt; `packer.tokenizer` (as `self.tok`) for the `add_special_tokens=False` encode and the completion decode                     |
| `roles/rollout/vllm.py::VLLMRollout`             | `VLLMRollout(llm, packer=..., sampling=...)` — **keyword-only, required**; `config/build.py::build_vllm_rollout(cfg, packer)` is the config path | `get_rollout_input` per prompt; warns when `adds_special_tokens()` is true                                                                                     |
| `algorithm/grpo.py::make_objective(packer=None)` | **optional** — the *fallback* shaper                                                                                                             | only for a trajectory carrying no `policy_info["prompt_ids"]`: `get_policy_input(tr.obs, completion)`, then `logprob_masks[:-1]` to select completion logprobs |

Two deliberate ergonomics in the rollout constructors:

- **The packer is required at *construction*, not at `generate`.** It is decidable at
  construction, and a packer-less rollout would otherwise bind, spawn every Ray actor,
  warm the vLLM engine — and only die on the first `generate()` with the whole cluster
  up. Weight-only rigs that never generate construct one explicitly.
- **Everything after `model` is keyword-only.** The 2nd positional slot changed meaning
  twice on `HFRollout` (`tokenizer` → `sampling` → `packer`), so a stale positional call
  would silently bind the wrong object and surface as an unrelated error later.
  Keyword-only turns every stale call into an immediate, accurate `TypeError`.
  `HFRollout` takes no `tokenizer` argument — the packer owns the tokenizer, reachable as
  `packer.tokenizer` (the rollout exposes it as `self.tok`).

### Objective shaping: stored ids are authoritative; the packer still owns side inputs

The objective passes `policy_info["prompt_ids"]` — the ids the rollout actually generated from —
to `packer.get_policy_input(..., prompt_ids=...)`. The packer must preserve them exactly. This is
correct-by-construction for the causal prefix while still letting a VLM packer recover
`pixel_values`, grids, attention masks, and other model inputs from the original `ObsBundle`.
Only when no packer is supplied does the decoder-only fallback concatenate stored prompt IDs and
completion IDs directly.

Consequences, which are easy to get backwards:

- **A new rollout body should record `prompt_ids`** in `policy_info`. Both shipped rollouts
  (`HFRollout`, `VLLMRollout`) do, so the stored path is the live one in production.
- **A packer remains optional only for decoder-only trajectories with stored IDs.** Native VLM OPD
  always supplies its registered packer; otherwise media conditioning would be silently discarded,
  and the objective now rejects model-specific batch fields without a `VLMPolicy` capability.
- **A trajectory with neither ids nor a packer raises** — `make_objective()` (no packer) fed a
  trajectory without `prompt_ids` fails loudly rather than guessing a prefix. Fix it by
  recording `prompt_ids`, or by passing `make_objective(packer=init_data_packer(<model>))`.
- A **string** completion always needs a packer (it cannot be concatenated with stored
  ids without tokenizing), so a string-emitting producer still needs a packer.

This is the §10 amendment in the
internal design spec, which supersedes the
packer-always-shapes rule stated in that spec's §4/§9.

### The special-token consistency guard

`DataPacker.adds_special_tokens()` compares `tokenizer(probe, add_special_tokens=True)` against
`add_special_tokens=False` and returns whether they differ; `assert_no_special_tokens()` is the
raising wrapper. **`VLLMRollout.__init__` warns — it does not refuse.**

Why it exists: vLLM is handed the prompt **string** and tokenizes it internally (with its own
defaults), whereas `get_policy_input` re-tokenizes with `add_special_tokens=False`. Those two
agree only for a tokenizer that adds nothing. With a BOS-adding tokenizer (llama) the rollout
would generate from one prefix while the objective scored under another — a silently desynced
training step, so the guard refused at construction. Qwen and GPT-2 pass.

That premise **no longer holds**, which is why the check is now advisory. `VLLMRollout` records
`req.prompt_token_ids` — vLLM's *own* internal tokenization — and the objective prefers those ids,
so the scored prefix IS the generated one whatever the tokenizer adds. Enforcing it would only
block `llama`, which the registry explicitly supports. It is not enforced on the fallback branch
either: a producer that records no `prompt_ids` has no engine tokenization to disagree with, so
the packer's re-derivation is by definition the prefix, and raising there would wrongly reject
synthetic/test producers. `HFRollout` needs no guard either way — it does the encode itself with
`add_special_tokens=False` and stores those same ids.

The warning remains because the residual risk is real but narrow: a *future* consumer that
re-tokenizes `obs` instead of using the stored ids would score a shifted prefix. Such a path
should call `assert_no_special_tokens()` explicitly.

Remedies if you hit it: use a no-BOS tokenizer (Qwen), or feed vLLM pre-tokenized prompts
(specced, not built — see spec §9.6).

## Troubleshooting

| symptom                                                                                                 | cause                                                                                                      | fix                                                                                           |
| ------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| `model_type='x' declares vision_config but has no native packer`                                        | an unregistered VLM was selected with the native backend                                                   | set `model_backend="hf"`, register a native packer, or pass an explicit packer                |
| `cannot classify HF model_type='x'`                                                                     | the HF config declares neither `vision_config` nor a recognized causal-LM architecture                     | use the correct model config or pass an explicit packer                                       |
| `Unable to import DataPacker factory '…'`                                                               | configured dotted path is unavailable on the driver or worker `PYTHONPATH`                                 | install/mount the extension module identically on every node and correct the factory path     |
| `DataPacker factory '…' must return a DataPacker`                                                       | a factory returned the wrong object type                                                                   | return a fresh `DataPacker` subclass instance from the zero-argument factory                  |
| `TypeError: …__init__() missing 1 required keyword-only argument: 'packer'`                             | rollout constructed without a packer at all                                                                | `HFRollout(model, packer=init_data_packer(name_or_path))` / `build_vllm_rollout(cfg, packer)` |
| `ValueError: HFRollout requires a DataPacker at construction …` / `VLLMRollout requires a DataPacker …` | `packer=None` passed explicitly (a bare `TypeError` would not say what to do)                              | build one with `init_data_packer(name_or_path)` and pass it                                   |
| `TypeError: …__init__() takes 2 positional arguments but 3 were given`                                  | a stale positional call — e.g. `HFRollout(m, tok)` / `VLLMRollout(llm, {...})`; only `model` is positional | pass `packer=` / `sampling=` by keyword                                                       |
| `ValueError: packer tokenizer adds special tokens …`                                                    | a BOS-adding tokenizer on the vLLM path                                                                    | see the guard section above                                                                   |
| `ValueError: DataPacker.setup: empty name_or_path`                                                      | packer built without a model id/path                                                                       | pass the same `name_or_path` the model/engine was built from                                  |
| `ValueError: policy_collate_fn got 0 samples`                                                           | empty batch / zero batch size upstream                                                                     | fix the batch source; the packer refuses rather than emitting a degenerate tensor             |
| `ValueError: get_policy_input: empty prompt`                                                            | `representations["prompt"]` tokenized to no tokens                                                         | check the env/source producing the prompt                                                     |

Behaviour above is pinned by `tests/test_data_packer_{base,init,decoder_only}.py`, and the
objective path is pinned bit-for-bit against the former inline math by
`tests/test_algorithm_grpo.py::test_packer_objective_matches_inline_bit_for_bit`.

### Two rollout-side hooks exist but have NO call site yet

`rollout_collate_fn` and `get_rollout_output` are declared on `BaseDataPacker` with passthrough
defaults, and **nothing in posttrain calls either one**. Both rollout bodies instead loop over
prompts calling `get_rollout_input` per item, and shape the engine's output inline —
`hf.py` slices `seqs[ki][plen:]` then `tok.decode(...)`; `vllm.py` takes `out.token_ids` and
`_token_logprobs(out)`.

For the decoder-only text path this is behaviourally identical (cosmos-rl's defaults are
passthrough too), so it costs nothing today. It matters for the FIRST non-text packer: in
cosmos-rl a new architecture supplies output handling by overriding `get_rollout_output`
(`rollout/worker/rollout_control.py` routes every payload through it), whereas here the same
change means editing `hf.py`/`vllm.py`. That is the one place our port narrows cosmos-rl's
extensibility, and it is worth closing when the VLM packer lands rather than before — the hook
shape should be driven by a real second consumer, not guessed.

Note this is a *call-site* gap, not the signature difference: our `get_rollout_output` deliberately
returns just `completions` instead of cosmos-rl's 5-tuple, and drops its Redis/UUID offload, because
`comm/codec.py`'s automatic metadata/bulk split already externalizes heavy fields — see §0.3 of the
internal design spec for that rationale.

## Which seam do I use

A `DataPacker` subclass (+ `register`) when one sample is one token sequence — that is what its four
abstract methods assume. When the model's own collator owns batch assembly, or the unit is not a token
sequence, write a **batch-assembly function** over `Trajectory` lists and hand it to the objective
(`data/packer/preference.py` for DPO, `model/gr00t/packer.py` for GR00T-N1.7 flow-BC). Same invariants:
weight-free, CPU-side, one owner per model × objective. Architecture statement: `DESIGN.md` v3,
"Batch-assembly seam"; spec §12.

## Deferred

The separate Qwen SFT packer does not unify the RL and SFT shaping implementations.
The DPO/diffusion media path remains shaped in `data/datasets/` plus the model adapter;
non-Qwen SFT templates require their own tested masking contract.
