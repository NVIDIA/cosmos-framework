# Custom datasets and DataPackers

**Come here when the built-in prompt or preference sources do not describe your data.** This is
the seam for pointing a run at your own dataset without changing framework code.

Applies to [`../guides/grpo.md`](../guides/grpo.md), [`../guides/dpo.md`](../guides/dpo.md),
[`../guides/ppo.md`](../guides/ppo.md), and [`../guides/opd.md`](../guides/opd.md). If you also need
different tensor shaping for a new model architecture, that is a different seam:
[`custom-datapacker.md`](custom-datapacker.md).

Posttrain can construct run-local dataset and DataPacker extensions from dotted callable paths in
the resolved config. The driver validates imports before placement; each owning worker constructs
its own object. This keeps local, torchrun, and Ray runs replayable without relying on process-local
registration side effects.

## Create an importable extension module

Put the following in a module available on every node's `PYTHONPATH`, for example
`my_project/posttrain_extensions.py`:

```python
from typing import Any

from cosmos_framework.posttrain.data.packer.base import DataPacker
from cosmos_framework.posttrain.data.packer.decoder_only_llm import (
    DecoderOnlyLLMDataPacker,
    RLPolicyInput,
)
from cosmos_framework.posttrain.data.packer.qwen3_vl import Qwen3VLDataPacker
from cosmos_framework.posttrain.schema import ObsBundle


class PromptDataset:
    """Map-style raw rows with a stable exact-resume identity."""

    def __init__(self, count: int = 100) -> None:
        self.count = count

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> dict[str, str]:
        return {"question": f"What is {index} + 1?", "answer": str(index + 1)}

    def cursor_fingerprint(self) -> str:
        return f"example-prompts-v1-count={self.count}"


def build_prompt_dataset(count: int = 100) -> PromptDataset:
    return PromptDataset(count)


class PreferenceDataset:
    """Canonical DPO items with metadata-only leakage identities."""

    def __init__(self, count: int = 100) -> None:
        self._items: list[tuple[str, dict[str, Any], dict[str, Any]]] = [
            (
                f"Describe sample {index}",
                {"sample": f"preferred-{index}"},
                {"sample": f"rejected-{index}"},
            )
            for index in range(count)
        ]

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> tuple[str, dict[str, Any], dict[str, Any]]:
        return self._items[index]

    def membership_ids(self) -> tuple[str, ...]:
        """Return stable aliases without reading or decoding media payloads."""
        return tuple(f"preference-pair-{index}" for index in range(len(self._items)))


def build_preference_dataset(count: int = 100) -> PreferenceDataset:
    return PreferenceDataset(count)


class CustomDataPacker(DecoderOnlyLLMDataPacker):
    """One model-owned contract with rollout-side and policy-side APIs."""

    def get_rollout_input(self, item: ObsBundle) -> str:
        prompt = super().get_rollout_input(item)
        return f"<domain>{prompt}"

    def get_policy_input(
        self,
        sample: ObsBundle,
        rollout_output: Any,
        n_ignore_prefix_tokens: int = 0,
        *,
        prompt_ids: list[int] | None = None,
    ) -> RLPolicyInput:
        return super().get_policy_input(
            sample,
            rollout_output,
            n_ignore_prefix_tokens,
            prompt_ids=prompt_ids,
        )


def build_default_packer() -> DataPacker:
    return CustomDataPacker()


def build_teacher_model_packer() -> DataPacker:
    """Construct the distinct Qwen3-VL teacher model's packing contract."""
    return Qwen3VLDataPacker()
```

A dataset factory is called as `factory(**factory_kwargs)` and must return a non-empty map-style
object with `__len__` and `__getitem__`. Iterable-only sources are rejected because stable indexing
is required for rank sharding, validation, and exact cursor resume. A custom DPO source used with
train-time validation must also expose `membership_ids() -> tuple[str, ...]`; the hook returns stable
semantic aliases from metadata only and must not read or decode media. A DataPacker factory is
zero-argument and must return a fresh `DataPacker`; posttrain calls `setup()` with the model path
after construction. A DataPacker class owns both rollout-side and policy-side APIs for that model
identity; actor processes construct separate instances from the same factory.

PPO checkpoints the data cursor at every configured boundary, so a custom PPO prompt source must
also expose a stable `cursor_fingerprint()` as in the example. The fingerprint must change whenever
the indexed content or indexing semantics change.

## Select a custom GRPO, PPO, or OPD prompt source

The custom source returns raw dictionaries. `prompt_column`, optional `response_column`, and optional
`items_fn` retain the same meaning as for a Hugging Face dataset. The framework renames the selected
columns to canonical `prompt` and `reference` representations.

```bash
PYTHONPATH=.:<extension-root> python -m cosmos_framework.posttrain.entrypoints.grpo -- \
  <experiment-and-model-overrides> \
  rl.data.factory=my_project.posttrain_extensions.build_prompt_dataset \
  +rl.data.factory_kwargs.count=1000 \
  rl.data.prompt_column=question \
  rl.data.response_column=answer
```

The same source-selection fields work with `entrypoints.opd` and `entrypoints.ppo`. PPO additionally
accepts `image_column`, `video_column`, `id_column`, and an explicit `metadata_columns` allowlist;
only those configured fields cross its data-role boundary. Set at most one of `rl.data.factory`,
`rl.data.path`, and `rl.data.name`. An unset source preserves each entry's built-in arithmetic source.

## Select a custom DPO preference source

DPO factories return `(prompt, winner, loser)` items. `winner` and `loser` must use the same payload
shape accepted by the existing preference objective. `DatasetPreferenceEnv` still owns shuffle,
rank sharding, cursor state, and validation traversal. When validation is enabled, both training and
held-out custom datasets must implement the metadata-only `membership_ids()` hook shown above so
the framework can reject overlap without enumerating and decoding every item.

```bash
PYTHONPATH=.:<extension-root> python -m cosmos_framework.posttrain.entrypoints.dpo -- \
  <experiment-and-model-overrides> \
  rl.data.factory=my_project.posttrain_extensions.build_preference_dataset \
  +rl.data.factory_kwargs.count=1000 \
  rl.data.shuffle=true
```

Do not combine `factory` with `jsonl_path`, `tar_path`, or `synthetic_pairs`.

### Cursor fingerprint migration

Custom prompt and preference cursor fingerprints include the dataset length. This
intentionally invalidates checkpoints written with the earlier factory-only fingerprint: accepting
the old value would silently bypass the new changed-row-count guard. A pre-change checkpoint resumed
with `rl.data.cursor_resume=strict` therefore fails before policy construction. If model-only,
non-exact source continuation is acceptable, set `rl.data.cursor_resume=off` for the migration run,
then return to `auto` or `strict` after starting a new checkpoint lineage. Otherwise finish the
strict-resume lineage with the older code before upgrading.

## Select custom DataPackers

`data_packer.factory` supplies the student model's DataPacker implementation:

```bash
PYTHONPATH=.:<extension-root> python -m cosmos_framework.posttrain.entrypoints.grpo -- \
  <experiment-and-model-overrides> \
  data_packer.factory=my_project.posttrain_extensions.build_default_packer
```

On GRPO/OPD, rollout, policy, tokenizer-dependent reward filtering, and the OPD teacher actor's
student codec construct fresh instances from this same factory. They must not select independent
factories: generation and policy scoring need one tokenizer, processor, chat-template, placeholder,
and mask contract. The DataPacker's rollout and policy methods encapsulate the engine-specific
operations. The GRPO/OPD policy actor constructs its packer eagerly during trainer assembly; reward
and teacher-model packers remain conditional on the features that consume them.
For PPO, the rollout-side packer produces the exact policy/reference/critic input mappings carried
through the shared trajectory; those model roles do not independently re-tokenize the sample.

Cross-token multimodal OPD may use a genuinely different teacher model. Set its separate factory
only when that teacher model is not covered by the built-in `model_type` registry:

```bash
PYTHONPATH=.:<extension-root> python -m cosmos_framework.posttrain.entrypoints.opd -- \
  <experiment-and-model-overrides> \
  data_packer.factory=my_project.posttrain_extensions.build_default_packer \
  data_packer.teacher_model_factory=my_project.posttrain_extensions.build_teacher_model_packer
```

An empty `factory` or `teacher_model_factory` falls back to the built-in `model_type` registry for
that model. The reward packer is built only when shared-prefix filtering needs a tokenizer, and the
teacher-model packer is built only for OPD `cross_token_multimodal` mode. Hydra's `+` prefix is
required only when adding a new key to the initially empty dataset `factory_kwargs` mapping.

## Validation and failure behavior

- Use `--validate` or `--validate-manifest` as usual. GRPO, PPO, and OPD import configured dataset
  and DataPacker paths during that preflight. DPO keeps its validation command manifest-only and
  imports its dataset factory during normal driver startup, still before placement. Construction
  remains in the owning worker.
- Test any factory path directly without starting training:

  ```bash
  PYTHONPATH=.:<extension-root> python -c \
    "from cosmos_framework.utils.serialization import load_callable; load_callable('my_project.posttrain_extensions.build_dataset')"
  ```

- The extension module, its dependencies, and any local data paths must exist in the same location on
  every node. Include the extension root in `PYTHONPATH` for both the driver and workers.
- Import failures, constructor failures, wrong return types, empty datasets, and mutually exclusive
  source selectors fail with the configured dotted path in the error context.
- Enabling DPO validation with a custom source that lacks `membership_ids()` fails during source
  identity setup instead of reading or decoding dataset items to synthesize aliases.
- `config_resolved.yaml` records factory paths and kwargs. Keep factory behavior deterministic for
  identical kwargs; external mutable state is not captured by that artifact.

The normative design and alternatives are recorded in
the internal design record.
