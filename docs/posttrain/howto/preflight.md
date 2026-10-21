# Manifest validation (`bind_check`)

**Come here before you allocate GPUs.** `--validate-manifest` resolves a recipe's roles, edges,
fields and transport and exits, so a misconfigured run fails in seconds instead of after the engine
and trainer have loaded.

This is one of three unrelated things posttrain calls "validation" — see
[`../glossary.md`](../glossary.md). Every capability guide's quick start begins with this check.

Manifest validation is a GPU-less preflight for a recipe's declared roles, edges, fields, and
transports. It is separate from held-out [model validation](train-time-validation.md).

Run it explicitly with:

```bash
"$POSTTRAIN_PY" -m cosmos_framework.posttrain.entrypoints.grpo \
  --config cosmos_framework/posttrain/configs/grpo_qwen3b_gsm8k_public_4gpu.py \
  --validate-manifest
```

This is the public GRPO recipe from the [run guide](run-posttrain.md#3-grpo); a successful check
prints `bind_check: ok`. Overrides for your own data or reward go after a `--`, as in a training
command, and are validated the same way.

`--validate` remains a deprecated alias. Normal `run()` performs the same check before Ray
placement and model construction. The explicit command prints the manifest and report, then exits
zero on success or one on failure.

`entrypoints.ppo`, `entrypoints.opd`, `entrypoints.nft` and `entrypoints.joint_ppo` accept the flag
too; the [run guide](run-posttrain.md) shows it for the public PPO, NFT and Joint PPO recipes. OPD
also validates its teacher signal and placement edges through the OPD manifest.

## What is checked

`bind_check` validates the `Manifest` built from the entrypoint's own role declarations and recipe
edges. It reports all errors together:

| category    | rejects                                                                      |
| ----------- | ---------------------------------------------------------------------------- |
| `wiring`    | missing roles, unknown implementations, or unsupported planes                |
| `field`     | consumer fields missing from the selected producer, or field-kind mismatches |
| `transport` | payloads unsupported by the selected mover                                   |

Undecidable runtime facts are reported as non-failing obligations. The transport check uses the
same capability predicate as runtime mover selection.

GRPO and OPD add an objective-specific temperature check to their GPU-less entrypoint preflight.
It never rewrites config:

- on-policy GRPO requires rollout and policy-loss temperatures to agree, with top-p one, top-k
  disabled, and neutral repetition penalty. The clipped objective's opt-in rollout-logprob
  denominator requires both temperatures to be one because vLLM returns raw rollout logprobs;
- top-k direct OPD requires teacher and policy-scoring temperatures to remain one, while nonzero
  rollout sampling controls remain data-generation choices;
- top-k advantage OPD additionally requires the on-policy rollout controls above; and
- DPCA retains its stronger all-neutral likelihood contract.

All grouped GRPO/OPD rollouts reject temperature zero because vLLM greedy generation requires one
sample, and reject nonzero values below `0.01` because the pinned backend rewrites them to `0.01`.
Validation-generation sampling, DPO `beta`, and diffusion schedules are separate concepts.

DPO has a config-independent topology, so its explicit preflight stays torch-free. GRPO's topology
depends on `rl.data` and `rl.reward`; validation therefore loads the config and checks that an
answer-requiring reward is paired with a source carrying an answer. A configured JSONL is also
opened in the current execution environment to catch an unreadable path before placement.
Online NFT also loads its selected config. It checks local/single placement, a configured
source and initialization checkpoint, native UniPC/base-shift settings, exact group/FSDP
equality, moving-old/frozen-reference roles, an immutable deployed reward-provider revision,
and presence of the reward-service environment variables. It then instantiates the
configured model-free dataloader and prepares its first selected-modality row. The
T2V path catches empty selection, incompatible caption schema, conditioned-frame,
and multi-row packing; the T2I path checks its prompt artifact and native image
schema. It does not scan every row. The smoke needs
the same data credentials and mounts as training; it neither downloads the checkpoint
nor calls the reward service.
The final online-NFT summary prints `PASS` only when manifest binding, runtime config,
and the source smoke all succeed; every other outcome prints `FAIL` and exits one.

The native T2V recipe decodes 93 frames from 24 latent frames. Its cluster acceptance
must cross the moving-old boundary: updates 1–10, checkpoint, exact resume, then update
11. That proves structural source continuation, frozen-reference preservation, moving-old
version 10, provider provenance, and post-resume weight publication. The model-free
pipeline and Ray smoke do not prove CUDA/FSDP numerics, live service behavior, or model
capacity. Fixed-tolerance Cosmos-RL parity remains a separate, inconclusive study rather
than a claim of this runnable vertical.

The qualified T2I profile applies a distinct, explicit preflight rather than

The public T2I contract and its artifact checks are documented in
the [public NFT run guide](run-posttrain.md#7-nft).

The manifest document written to the run log is schema-versioned. Load its `manifest` field with
`manifest_from_document()` before passing it to `bind_check`; the loader rejects unknown fields,
versions, planes, duplicate identities, and malformed roles or edges.

## Adding a capability

Implementations own their descriptors and register them by `impl_key`:

```python
register_capability(
    "my_impl",
    CapabilityDescriptor(
        produces=(FieldSpec("targets", SUPERVISION_KINDS[Preference]),),
        planes=frozenset({Plane.DATA}),
    ),
)
```

The registry is process-global and has no unregister operation. A descriptor may include a remedy
for field errors because the checker sees the selected role, not the config option that selected it.

## Limits

Passing the preflight proves declared structural compatibility and, when checked in the same
environment, JSONL readability. It does not prove that every row is valid, that a reward is
calibrated, or that a sampled group has useful reward variance. Runtime data and reward boundaries
still reject missing, null, or non-finite required values.

`nonempty` is only a connectivity reward: an all-nonempty group has equal rewards and zero
group-relative advantage. DPO's fixed manifest mainly guards future topology edits; placement-aware
checks remain runtime obligations until the recipe exposes a variable mesh.
