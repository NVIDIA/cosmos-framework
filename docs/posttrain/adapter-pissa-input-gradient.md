# Versioned PiSSA input gradients

Factorized PiSSA now has an opt-in `pissa_input_gradient="shared_fp32_v1"`
setting. It shares the FP32 input conversion used by the trained and initial
low-rank branches. Their input gradients therefore cancel in FP32 before one
conversion to the input dtype. The frozen base projection keeps its existing
precision boundary. The forward expression, scaling, trainable factors, frozen
initial factors and residual-base representation are unchanged.

The default remains `separate_casts_v1`. It preserves the original backward graph
and is omitted from resolved manifests so existing identities remain unchanged.
The new setting requires `method="pissa"`, `execution="factorized_v1"`, zero
dropout and `model_execution="default_v1"`. LoRA, DoRA, dense PiSSA, rounded-weight
execution and other whole-model execution profiles reject the opt-in setting.

For example, a Qwen2.5 text recipe can explicitly select:

```python
AdapterComponentConfig(
    component="language",
    targets=["model.layers.*.self_attn.q_proj", "model.layers.*.self_attn.v_proj"],
    method="pissa",
    initialization="svd_lowrank",
    execution="factorized_v1",
    pissa_input_gradient="shared_fp32_v1",
    rank=8,
    alpha=16,
    use_rslora=True,
    dropout=0,
)
```

The setting belongs to each component. Tied aliases must select the same policy,
as they must for all other adapter settings. It creates no new parameters and
does not change optimizer ownership. Activation replay reconstructs the same
shared conversion without persistent activation or gradient caches.

Portable artifacts using the opt-in setting have format
`cosmos-posttrain-adapter-v5`; their resolved settings record the backward policy.
Full checkpoints include that identity in the existing exact-resume contract.
Changing the policy on resume or warm start fails identity validation. Resume
does not repeat PiSSA decomposition. Older runtimes reject the new format rather
than silently applying legacy backward semantics. Merged weights use the existing
PiSSA conversion and preserve the source adapter identity in export provenance.

This is a candidate correction under qualification. The preserved A100 diagnostic
used the two original saved first-update states for Qwen2.5-0.5B rsPiSSA/FSDP2 and
Mistral-7B PiSSA/FSDP2. Sharing the cast preserved every observed forward output.
It removed Qwen2.5's cross-state gradient gate violations in that diagnostic;
Mistral still failed. The new gradients differ from the legacy distributed
trajectory, so this does **not** validate the original failed reference or an
uninterrupted new training trajectory. Installed two-host multi-update, resume,
warm-start and final artifact qualification must pass before campaign credit.
Legacy failures and numerical thresholds remain unchanged.
