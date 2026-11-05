# Check Bimanual training data

Run this before loading model weights. Supply a completed local Bimanual bundle
containing `manifest.json`, `samples.json`, encoded assets and `resources/`.
The command reads the bundle; it does not create training data or change files.

```bash
python -m cosmos_framework.scripts.validate_bimanual_data --bundle /data/bimanual
```

The static check uses only the Python standard library. It checks sample counts,
unique identities, transform references, source-row mappings, and the presence of
local assets, normalization statistics and tokenizer files. Missing files and
paths outside the bundle fail, including symlinks that escape it.
Success reports `passed: true` and `materialized: false`; media has not been decoded.

With the Framework training dependencies installed, check one packed sample
through the [native stage loader](../cosmos_framework/data/generator/local_simulation/bimanual_config.py):

```bash
python -m cosmos_framework.scripts.validate_bimanual_data \
  --bundle /data/bimanual --materialize --stage 0
```

Choose the stage you intend to train (`0`, `1`, `2` or `3`). The native loader
selects the bundle's robot source and materializes its user-selected time
windows. Window length is not fixed by this loader; each sample must preserve
the native `T+1` observation / `T` action alignment. It runs
the existing decoding, stage-specific rendering, action transforms,
tokenization and packing.
The check uses seed 42, zero worker processes and a one-sample batch limit;
Stage 0 retains its 400,000-token ceiling. It does not alter the training recipe.

Success reports `materialized: true`, the batch's sample/token counts and tensor
shapes/dtypes. Non-finite tensors fail. Inspect additional representative
windows before training. No model checkpoint or optimizer is required.

The versioned single-source robot example and its checksum are published under
[`examples/training`](https://huggingface.co/nvidia/Cosmos3-Nano-Sim-Bimanual/tree/main/examples/training).
Use its layout as a reference when constructing a user-owned bundle. The
example is intentionally small and validates integration, not convergence or
dataset quality. Framework does not convert raw robot data into this bundle.

For released-model inference inputs, use `tools/check_bimanual_mixed_inputs.py`
instead; an inference request is not a training bundle.
