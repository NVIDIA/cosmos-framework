# Single-GPU Framework video training and evaluation

Use the same `scripts/cosmos_workflow.py` planner as multi-GPU training, with
`--platform docker --nodes 1 --gpus-per-node 1` and an explicitly selected host
GPU when needed. No backend flag or alternate training wrapper is required.
Confirm image, data, model, output paths, memory requirements, and launch with
the user. GPU count alone does not prove that a checkpoint fits.

The planner owns native TOML, model preparation, device mapping, preflight,
record binding, and rendering. Keep dense/PEFT, sampling, seed, and effective
batch choices explicit. Do not silently turn a full run into a smoke subset.

For evaluation, use `scripts/evaluation_workflow.py` with the sealed training
plan and structured status. Verify the Framework checkpoint export, resolve
the missing evaluation inputs, and submit the emitted bundle with one GPU.
See `cosmos-reason-evaluate.md` for checkpoint, metric, and terminal contracts.
