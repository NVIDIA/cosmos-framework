# Prepare the Cosmos Framework environment

Set `FRAMEWORK_DIR` to the absolute Framework checkout root. The installation
commands define `SIM_TRANSFER_SKILL` as the path to this bundle. Run subsequent
commands in Framework's environment. Training and generation require an
allocated NVIDIA GPU runtime.

Use the Sim-Transfer code and lockfile from the Framework checkout containing
this skill. The [public repository](https://github.com/NVIDIA/cosmos-framework)
provides the source and installation files.

## Install the matched environment

Use Linux, Python **3.13**, uv **0.11.29** and a CUDA group supported by your
NVIDIA driver. The command below invokes that uv version through `uvx`.
Set `FRAMEWORK_DIR` to the destination for a new checkout:

```bash
export FRAMEWORK_DIR="${FRAMEWORK_DIR:?Set the absolute Framework checkout destination}"
git clone https://github.com/NVIDIA/cosmos-framework.git "$FRAMEWORK_DIR"
cd "$FRAMEWORK_DIR"
export SIM_TRANSFER_SKILL="$FRAMEWORK_DIR/.agents/skills/cosmos3-nano-sim-transfer"
uvx --from uv==0.11.29 uv sync --locked --python 3.13 --all-extras --group cu130-train
source .venv/bin/activate
export COSMOS_INTERNAL=0 COSMOS_TRAINING=1 COSMOS_SMOKE=0
export COSMOS_DEVICE=cuda WANDB_MODE=disabled
export OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
```

For an existing checkout, skip cloning, set `FRAMEWORK_DIR` to its root and
run the remaining commands there. For CUDA 12.8, use `cu128-train` consistently.
Both training groups include matched NPP packages. Complete the
[native-library and CUDA-toolkit setup](environment.md) on each host.
Use the [multi-node launcher](../post-training/references/multinode-training.md) for two four-GPU
hosts.

[Nano conversion](../post-training/references/nano-conversion.md)
uses this checkout and environment, checks the converter's source hash and
records the actual Framework revision. No separate converter checkout is needed.

## Enable Hugging Face downloads

Sign in with `hf auth login`, or supply your own `HF_TOKEN` for a non-interactive
job. Accept the license for each gated model you download. The default inference
commands also load [Cosmos-Guardrail1](https://huggingface.co/nvidia/Cosmos-Guardrail1)
for its blocklist and face-blur resources and
[Qwen3Guard-Gen-0.6B](https://huggingface.co/Qwen/Qwen3Guard-Gen-0.6B)
for text filtering. Allow Hugging Face access on the first run and retain the
cache for subsequent runs. Keep tokens out of command output, saved notebooks
and support logs.

## Continue with the example workflow

Download the [Hugging Face examples](hf-sample.md). Use **example2** for
captioning, [bundle preparation](local-dataset.md), training and
inference. The **example1** walkthrough covers long-rollout inference. Saved
captions need no caption-service credentials; follow [captioning](../post-training/references/caption-guide.md)
to generate new captions.

The example download contains media and captions. The
[bundle preparation guide](local-dataset.md) downloads the text
tokenizer from Hugging Face.
[Nano conversion](../post-training/references/nano-conversion.md)
downloads the model and video VAE for training. Continue with
[post-training](../post-training/SKILL.md) or
[inference](../inference/SKILL.md).

## References

- [Environment and native-library requirements](environment.md)
- [Two-node training launcher](../post-training/references/multinode-training.md)
- [Nano checkpoint conversion](../post-training/references/nano-conversion.md)
- [Hugging Face examples](hf-sample.md)
- [Local training bundle preparation](local-dataset.md)
