# Environment Variables

- `IMAGINAIRE_OUTPUT_ROOT=outputs/train`: Output directory for training DCP checkpoints (default is repo-relative). We recommend at least 1 TB of free disk space.
- `COSMOS_SMOKE=1`: Enable smoke test.
- `COSMOS_VERBOSE=1`: Enable verbose console output.
- `COSMOS_GAPPED_REPLAY_TORCH=1`: Run maskless replay's gapped same-view attention forward in PyTorch (FlashAttention-2 varlen) instead of FlashAttention-4 on any NVIDIA GPU. Workstation Blackwell (sm12x: RTX PRO 6000 / RTX 50) uses this path without the variable. Inference only: a backward through it raises.
- `WANDB_API_KEY`: Required when the recipe TOML's `job.wandb_mode = "online"`. Recipes default to `wandb_mode = "disabled"`; flip the TOML and export this key to log to [Weights & Biases](https://wandb.ai/).

[Hugging Face](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables):

- [HF_HOME]: Directory for checkpoints and datasets. We recommend at least 1 TB of free disk space.

[UV](https://docs.astral.sh/uv/reference/environment/):

- [UV_CACHE_DIR]: Directory for dependencies.
