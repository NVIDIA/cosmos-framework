# Environment Variables

- `IMAGINAIRE_OUTPUT_ROOT=outputs/train`: Output directory for training DCP checkpoints (default is repo-relative). We recommend at least 1 TB of free disk space.
- `COSMOS_SMOKE=1`: Enable smoke test.
- `COSMOS_VERBOSE=1`: Enable verbose console output.
- `WANDB_API_KEY`: Required when the recipe TOML's `job.wandb_mode = "online"`. Recipes default to `wandb_mode = "disabled"`; flip the TOML and export this key to log to [Weights & Biases](https://wandb.ai/).

[Hugging Face](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables):

- [HF_HOME]: Directory for checkpoints and datasets. We recommend at least 1 TB of free disk space.

[UV](https://docs.astral.sh/uv/reference/environment/):

- [UV_CACHE_DIR]: Directory for dependencies.
