# Download the Hugging Face examples

Download the examples from [Cosmos3-Nano-Sim-Transfer](https://huggingface.co/nvidia/Cosmos3-Nano-Sim-Transfer)
at revision `03efb0d86d004fdfc64ffc867dffd5de5df39826`.

| Example                                                                                                                                       | Source width × height | Frames / FPS | Use                                                                                     |
| --------------------------------------------------------------------------------------------------------------------------------------------- | --------------------- | ------------ | --------------------------------------------------------------------------------------- |
| [example2](https://huggingface.co/nvidia/Cosmos3-Nano-Sim-Transfer/tree/03efb0d86d004fdfc64ffc867dffd5de5df39826/assets/sample_data/example2) | 1920 × 1080           | 121 / 30     | Default for prompting, data preparation, Stage 0–3 training and four-modality inference |
| [example1](https://huggingface.co/nvidia/Cosmos3-Nano-Sim-Transfer/tree/03efb0d86d004fdfc64ffc867dffd5de5df39826/assets/sample_data/example1) | 384 × 384             | 600 / 30     | Long-rollout depth and segmentation walkthrough                                         |

Each HF folder contains five files: source RGB, depth, segmentation, a plain-text
prompt and an upsampled prompt JSON. Use `example2` for the complete workflow.
The [long-rollout walkthrough](../inference/SKILL.md#run-example1-long-rollout)
uses `example1` with depth and segmentation controls.

## Download and prepare example2

Complete [Framework setup](setup.md) and use your
Hugging Face login. Run in Bash with the Framework environment active.
`SIM_TRANSFER_OUTPUT` is your chosen destination for prepared requests,
training inputs and generated results. Choose a fresh directory; rerunning
preparation replaces its request JSON files and `training.jsonl`.

```bash
export SIM_TRANSFER_ASSET_REVISION="03efb0d86d004fdfc64ffc867dffd5de5df39826"
export SIM_TRANSFER_DATA_ROOT="$(hf download nvidia/Cosmos3-Nano-Sim-Transfer \
  --revision "$SIM_TRANSFER_ASSET_REVISION" --include 'assets/sample_data/example2/*')"
export SIM_TRANSFER_SAMPLE_ID="example2"
export SIM_TRANSFER_DATASET="$SIM_TRANSFER_DATA_ROOT/assets/sample_data/example2"
export SIM_TRANSFER_OUTPUT="${SIM_TRANSFER_OUTPUT:?Choose a destination for the example2 workflow}"
python "$SIM_TRANSFER_SKILL/scripts/prepare_hf_sample.py" \
  --example example2 --sample-dir "$SIM_TRANSFER_DATASET" \
  --output "$SIM_TRANSFER_OUTPUT"
```

This downloads only the five `example2` assets. The preparation script creates
four native inference requests and one training-input row from those assets.
It uses the saved upsampled JSON as the prompt. Edge and blur requests pass
source RGB to the native transforms; they do not require stored control videos.
Depth and segmentation requests use their HF control videos.

Continue with the [Quick Start](quick-start.md). For training, [build the native
bundle](local-dataset.md) from the prepared row and a Hugging Face tokenizer.
The source videos retain their original resolution; the pipeline selects
832 × 480 at the 480 tier or 1280 × 720 at the 720 tier.

Required model resources come from [Cosmos3-Nano-Sim-Transfer](https://huggingface.co/nvidia/Cosmos3-Nano-Sim-Transfer),
[Cosmos3-Nano](https://huggingface.co/nvidia/Cosmos3-Nano) and
[Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B).
The [inference guide](../inference/SKILL.md#1-select-the-matched-inference-checkpoint)
and [Nano conversion guide](../post-training/references/nano-conversion.md)
provide their download commands.
