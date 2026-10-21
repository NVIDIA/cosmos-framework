# Prepare the Transfer training data

Use [example2 on Hugging Face](https://huggingface.co/nvidia/Cosmos3-Nano-Sim-Transfer/tree/main/assets/sample_data/example2)
for the data-loading and Stage 0–3 walkthrough. Its RGB, depth and segmentation
videos contain 121 frames at 30 FPS and 1920 × 1080. Its structured prompt
describes that full interval.

## Build the example2 bundle

Complete [setup](setup.md) and
[download and prepare example2](hf-sample.md#download-and-prepare-example2).
The HF sample contains media and prompts. Build the manifest, sample index and
tokenizer resources required by Framework's `LocalSimulationDataset`:

```bash
export SIM_TRANSFER_TOKENIZER_SNAPSHOT="$(hf download nvidia/Cosmos3-Nano \
  --revision e59a53c25979a090fa8706c9acc0c254a6e89b92 --include 'text_tokenizer/*')"
export SIM_TRANSFER_TEXT_TOKENIZER="$SIM_TRANSFER_TOKENIZER_SNAPSHOT/text_tokenizer"
export SIM_TRANSFER_TRAINING_BUNDLE="$SIM_TRANSFER_OUTPUT/training-bundle"
python "$SIM_TRANSFER_SKILL/scripts/paibench_local_bundle.py" prepare-custom \
  --manifest "$SIM_TRANSFER_OUTPUT/training.jsonl" \
  --tokenizer "$SIM_TRANSFER_TEXT_TOKENIZER" \
  --output "$SIM_TRANSFER_TRAINING_BUNDLE" \
  --dataset-name sim_transfer_example2
export COSMOS_SIM_TRANSFER_LOCAL_BUNDLE="$SIM_TRANSFER_TRAINING_BUNDLE"
```

Use a fresh bundle destination when rebuilding. The preparation wrapper calls
Framework's native `build_transfer_bundle`, retains the source media and embeds
the reviewed structured caption in each indexed record. The training recipe
consumes this bundle, rather than the downloaded media folder.

### Read media from an existing captured bundle

Follow each record's `assets` mapping in `samples.json`. The Transfer loader
reads the mapped RGB, depth and segmentation files directly as encoded media;
it does not extract a tar archive at those paths. In older captured bundles,
an asset named `assets/<modality>/<hash>.tar` can contain an MP4 payload copied
from a source archive's byte range. Check its actual format with `file` or
`ffprobe` rather than running `tar` on that individual asset. Extract the outer
bundle archive once, then retain its relative paths and sample index together.

The current native builder writes those media files with `.mp4` names. To use
an older captured asset as an inference control, select the aligned clip and
modality from its record, verify its decoded frame count, FPS and geometry,
then copy it to a separate `.mp4` path for the inference request. Renaming an
unrelated payload does not convert it to video. Keep the training bundle intact
so its manifest and resume identity continue to match.

## Native preprocessing

The loader chooses one control per draw with equal weights across edge, blur,
depth and segmentation. It resizes RGB directly with antialiased bicubic,
depth with antialiased bilinear, and segmentation with nearest neighbor.
It computes edge and blur from the resized RGB during loading. No precomputed
edge or blur video is needed. Control and target share the same frame sampling
and canvas; reflection padding is not used.

At the 480 tier, example2 produces control and target tensors of shape
`[3,121,480,832]`. At 720, the shape is `[3,121,720,1280]`. Stage 0 retains its
480/720 mixture; Stages 1–3 use 480. The bundle builder retains the native
length index so ranks in a training group draw the same latent length.

The structured annotation comes from `example2_upsampled_prompt.json` on HF.
Preparation removes saved output metadata so the loader can append the actual
sampled duration, FPS and resolution. After text retention, training chooses
the structured JSON with probability 0.93 and its `temporal_caption` with
probability 0.07. `example2_prompt.txt` is the separate source prompt, not the
7% branch. Stages 0–2 apply text dropout 0.1; Stage 3 applies zero dropout.

Continue with [Stage 0–3 training](../post-training/references/local-training.md).
Use this single example to learn the workflow; use separate training and
held-out recordings for model development.

## Prepare your own examples

Use the [captioning guide](../post-training/references/caption-guide.md) to produce a
structured caption with a nonempty `temporal_caption`. Each training-input row
names one RGB clip, its optional depth and segmentation controls, and its
structured caption. The example2 row demonstrates this format:

```json
{
  "identity": "example2",
  "rgb": "example2.mp4",
  "depth": "example2_depth.mp4",
  "seg": "example2_segmentation.mp4",
  "caption_path": "example2_upsampled_prompt.json",
  "caption_start": 0,
  "caption_end": 121
}
```

The preparation script fills those fields from the HF download. For another
dataset, supply one JSONL row per clip to the same `prepare-custom` command.
Relative asset names resolve beside the manifest. Caption bounds are
start-inclusive and end-exclusive and cover the entire decoded clip.
For a shorter interval, first export aligned media and caption that interval.

Keep the original media resolution. RGB and a caption are required; include
depth and segmentation to retain the equal four-modality mixture. Edge and
blur are generated during loading. Preparation excludes misaligned media,
invalid captions and clips that exceed the native geometry or temporal limits.
It selects an aspect bucket shared by the 480/720 streams and applies the 3%
distortion limit at both tiers. Rejected rows appear in the preparation summary;
accepted clips continue into the native bundle.
