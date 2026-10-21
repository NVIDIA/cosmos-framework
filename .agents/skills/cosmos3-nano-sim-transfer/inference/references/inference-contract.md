# Inference parameters

These settings define the Framework Stage 3 SF-DMD recipe used by the
[Quick Start](../../shared/quick-start.md). Use the [shared model contract](../../shared/model-contract.md)
for control, prompt, geometry and timing. Other checkpoints need their
matching config and runtime.

[Checkpoint selection](../SKILL.md#1-select-the-matched-inference-checkpoint)
covers the HF release directory with its own `config.json` and native DCP.
EMA selection applies to DCP; an HF export already contains the chosen weights.

## Bundled request defaults

Set `--defaults-file` to
`$SIM_TRANSFER_SKILL/inference/assets/sample_args.json`
for the Stage 3 profile. Locally prepared requests override its 24 FPS with
30 FPS; decoded input media determines the actual transfer timeline.

Run `cosmos_framework.scripts.inference` from Framework. Native DCP uses
`cosmos_framework/configs/simulation/local.py` and `sim_local_transfer`.
Hydra resolves the full recipe before inference disables training work, so
supply the Stage 0/Stage 2 parent and prepared-bundle environment bindings.
HF uses its own `config.json` without a training experiment. Use native DCP
EMA checkpoints for inference from newly trained Stage 3 models.

The [HF assets](../../shared/hf-sample.md) contain source RGB, depth,
segmentation and prompts. The walkthrough creates request JSON locally.
For `example2`, native inference derives edge and blur from RGB after resizing.

## Preserve geometry and time

Select the least-distorted supported canvas, with at most 3% aspect
distortion, and preserve the full field of view. The shared contract defines
interpolation for each modality. Precomputed depth and segmentation must
share the RGB camera, source times and field of view.

The native paired-input checks allow one-pixel aspect rounding, relative FPS
difference up to `1e-3` and relative timestamp differences within half a
reference frame. Independently looping, trimming or retiming one stream
breaks alignment.

| Example    | Use                                                | Source video                      | Processing window / decoding cap | Model output                    |
| ---------- | -------------------------------------------------- | --------------------------------- | -------------------------------- | ------------------------------- |
| `example2` | Complete workflow and four-control inference       | 1920 × 1080, 121 frames at 30 FPS | 121 / 121                        | 832 × 480, 121 frames at 30 FPS |
| `example1` | Long-rollout inference only, depth or segmentation | 384 × 384, 600 frames at 30 FPS   | 601 / 600                        | 640 × 640, 600 frames at 30 FPS |

The causal loader preserves source frame count for one-latent chunks.
`example1` uses a 601-frame processing window and `max_frames: 600` to retain
all 600 decoded frames; it does not trim the clip to an aligned count.
Changing `num_frames` or `fps` alone does not extend or retime input.

## Parameter policy

| Parameter               | Reference setting                                               | Purpose                                                       |
| ----------------------- | --------------------------------------------------------------- | ------------------------------------------------------------- |
| Weights                 | Matched Stage 3 SF-DMD HF export, or native DCP EMA             | Keep runtime, model config, tokenizer and autoencoder matched |
| Control                 | One of edge, blur, depth or segmentation                        | Control first, RGB target second; shared temporal positions   |
| Causal chunk            | One latent                                                      | Match the causal model and autoencoder factor four            |
| Sampler                 | `[1, 15/16, 5/6, 5/8]` then zero                                | Preserve the already-shifted four-step stochastic schedule    |
| Guidance and RGB prefix | Text/control guidance 1; zero real prefix                       | Use the distilled Stage 3 input route                         |
| Geometry                | Direct resize, at most 3% distortion, 480 tier                  | Preserve field of view and the supported canvas               |
| Caption and seed        | One window-matched caption; seed 2025                           | Hold fixed for comparisons                                    |
| Numerical runtime       | BF16; context/guidance parallelism 1; diffusion cache off       | Choose eager or the optimized compile-and-graph path          |
| Delivery                | Model-native RGB                                                | Keep resized delivery copies separate                         |

`example2` uses full history with no sink:
`--kv-cache-inference-size 0 --attention-sink-size 0`. These flags apply to
both HF and DCP, mapping to `model.config.kv_cache_inference_size=null` and
`model.config.attention_sink_size=0`.

The [example1 long-rollout](../SKILL.md#run-example1-long-rollout) uses
`--kv-cache-inference-size 30 --attention-sink-size 3`: a finite 30-latent
history window with three pinned entries. A positive cache size is measured
in latent frames. Attention sinks require a finite window. Cache eviction
must preserve the temporal position clock.

## Runtime options

Use BF16 attention-cache storage (`model.config.kv_cache_dtype=null`).
The [optimized inference command](../SKILL.md#use-optimized-inference) pairs
`torch.compile` with CUDA Graphs, compiles language decoder blocks, and keeps
one latent per autoregressive step. Eager execution disables both compilation
and graphs. Direct resize requires non-streaming control encoding
(`condition_stream_encode=false`).

Start a new request for a different caption. See the
[output guide](output.md) for saved files and overwrite behavior.

## Native outputs

The walkthrough sets `transfer_output_size: model`, `show_input: false` and
`show_control_condition: false`. Its `vision.mp4` is model-native RGB. A saved
`control_<modality>.mp4` shows the conditioning input.

For source-size delivery, `transfer_output_size: original` retains
`vision_model.mp4` and writes the resized copy as `vision.mp4`. Restoration
uses bicubic antialiasing and clamps the range. An odd source dimension is
padded by one bottom or right pixel for encoding. Native
`sample_outputs.json` records height/width pairs in
`outputs[].content.transfer_geometry`: `source_hw`, `model_hw`, `resized_hw`,
`output_hw` and `encoder_padding_hw`. `native_vision_file` identifies the
retained model-size file. See [output files](output.md).
