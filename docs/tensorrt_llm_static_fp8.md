# Cosmos3 Nano and Super static FP8 with TensorRT-LLM

TensorRT-LLM can run ModelOpt-calibrated Cosmos3 Nano and Super FP8
checkpoints directly. The checkpoint metadata selects static FP8, including
its calibrated weight and activation scales, so no quantization CLI flag is
needed.

This guide covers the four validated single-GPU modes:

| Mode                 | Prompt or condition                   | Output    |
| -------------------- | ------------------------------------- | --------- |
| Text to image (T2I)  | Structured text prompt                | PNG image |
| Text to video (T2V)  | Structured text prompt                | MP4 video |
| Image to video (I2V) | Structured prompt and reference image | MP4 video |
| Video to video (V2V) | Structured prompt and reference video | MP4 video |

FP8 Cosmos3 is single-GPU only. TensorRT-LLM rejects tensor parallelism,
Ulysses parallelism, context parallelism, CFG parallelism, or parallel VAE
sizes greater than one for these checkpoints. Use a BF16 checkpoint for a
multi-GPU deployment.

## Prerequisites

Build and install TensorRT-LLM from its `main` branch by following the
[TensorRT-LLM source-build guide](https://nvidia.github.io/TensorRT-LLM/installation/build-from-source.html).
The source checkout supplies the Cosmos3 example, prompt files, and one-GPU
configuration files used below.

Install `ffmpeg` so TensorRT-LLM can encode MP4 output. Install the Cosmos3
guardrail package, accept the gated
[`nvidia/Cosmos-1.0-Guardrail`](https://huggingface.co/nvidia/Cosmos-1.0-Guardrail)
terms, and authenticate to Hugging Face before generation:

```bash
pip install cosmos_guardrail==0.3.0
pip uninstall -y opencv-python
pip install opencv-python-headless
sudo apt-get install -y ffmpeg
hf auth login
```

Obtain local ModelOpt FP8 checkpoint directories for Cosmos3 Nano and Super.
There are no directly loadable FP8 Hub IDs; pass each local checkpoint
directory to `--model`.

Set the paths used by the commands in this guide:

```bash
export TRTLLM_ROOT=/path/to/TensorRT-LLM
export COSMOS3_NANO_FP8=/path/to/cosmos3-nano-fp8-14072026
export COSMOS3_SUPER_FP8=/path/to/cosmos3-super-fp8-14072026
export COSMOS3_FP8_OUTPUTS=$PWD/outputs/tensorrt_llm_static_fp8
mkdir -p "$COSMOS3_FP8_OUTPUTS"
```

The checkpoints may carry a `diffusion_step_policy` in their quantization
metadata. TensorRT-LLM applies that checkpoint-owned policy automatically; do
not add a separate runtime override for it.

## Cosmos3 Nano FP8

### Text to image

The text-to-image config warms the 1024x1024, one-frame shape. The structured
prompt selects image mode, and `--output_type image` makes the expected output
explicit.

```bash
python "$TRTLLM_ROOT/examples/visual_gen/models/cosmos3/cosmos3.py" \
  --model "$COSMOS3_NANO_FP8" \
  --visual_gen_args "$TRTLLM_ROOT/examples/visual_gen/configs/cosmos3-t2i-1gpu.yaml" \
  --prompt_file "$TRTLLM_ROOT/examples/visual_gen/models/cosmos3/prompts/t2i.json" \
  --output_type image \
  --output_path "$COSMOS3_FP8_OUTPUTS/nano_t2i.png"
```

### Text to video

```bash
python "$TRTLLM_ROOT/examples/visual_gen/models/cosmos3/cosmos3.py" \
  --model "$COSMOS3_NANO_FP8" \
  --visual_gen_args "$TRTLLM_ROOT/examples/visual_gen/configs/cosmos3-nano-1gpu.yaml" \
  --prompt_file "$TRTLLM_ROOT/examples/visual_gen/models/cosmos3/prompts/t2v.json" \
  --output_path "$COSMOS3_FP8_OUTPUTS/nano_t2v.mp4"
```

### Image to video

The example I2V prompt names its reference image with an HTTPS URL. Use
`--image_path` to replace it with a local path, `file://` URL, HTTP(S) URL, or
`data:` URI.

```bash
python "$TRTLLM_ROOT/examples/visual_gen/models/cosmos3/cosmos3.py" \
  --model "$COSMOS3_NANO_FP8" \
  --visual_gen_args "$TRTLLM_ROOT/examples/visual_gen/configs/cosmos3-nano-1gpu.yaml" \
  --prompt_file "$TRTLLM_ROOT/examples/visual_gen/models/cosmos3/prompts/i2v.json" \
  --output_path "$COSMOS3_FP8_OUTPUTS/nano_i2v.mp4"
```

### Video to video

This command uses the Nano T2V output above as the reference video. By default,
the first five decoded pixel frames condition the output.

```bash
python "$TRTLLM_ROOT/examples/visual_gen/models/cosmos3/cosmos3.py" \
  --model "$COSMOS3_NANO_FP8" \
  --visual_gen_args "$TRTLLM_ROOT/examples/visual_gen/configs/cosmos3-nano-1gpu.yaml" \
  --prompt_file "$TRTLLM_ROOT/examples/visual_gen/models/cosmos3/prompts/v2v.json" \
  --video_path "$COSMOS3_FP8_OUTPUTS/nano_t2v.mp4" \
  --output_path "$COSMOS3_FP8_OUTPUTS/nano_v2v.mp4"
```

## Cosmos3 Super FP8

The same one-GPU interfaces apply to the Super checkpoint. Do not use
`cosmos3-super-4gpu.yaml`; static FP8 rejects that parallel configuration.

### Text to image

```bash
python "$TRTLLM_ROOT/examples/visual_gen/models/cosmos3/cosmos3.py" \
  --model "$COSMOS3_SUPER_FP8" \
  --visual_gen_args "$TRTLLM_ROOT/examples/visual_gen/configs/cosmos3-t2i-1gpu.yaml" \
  --prompt_file "$TRTLLM_ROOT/examples/visual_gen/models/cosmos3/prompts/t2i.json" \
  --output_type image \
  --output_path "$COSMOS3_FP8_OUTPUTS/super_t2i.png"
```

### Text to video

The file name `cosmos3-nano-1gpu.yaml` is historical: TensorRT-LLM documents
the configuration as the shared one-GPU config for both Nano and Super.

```bash
python "$TRTLLM_ROOT/examples/visual_gen/models/cosmos3/cosmos3.py" \
  --model "$COSMOS3_SUPER_FP8" \
  --visual_gen_args "$TRTLLM_ROOT/examples/visual_gen/configs/cosmos3-nano-1gpu.yaml" \
  --prompt_file "$TRTLLM_ROOT/examples/visual_gen/models/cosmos3/prompts/t2v.json" \
  --output_path "$COSMOS3_FP8_OUTPUTS/super_t2v.mp4"
```

### Image to video

```bash
python "$TRTLLM_ROOT/examples/visual_gen/models/cosmos3/cosmos3.py" \
  --model "$COSMOS3_SUPER_FP8" \
  --visual_gen_args "$TRTLLM_ROOT/examples/visual_gen/configs/cosmos3-nano-1gpu.yaml" \
  --prompt_file "$TRTLLM_ROOT/examples/visual_gen/models/cosmos3/prompts/i2v.json" \
  --output_path "$COSMOS3_FP8_OUTPUTS/super_i2v.mp4"
```

### Video to video

```bash
python "$TRTLLM_ROOT/examples/visual_gen/models/cosmos3/cosmos3.py" \
  --model "$COSMOS3_SUPER_FP8" \
  --visual_gen_args "$TRTLLM_ROOT/examples/visual_gen/configs/cosmos3-nano-1gpu.yaml" \
  --prompt_file "$TRTLLM_ROOT/examples/visual_gen/models/cosmos3/prompts/v2v.json" \
  --video_path "$COSMOS3_FP8_OUTPUTS/super_t2v.mp4" \
  --output_path "$COSMOS3_FP8_OUTPUTS/super_v2v.mp4"
```

## Verify the artifacts

Decode the two PNG files and print their dimensions:

```bash
python - \
  "$COSMOS3_FP8_OUTPUTS/nano_t2i.png" \
  "$COSMOS3_FP8_OUTPUTS/super_t2i.png" <<'PY'
from PIL import Image
import sys

for path in sys.argv[1:]:
    with Image.open(path) as image:
        image.verify()
        print(path, image.format, image.size)
PY
```

Inspect and fully decode every generated video stream:

```bash
for output in \
  "$COSMOS3_FP8_OUTPUTS/nano_t2v.mp4" \
  "$COSMOS3_FP8_OUTPUTS/nano_i2v.mp4" \
  "$COSMOS3_FP8_OUTPUTS/nano_v2v.mp4" \
  "$COSMOS3_FP8_OUTPUTS/super_t2v.mp4" \
  "$COSMOS3_FP8_OUTPUTS/super_i2v.mp4" \
  "$COSMOS3_FP8_OUTPUTS/super_v2v.mp4"; do
  ffprobe -v error \
    -show_entries stream=codec_type,codec_name,width,height,r_frame_rate,nb_frames \
    -of json \
    "$output"
  ffmpeg -v error -i "$output" -f null -
done
```

A successful encode/decode proves the interface and artifact contract; it is
not a claim that FP8 output quality matches BF16.

## Scope

The Nano and Super FP8 checkpoints also contain an audio tower, so T2AV and
TI2AV requests run rather than being refused. Audio has not received the same
FP8 exercise as the four image/video modes above, and no FP8 audio-quality
claim is made here.

For the complete TensorRT-LLM Cosmos3 example contract, including serving,
guardrails, media dependencies, and other checkpoint families, see the
[TensorRT-LLM Cosmos3 example](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/visual_gen/models/cosmos3).
