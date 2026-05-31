# Prompt Upsampling

> **Skill:** `.agents/skills/cosmos3-inference/SKILL.md`

<!--TOC-->

______________________________________________________________________

**Table of Contents**

- [Overview](#overview)
- [Setup](#setup)
- [Inputs](#inputs)
- [Outputs](#outputs)
- [Quick Start](#quick-start)
  - [Text-to-Image](#text-to-image)
  - [Text-to-Video](#text-to-video)
  - [Image-to-Video](#image-to-video)
- [Reasoner Templates](#reasoner-templates)
- [Posttrained I2V](#posttrained-i2v)
- [Template Files](#template-files)

______________________________________________________________________

<!--TOC-->

## Overview

The prompt upsampler converts one prompt per line into structured Cosmos3 JSON prompts. It supports:

- `text2image`
- `text2video`
- `image2video`
- `posttrain_image2video`

Run it as:

```shell
python -m cosmos_framework.inference.prompt_upsampling --help
```

## Setup

Run commands from the repository root. Configure the OpenAI-compatible endpoint and API token:

```shell
export PROMPT_UPSAMPLER_ENDPOINT_URL="[required endpoint url]"
export PROMPT_UPSAMPLER_API_TOKEN="[api key]"
export PROMPT_UPSAMPLER_MODEL_NAME="[model name]"
```

Arguments:

- `--input`: Text file with one prompt per non-empty line.
- `--output`: Output directory. The CLI writes one `prompt_<index>.json` file per input prompt.
- `--mode`: One of `text2image`, `text2video`, `image2video`, or `posttrain_image2video`.
- `--endpoint-url`: OpenAI-compatible endpoint URL. Defaults to `PROMPT_UPSAMPLER_ENDPOINT_URL`.
- `--model`: Model name. Defaults to `PROMPT_UPSAMPLER_MODEL`.
- `--api-token`: API token. Defaults to `PROMPT_UPSAMPLER_API_TOKEN`.
- `--prompt-template`: Optional prompt template override for the selected mode.
- `--json-template`: Optional JSON schema template override for the selected mode.

## Inputs

Prompt files contain one non-empty prompt per line:

```text
A humanoid robot carefully assembles a gearbox on a workbench.
A red sports car drives through rain at night.
```

For `image2video` and `posttrain_image2video`, provide either one shared image:

```shell
--image-url inputs/prompt_upsampler/image_inputs/car_driving.jpg
```

Or provide one image per prompt:

```shell
--image-list inputs/prompt_upsampler/images.txt
```

Image-list files must have the same number of non-empty lines as the prompt file:

```text
inputs/prompt_upsampler/image_inputs/car_driving.jpg
inputs/prompt_upsampler/image_inputs/humanoid_robot.jpg
```

Local image paths, HTTP(S) URLs, and `data:` URLs are accepted.

## Outputs

The CLI creates the output directory and writes one JSON file per input prompt:

```text
outputs/prompt_upsampler/upsampled_t2i_prompts_opus/
+-- prompt_0.json
`-- prompt_1.json
```

Every mode writes one record per prompt with the same shape: the upsampled JSON is a compact string under `prompt`. A record also includes a `negative_prompt` string when the template or model produces one.

```json
{
  "prompt": "{\"subjects\": [...], \"resolution\": {\"H\": 480, \"W\": 832}, ...}"
}
```

## Quick Start

The built-in external API templates are used by default. They live in `cosmos_framework/inference/prompting_templates/external_api`.

For example, to use Opus to generate upsampled prompts, set

```shell
export PROMPT_UPSAMPLER_ENDPOINT_URL="https://api.anthropic.com/v1/"
export PROMPT_UPSAMPLER_MODEL_NAME="claude-opus-4-6"
```

### Text-to-Image

```shell
python -m cosmos_framework.inference.prompt_upsampling \
    --input inputs/prompt_upsampler/prompts_t2i.txt \
    --output outputs/prompt_upsampler/upsampled_t2i_prompts_opus \
    --mode text2image \
    --endpoint-url "${PROMPT_UPSAMPLER_ENDPOINT_URL}" \
    --model "${PROMPT_UPSAMPLER_MODEL_NAME}" \
    --api-token "${PROMPT_UPSAMPLER_API_TOKEN}" \
    --resolution 720 \
    --aspect-ratio "1,1"
```

### Text-to-Video

```shell
python -m cosmos_framework.inference.prompt_upsampling \
    --input inputs/prompt_upsampler/prompts_t2v.txt \
    --output outputs/prompt_upsampler/upsampled_t2v_prompts_opus \
    --mode text2video \
    --endpoint-url "${PROMPT_UPSAMPLER_ENDPOINT_URL}" \
    --model "${PROMPT_UPSAMPLER_MODEL_NAME}" \
    --api-token "${PROMPT_UPSAMPLER_API_TOKEN}" \
    --resolution 720 \
    --aspect-ratio "16,9" \
    --duration "5s" \
    --fps 24
```

### Image-to-Video

```shell
python -m cosmos_framework.inference.prompt_upsampling \
    --input inputs/prompt_upsampler/prompts_i2v.txt \
    --image-list inputs/prompt_upsampler/images.txt \
    --output outputs/prompt_upsampler/upsampled_i2v_prompts_opus \
    --mode image2video \
    --endpoint-url "${PROMPT_UPSAMPLER_ENDPOINT_URL}" \
    --model "${PROMPT_UPSAMPLER_MODEL_NAME}" \
    --api-token "${PROMPT_UPSAMPLER_API_TOKEN}" \
    --resolution 720 \
    --aspect-ratio "16,9" \
    --duration "5s" \
    --fps 24
```

## Reasoner Templates

Reasoner-style prompts use the templates in `cosmos_framework/inference/prompting_templates/reasoner`. Pass them explicitly with `--prompt-template` and `--json-template`.

For using reasoner prompt upsampler in this script, launch a VLLM server. Set the model name, endpoint url and API token served by your endpoint:

```shell
export PROMPT_UPSAMPLER_REASONER_MODEL="cosmos3-reasoner"
```

Text-to-image:

```shell
python -m cosmos_framework.inference.prompt_upsampling \
    --input inputs/prompt_upsampler/prompts_t2i.txt \
    --output outputs/prompt_upsampler/upsampled_t2i_prompts_reason \
    --mode text2image \
    --endpoint-url "${PROMPT_UPSAMPLER_ENDPOINT_URL}" \
    --model "${PROMPT_UPSAMPLER_REASONER_MODEL}" \
    --api-token "${PROMPT_UPSAMPLER_API_TOKEN}" \
    --prompt-template cosmos_framework/inference/prompting_templates/reasoner/reasoner_t2i_prompt.txt \
    --json-template cosmos_framework/inference/prompting_templates/reasoner/reasoner_t2i_json_schema.json \
    --resolution 720 \
    --aspect-ratio "1,1"
```

Text-to-video:

```shell
python -m cosmos_framework.inference.prompt_upsampling \
    --input inputs/prompt_upsampler/prompts_t2v.txt \
    --output outputs/prompt_upsampler/upsampled_t2v_prompts_reason \
    --mode text2video \
    --endpoint-url "${PROMPT_UPSAMPLER_ENDPOINT_URL}" \
    --model "${PROMPT_UPSAMPLER_REASONER_MODEL}" \
    --api-token "${PROMPT_UPSAMPLER_API_TOKEN}" \
    --prompt-template cosmos_framework/inference/prompting_templates/reasoner/reasoner_t2v_prompt.txt \
    --json-template cosmos_framework/inference/prompting_templates/reasoner/reasoner_t2v_json_schema.json \
    --resolution 720 \
    --aspect-ratio "16,9" \
    --duration "5s" \
    --fps 24
```

Image-to-video:

```shell
python -m cosmos_framework.inference.prompt_upsampling \
    --input inputs/prompt_upsampler/prompts_i2v.txt \
    --image-list inputs/prompt_upsampler/images.txt \
    --output outputs/prompt_upsampler/upsampled_i2v_prompts_reason \
    --mode image2video \
    --endpoint-url "${PROMPT_UPSAMPLER_ENDPOINT_URL}" \
    --model "${PROMPT_UPSAMPLER_REASONER_MODEL}" \
    --api-token "${PROMPT_UPSAMPLER_API_TOKEN}" \
    --prompt-template cosmos_framework/inference/prompting_templates/reasoner/reasoner_i2v_prompt.txt \
    --json-template cosmos_framework/inference/prompting_templates/reasoner/reasoner_i2v_json_schema.json \
    --resolution 720 \
    --aspect-ratio "16,9" \
    --duration "5s" \
    --fps 24
```

## Posttrained I2V

The `posttrain_image2video` mode targets the post-trained Cosmos3-Super-Image2Video model. It produces both an upsampled JSON prompt and a per-sample contextual negative prompt. We use "claude-opus-4-7" for our Cosmos3-Super-Image2Video model.

```shell
python -m cosmos_framework.inference.prompt_upsampling \
    --input inputs/prompt_upsampler/prompts_i2v.txt \
    --image-list inputs/prompt_upsampler/images.txt \
    --output outputs/prompt_upsampler/upsampled_i2v_pos_neg \
    --mode posttrain_image2video \
    --endpoint-url "${PROMPT_UPSAMPLER_ENDPOINT_URL}" \
    --model "${PROMPT_UPSAMPLER_MODEL_NAME}" \
    --api-token "${PROMPT_UPSAMPLER_API_TOKEN}" \
    --prompt-template cosmos_framework/inference/prompting_templates/external_api/posttrained_i2v_prompt.txt \
    --json-template cosmos_framework/inference/prompting_templates/external_api/posttrained_i2v_json_schema.json \
    --resolution 480 \
    --aspect-ratio "16,9" \
    --duration "8s" \
    --fps 24
```

The posttrained I2V templates emit a `<negative_prompt>` block, so each output record also carries a `negative_prompt` field.

## Template Files

Default external API templates:

- `cosmos_framework/inference/prompting_templates/external_api/t2i_prompt.txt`
- `cosmos_framework/inference/prompting_templates/external_api/t2i_json_schema.json`
- `cosmos_framework/inference/prompting_templates/external_api/t2v_i2v_video_prompt.txt`
- `cosmos_framework/inference/prompting_templates/external_api/t2v_i2v_video_json_schema.json`
- `cosmos_framework/inference/prompting_templates/external_api/posttrained_i2v_prompt.txt`
- `cosmos_framework/inference/prompting_templates/external_api/posttrained_i2v_json_schema.json`

Reasoner templates:

- `cosmos_framework/inference/prompting_templates/reasoner/reasoner_t2i_prompt.txt`
- `cosmos_framework/inference/prompting_templates/reasoner/reasoner_t2i_json_schema.json`
- `cosmos_framework/inference/prompting_templates/reasoner/reasoner_t2v_prompt.txt`
- `cosmos_framework/inference/prompting_templates/reasoner/reasoner_t2v_json_schema.json`
- `cosmos_framework/inference/prompting_templates/reasoner/reasoner_i2v_prompt.txt`
- `cosmos_framework/inference/prompting_templates/reasoner/reasoner_i2v_json_schema.json`
