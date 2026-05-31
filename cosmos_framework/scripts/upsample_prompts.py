# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import asyncio
import base64
import json
import math
import mimetypes
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import openai
import pydantic
import tyro
from tqdm import tqdm

from cosmos_framework.inference.args import ModelMode, OmniSampleArgs, OmniSampleOverrides, OmniSetupOverrides
from cosmos_framework.model.vfm.upsampler.prompts import build_messages, clean_response
from cosmos_framework.utils import log

if TYPE_CHECKING:
    from cosmos_framework.configs.base.defaults.model_config import OmniMoTModelConfig

_PACKAGE_DIR = Path(__file__).parents[1].absolute()


class PromptUpsamplerArgs(pydantic.BaseModel):
    endpoint_url: str = "http://localhost:8000/v1"
    """The URL of the API server."""
    model: str | None = None
    """The model to use.

    If not provided, the first model in the list will be used.
    """

    debug: bool = False
    """If True, save raw API responses for debugging."""
    max_workers: int = 16
    """Maximum number of concurrent requests to the API."""
    max_retries: int = 5
    """Maximum number of retries for each request."""


class Args(pydantic.BaseModel):
    input_files: Annotated[list[Path], tyro.conf.arg(aliases=("-i",))]
    """Path to the input sample argument files."""
    # output_dir: Annotated[Path, tyro.conf.arg(aliases=("-o",))]
    # """Output directory."""

    setup: tyro.conf.OmitArgPrefixes[OmniSetupOverrides] = OmniSetupOverrides.model_construct()
    """Setup arguments."""
    prompt_upsampler: PromptUpsamplerArgs = PromptUpsamplerArgs.model_construct()
    """Prompt upsampler arguments."""


class Sample(pydantic.BaseModel):
    overrides: OmniSampleOverrides
    args: OmniSampleArgs
    messages: list


_TASKS = {
    ModelMode.TEXT2IMAGE: "t2i",
    ModelMode.TEXT2VIDEO: "t2v",
    ModelMode.IMAGE2VIDEO: "i2v",
}


def _dump_json(obj: Any, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))


def _model_dump_json(obj: pydantic.BaseModel, path: Path, **kwargs):
    _dump_json(obj.model_dump(mode="json", **kwargs), path)


async def _process_sample(
    args: Args,
    client: openai.AsyncOpenAI,
    sample: Sample,
):
    assert args.prompt_upsampler.model
    for i_retry in range(args.prompt_upsampler.max_retries):
        msg_prefix = f"['{sample.args.name}'|{i_retry + 1}]"
        # Send request
        try:
            response = await client.chat.completions.create(
                model=args.prompt_upsampler.model,
                messages=sample.messages,
                seed=i_retry,
                max_tokens=20000,
                temperature=0.7,
                top_p=0.8,
                presence_penalty=1.5,
                extra_body={"top_k": 20, "min_p": 0.0},
            )
        except Exception as e:
            log.warning(f"{msg_prefix} API Error: {e}")
            await asyncio.sleep(1)  # Backoff before retrying
            continue

        if args.prompt_upsampler.debug:
            retry_dir = sample.args.output_dir / f"{i_retry}"
            retry_dir.mkdir(parents=True, exist_ok=True)
            _model_dump_json(response, retry_dir / "prompt_upsampler_response.json")

        assert len(response.choices) == 1
        choice = response.choices[0]
        if choice.finish_reason != "stop" or not choice.message.content:
            log.warning(f"{msg_prefix} Invalid response: {choice.finish_reason}")
            continue

        # Extract final prompt
        text = choice.message.content.strip()
        text, info = clean_response(text)
        text = text.removeprefix("```json\n").removesuffix("```")
        try:
            prompt_json = json.loads(text)
        except json.JSONDecodeError as e:
            log.warning(f"{msg_prefix} Invalid JSON response: {e}")
            continue
        if not isinstance(prompt_json, dict):
            log.warning(f"{msg_prefix} Invalid JSON type: {type(prompt_json)}")
            continue
        if not prompt_json.get("scene_imagination"):
            log.warning(f"{msg_prefix} Empty JSON response")
            continue
        prompt = json.dumps(prompt_json)

        sample_overrides = sample.overrides.model_copy(
            update={
                "prompt": prompt,
                "prompt_path": None,
            }
        )
        _model_dump_json(sample_overrides, Path(f"{sample.args.output_dir}.json"), exclude_none=True)
        return
    log.warning(f"['{sample.args.name}'] Failed to get response")


async def process_sample(
    args: Args,
    client: openai.AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    sample: Sample,
):
    async with semaphore:
        return await _process_sample(args, client, sample)


async def upsample_prompts(args: Args):
    setup_args = args.setup.build_setup()
    sample_overrides_list = OmniSampleOverrides.from_files(args.input_files, overrides=setup_args.sample_overrides)
    if not sample_overrides_list:
        raise ValueError(f"No samples found for {args.input_files}")
    log.info(f"Loaded {len(sample_overrides_list)} samples")

    model_config: "OmniMoTModelConfig" = setup_args.load_model_config().config

    # Build samples
    samples: list[Sample] = []
    for sample_overrides in sample_overrides_list:
        assert sample_overrides.name
        raw_sample_overrides = sample_overrides.model_copy(deep=True)
        sample_overrides.output_dir = setup_args.output_dir / sample_overrides.name
        if sample_overrides.sample_meta.model_mode not in _TASKS:
            log.info(f"Skipping '{sample_overrides.name}'")
            _model_dump_json(raw_sample_overrides, Path(f"{sample_overrides.output_dir}.json"), exclude_none=True)
            continue
        sample_overrides.download(sample_overrides.output_dir / "inputs")
        sample_args = sample_overrides.build_sample(model_config=model_config)
        is_video = sample_args.num_frames > 1
        messages = build_messages(
            task=_TASKS[sample_args.model_mode],
            description=sample_args.prompt,
            aspect_ratio=str(sample_args.aspect_ratio),
            resolution_w=sample_args.vision_size[0],
            resolution_h=sample_args.vision_size[1],
            fps=sample_args.fps if is_video else None,
            duration_secs=math.ceil(sample_args.duration) if is_video else None,
        )
        assert len(messages) == 2 and messages[1]["role"] == "user"
        user_message = messages[1]
        user_content = [
            {"type": "text", "text": user_message.pop("content")},
        ]
        if sample_args.vision_path:
            vision_url = str(sample_args.vision_path)
            if "://" not in vision_url:
                vision_url = _base64_encode(sample_args.vision_path)
            user_content.insert(0, {"type": "image_url", "image_url": {"url": vision_url}})
        user_message["content"] = user_content
        sample = Sample(args=sample_args, overrides=raw_sample_overrides, messages=messages)
        if args.prompt_upsampler.debug:
            _model_dump_json(sample.args, sample.args.output_dir / "sample_args.json")
            _dump_json(sample.messages, sample.args.output_dir / "prompt_upsampler_messages.json")
        else:
            shutil.rmtree(sample.args.output_dir, ignore_errors=True)
        samples.append(sample)

    client = openai.AsyncOpenAI(
        api_key="EMPTY",
        base_url=args.prompt_upsampler.endpoint_url,
        timeout=3600,
    )
    if not args.prompt_upsampler.model:
        models = await client.models.list()
        args.prompt_upsampler.model = models.data[0].id
        log.info(f"Using model: {args.prompt_upsampler.model}")

    # Process samples
    semaphore = asyncio.Semaphore(args.prompt_upsampler.max_workers)
    tasks = [
        process_sample(
            args=args,
            client=client,
            semaphore=semaphore,
            sample=sample,
        )
        for sample in samples
    ]
    for result in tqdm(asyncio.as_completed(tasks), desc="Upsampling", total=len(samples)):
        await result


def _base64_encode(path: Path) -> str:
    mime_type = mimetypes.guess_type(str(path))[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def main():
    args = tyro.cli(Args, description=__doc__)
    asyncio.run(upsample_prompts(args))


if __name__ == "__main__":
    main()
