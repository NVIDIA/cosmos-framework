# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Rewrite a dataset's inference-prompt JSON files to use structured-JSON prompts.

The example dataset ships per-clip inference prompts under
``<val>/inference_prompt{,_i2v,_v2v}/<episode>.json`` whose ``prompt`` field is a
**dense** narrative.  Once the dataset carries structured-JSON captions, the
inference example should use the **same** format so it matches what the model is
trained on.  This script replaces each file's ``prompt`` with the serialized
structured caption (from the clip's ``caption.json``), preserving every other
field (``name``, ``resolution``, ``aspect_ratio``, ``num_frames``, ``fps``,
``vision_path``).  It is idempotent and re-runnable.

Usage
-----
    python -m cosmos_framework.scripts.inference_prompts_to_json \
        --val-dir /path/to/sft_dataset_bridge/val

    # captions live elsewhere, or only update specific variants:
    python -m cosmos_framework.scripts.inference_prompts_to_json \
        --val-dir /path/to/val --captions-dir /path/to/val/captions --dry-run
"""

import json
import sys
from pathlib import Path
from typing import Annotated

import tyro

from cosmos_framework.inference.structured_caption import caption_json_to_prompt


def main(
    val_dir: Annotated[Path, tyro.conf.arg(help="Dataset split dir containing inference_prompt*/ and captions/.")],
    captions_dir: Annotated[
        Path | None, tyro.conf.arg(help="Dir with <episode>/caption.json (default: <val-dir>/captions).")
    ] = None,
    inference_prompt_glob: str = "inference_prompt*",
    dry_run: bool = False,
) -> None:
    """Replace dense `prompt` fields with the serialized structured JSON caption."""
    captions_dir = captions_dir or (val_dir / "captions")
    prompt_dirs = sorted(d for d in val_dir.glob(inference_prompt_glob) if d.is_dir())
    if not prompt_dirs:
        print(f"No '{inference_prompt_glob}' directories found under {val_dir}", file=sys.stderr)
        sys.exit(1)

    n_updated = 0
    n_missing_caption = 0
    n_files = 0

    for prompt_dir in prompt_dirs:
        for prompt_path in sorted(prompt_dir.glob("*.json")):
            n_files += 1
            episode = prompt_path.stem
            caption_json_path = captions_dir / episode / "caption.json"
            if not caption_json_path.exists():
                print(f"  MISS {prompt_dir.name}/{episode}: no caption.json at {caption_json_path}")
                n_missing_caption += 1
                continue

            try:
                caption_json = json.loads(caption_json_path.read_text())
            except json.JSONDecodeError as e:
                print(f"  MISS {prompt_dir.name}/{episode}: caption.json invalid ({e})")
                n_missing_caption += 1
                continue

            record = json.loads(prompt_path.read_text())
            record["prompt"] = caption_json_to_prompt(caption_json)

            if dry_run:
                print(f"  DRY  {prompt_dir.name}/{episode}: would set prompt ({len(record['prompt'])} chars)")
            else:
                prompt_path.write_text(json.dumps(record, indent=4, ensure_ascii=False))
                print(f"  OK   {prompt_dir.name}/{episode}")
            n_updated += 1

    print(
        f"\n{'Would update' if dry_run else 'Updated'} {n_updated}/{n_files} prompt files "
        f"across {len(prompt_dirs)} dir(s); {n_missing_caption} missing caption.json"
    )
    if n_updated == 0:
        print("ERROR: No prompt files updated.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    tyro.cli(main)
