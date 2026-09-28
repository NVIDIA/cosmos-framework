# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serve a prepared reasoner checkpoint without an external action runtime."""

import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-name", help="Public API model identifier (defaults to model-path)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--media-root", action="append", default=[], help="Allowed local media mount; repeatable")
    parser.add_argument("--allow-remote-media", action="store_true", help="Allow HTTP media from trusted callers only")
    parser.add_argument("--dtype", default="bfloat16", choices=["auto", "float16", "float32", "bfloat16"])
    args = parser.parse_args()

    import uvicorn

    from cosmos_framework.inference.reasoner.runtime import CosmosFrameworkRuntime
    from cosmos_framework.inference.reasoner.service import create_app

    model_path = args.model_path.removeprefix("hf_model://")
    if "://" in model_path:
        parser.error("model-path must be local or a Hugging Face model ID")
    runtime = CosmosFrameworkRuntime(model_path, dtype=args.dtype)
    app = create_app(
        runtime, args.model_name or model_path, allowed_roots=args.media_root, allow_remote=args.allow_remote_media
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
