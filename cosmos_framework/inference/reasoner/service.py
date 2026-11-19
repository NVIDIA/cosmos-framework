# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Local reasoner HTTP service with an OpenAI-compatible chat endpoint.

No cloud callbacks or model-management service are required. Authentication and
TLS belong at the deployment ingress; bind to loopback unless that is configured.
"""

from __future__ import annotations

import base64
import binascii
import copy
import threading
import time
import uuid
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlparse

MEDIA_TYPES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "image/bmp": ".bmp",
    "image/tiff": ".tiff",
    "video/mp4": ".mp4",
    "video/webm": ".webm",
    "video/quicktime": ".mov",
    "video/x-matroska": ".mkv",
    "video/x-msvideo": ".avi",
}
MAX_MEDIA_BYTES = 64 * 1024 * 1024


def resolve_media(value, directory, allowed_roots, *, allow_remote=False):
    """Decode inline media, or resolve a file inside an explicit media mount."""
    if not isinstance(value, str):
        raise ValueError("Media URLs must be strings")
    if value.startswith("data:"):
        header, separator, encoded = value.partition(",")
        mime = header[5:].removesuffix(";base64")
        if not separator or not header.endswith(";base64") or mime not in MEDIA_TYPES:
            raise ValueError("Unsupported media data URI")
        if len(encoded) > 4 * ((MAX_MEDIA_BYTES + 2) // 3):
            raise ValueError("Inline media exceeds 64 MiB")
        try:
            data = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error) as error:
            raise ValueError("Invalid base64 media") from error
        if len(data) > MAX_MEDIA_BYTES:
            raise ValueError("Inline media exceeds 64 MiB")
        path = Path(directory) / (uuid.uuid4().hex + MEDIA_TYPES[mime])
        path.write_bytes(data)
        return str(path)
    scheme = urlparse(value).scheme
    if scheme in {"http", "https"} and allow_remote:
        return value
    if scheme:
        raise ValueError("Only inline media or mounted local files are enabled")
    path = Path(value).expanduser().resolve()
    if not any(path.is_relative_to(Path(root).resolve()) for root in allowed_roots):
        raise ValueError("Media path is outside the configured media roots")
    if not path.is_file() or path.suffix.lower() not in set(MEDIA_TYPES.values()):
        raise ValueError("Media file is missing or has an unsupported extension")
    return str(path)


def prepare_messages(messages, directory, allowed_roots=(), *, allow_remote=False):
    """Preserve conversation order and convert API media parts to processor parts."""
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")
    messages = copy.deepcopy(messages)
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in {"system", "user", "assistant"}:
            raise ValueError("Supported roles are system, user, assistant")
        content = message.get("content", "")
        if isinstance(content, str):
            continue
        if not isinstance(content, list):
            raise ValueError("Message content must be text or a list of content parts")
        for item in content:
            if not isinstance(item, dict):
                raise ValueError("Content parts must be objects")
            kind = item.get("type")
            if kind == "text" and isinstance(item.get("text"), str):
                continue
            if kind not in {"image_url", "video_url", "image", "video"}:
                raise ValueError(f"Unsupported content type: {kind}")
            value = item.get(kind)
            if isinstance(value, dict):
                value = value.get("url")
            media_kind = kind.removesuffix("_url")
            resolved = resolve_media(value, directory, allowed_roots, allow_remote=allow_remote)
            item.clear()
            item.update(type=media_kind, **{media_kind: resolved})
    return messages


def chat_task(messages):
    """Build a runtime task from prepared chat messages.

    Transformers backends consume ``prompt`` (the full conversation); the native
    Cosmos3 backend reads the last user turn from ``question`` and its media
    from ``media_paths``, so both views are filled in.
    """
    task = {"id": "0", "prompt": messages}
    user_turns = [message for message in messages if message["role"] == "user"]
    if not user_turns:
        raise ValueError("messages must contain a user message")
    content = user_turns[-1].get("content", "")
    parts = [{"type": "text", "text": content}] if isinstance(content, str) else content
    task["question"] = "\n".join(part["text"] for part in parts if part["type"] == "text")
    media = [(part["type"], part[part["type"]]) for part in parts if part["type"] in {"image", "video"}]
    if media:
        task["media_paths"] = [path for _kind, path in media]
        task["media_mode"] = media[0][0]
    return task


def create_app(runtime, model_name, *, allowed_roots=(), allow_remote=False):
    """Build an app around an injected runtime (also usable with a CPU test double)."""
    from fastapi import FastAPI, HTTPException

    app = FastAPI(title="Cosmos Reasoner")
    lock = threading.Lock()

    @app.get("/health")
    def health():
        return {"status": "healthy", "model_loaded": True}

    @app.get("/info")
    def info():
        return {"model": model_name, "endpoints": ["/infer", "/v1/chat/completions"]}

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": model_name, "object": "model", "owned_by": "cosmos"}]}

    def build_tasks(payload, directory, *, chat):
        if payload.get("stream"):
            raise ValueError("Streaming is not supported")
        if payload.get("model", model_name) != model_name:
            raise ValueError("Requested model is not loaded")
        generation = {key: payload[key] for key in ("temperature", "top_p", "top_k", "seed") if key in payload}
        generation["max_tokens"] = int(payload.get("max_tokens", payload.get("max_new_tokens", 4096)))
        if not 1 <= generation["max_tokens"] <= 32768:
            raise ValueError("max_tokens must be between 1 and 32768")
        vision = {
            key: payload[key]
            for key in ("fps", "num_frames", "min_pixels", "max_pixels", "total_pixels")
            if key in payload
        }
        if chat:
            messages = prepare_messages(payload.get("messages"), directory, allowed_roots, allow_remote=allow_remote)
            return [chat_task(messages)], generation, vision
        media = payload.get("media", [])
        media = [media] if isinstance(media, str) else media
        if not isinstance(media, list) or not media:
            raise ValueError("media must contain at least one file")
        question = payload.get("prompt", "Describe this media.")
        tasks = []
        for value in media:
            path = resolve_media(value, directory, allowed_roots, allow_remote=allow_remote)
            prompt = []
            if payload.get("system_prompt"):
                prompt.append({"role": "system", "content": payload["system_prompt"]})
            prompt.append({"role": "user", "content": question})
            kind = (
                "video"
                if Path(urlparse(path).path).suffix.lower() in {".mp4", ".webm", ".mov", ".mkv", ".avi"}
                else "image"
            )
            tasks.append(
                {
                    "id": str(len(tasks)),
                    "prompt": prompt,
                    "question": question,
                    "media_paths": [path],
                    "media_mode": kind,
                }
            )
        return tasks, generation, vision

    def generate(payload, *, chat):
        with TemporaryDirectory(prefix="cosmos-request-") as directory:
            # Only request validation maps to 400; runtime failures stay server errors.
            try:
                tasks, generation, vision = build_tasks(payload, directory, chat=chat)
            except (ValueError, TypeError) as error:
                raise HTTPException(status_code=400, detail=str(error)) from error
            # Model caches are mutable and must not be shared by concurrent calls.
            with lock:
                responses, _ = runtime.generate_tasks(tasks, generation_config=generation, vision_config=vision)
        if not chat:
            return {"response": responses[0] if len(responses) == 1 else responses, "total_files": len(tasks)}
        return {
            "id": "chatcmpl-" + uuid.uuid4().hex,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": responses[0]}, "finish_reason": "stop"}
            ],
        }

    @app.post("/v1/chat/completions")
    def chat_completions(payload: dict):
        return generate(payload, chat=True)

    @app.post("/infer")
    def infer(payload: dict):
        return generate(payload, chat=False)

    return app
