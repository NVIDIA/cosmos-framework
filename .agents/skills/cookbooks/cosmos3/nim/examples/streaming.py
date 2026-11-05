# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Process a video through Reasoner streaming sessions over WebSocket or REST.

Adapted from cosmos-genai-nim/cosmos3/examples/streaming_ws.py.
Requires NIM_ENABLE_STREAMING=true on the Reasoner.
"""

import argparse
import asyncio
import base64
import io
import json
import math
import os
import sys
from pathlib import Path

import requests

from common import require_streaming

NIM_URL = os.environ.get("NIM_URL", "http://localhost:8000").rstrip("/")


ASSETS = Path(__file__).resolve().parents[2] / "reasoner" / "assets"
DEFAULT_VIDEO = ASSETS / "video_caption.mp4"

SYSTEM_PROMPT = (
    "You are monitoring a live video feed. For each frame, report in one short "
    "sentence what is happening and flag anything hazardous."
)


def _ws_url(nim_url: str) -> str:
    """Turn the NIM's HTTP base URL into the streaming WebSocket URL."""
    scheme, _, rest = nim_url.rstrip("/").partition("://")
    if scheme not in {"http", "https"} or not rest:
        raise ValueError("NIM_URL must be an HTTP(S) base URL")
    ws_scheme = "wss" if scheme == "https" else "ws"
    return f"{ws_scheme}://{rest}/v1/streaming/ws"


def iter_frames(video: Path, fps: float, max_frames: int, loop: bool = False):
    """Yield JPEG-encoded frames sampled from ``video`` at ``fps``.

    With ``loop``, the clip repeats until ``max_frames`` is reached, so a short
    sample can drive a session far longer than its own length -- which is how
    you exercise sustained retention/eviction rather than a single window.
    """
    try:
        import av
    except ImportError as exc:  # pragma: no cover - depends on the host env
        raise SystemExit(
            "Initialize the cookbook environment with: uv sync --locked"
        ) from exc

    emitted = 0
    while True:
        before = emitted
        with av.open(str(video)) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            source_fps = float(stream.average_rate or 30.0)
            step = max(1, round(source_fps / fps))
            for index, frame in enumerate(container.decode(stream)):
                if index % step:
                    continue
                buf = io.BytesIO()
                frame.to_image().save(buf, format="JPEG", quality=85)
                yield buf.getvalue()
                emitted += 1
                if emitted >= max_frames:
                    return
        if emitted == before:
            raise ValueError(f"No frames decoded from {video}")
        if not loop:
            return


def session_request(args: argparse.Namespace, model: str) -> dict:
    request = {
        "model": model,
        "system_prompt": SYSTEM_PROMPT,
        "question": args.question,
        "fps": args.fps,
        "sampling": {"max_tokens": args.max_tokens},
    }
    if args.max_video_segments is not None:
        request["retention"] = {"max_video_segments": args.max_video_segments}
    return request


def print_frame(reply: dict, quiet: bool) -> None:
    latency = reply.get("latency_s")
    suffix = f"  ({latency:.2f}s)" if latency is not None else ""
    if not quiet or reply["frame_index"] % 10 == 0:
        print(f"frame {reply['frame_index']:>4}: {reply['text']}{suffix}")


async def run_websocket(args: argparse.Namespace, create: dict) -> int:
    try:
        import websockets
    except ImportError as exc:
        raise SystemExit(
            "Initialize the cookbook environment with: uv sync --locked"
        ) from exc

    url = _ws_url(args.nim_url)
    print(f"connecting to {url}")

    # max_size bounds inbound frames only; our outbound frames can be large.
    async with websockets.connect(url, max_size=None, open_timeout=30) as ws:
        create = {**create, "op": "create", "frame_format": args.frame_format}
        await ws.send(json.dumps(create))
        created = json.loads(await ws.recv())
        if created.get("op") != "created":
            print(f"session create failed: {created}", file=sys.stderr)
            return 1
        print(f"session {created['session_id']} on {created['model']}\n")

        frames = 0
        for payload in iter_frames(args.video, args.fps, args.max_frames, args.loop):
            if args.frame_format == "binary":
                await ws.send(payload)
            else:
                await ws.send(
                    json.dumps(
                        {
                            "op": "frame",
                            "image_b64": base64.b64encode(payload).decode(),
                        }
                    )
                )

            reply = json.loads(await ws.recv())
            if reply.get("op") == "error":
                print(f"[error {reply['code']}] {reply['message']}", file=sys.stderr)
                if reply.get("fatal"):
                    return 1
                continue

            frames += 1
            print_frame(reply, args.quiet)

        await ws.send(json.dumps({"op": "delete"}))
        closed = json.loads(await ws.recv())
        print(f"\nclosed after {closed.get('frames', frames)} frames")
    return 0


def rest_result(response: requests.Response) -> dict:
    if not response.ok:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text}")
    return response.json()


def run_rest(args: argparse.Namespace, create: dict) -> int:
    url = f"{args.nim_url.rstrip('/')}/v1/streaming/sessions"
    with requests.Session() as client:
        created = rest_result(client.post(url, json=create, timeout=30))
        session_url = f"{url}/{created['session_id']}"
        cleanup_failed = False
        try:
            print(f"session {created['session_id']} on {created['model']}\n")
            for payload in iter_frames(args.video, args.fps, args.max_frames, args.loop):
                # JSON/base64 works through both standalone NIM and Dynamo.
                response = client.post(
                    f"{session_url}/frame",
                    json={"image_b64": base64.b64encode(payload).decode("ascii")},
                    timeout=1800,
                )
                print_frame(rest_result(response), args.quiet)
        finally:
            # HTTP disconnects don't release sessions. Attempt deletion even
            # after a failed frame, decoding error, or keyboard interrupt.
            try:
                response = client.delete(session_url, timeout=30)
                if response.status_code != 404:
                    closed = rest_result(response)
                    print(f"\nclosed after {closed['frames']} frames")
            except (requests.RequestException, RuntimeError, ValueError) as exc:
                cleanup_failed = True
                print(f"session cleanup failed: {exc}; idle timeout is the backstop", file=sys.stderr)
    return 1 if cleanup_failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nim-url", default=NIM_URL)
    parser.add_argument(
        "--backend", choices=("nim", "dynamo"), default="nim",
        help="Serving interface: NIM endpoint or shared Dynamo frontend (default: nim)",
    )
    parser.add_argument(
        "--transport", choices=("websocket", "rest"), default="websocket",
        help="Session API transport (default: websocket)",
    )
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int, default=30)
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Repeat the clip until --max-frames, to drive long sessions.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Print only every 10th caption (for long runs).",
    )
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument(
        "--max-video-segments", type=int,
        help="Override the server retention window (default: use server setting)",
    )
    parser.add_argument(
        "--frame-format", choices=("binary", "base64"),
        help="Frame encoding: WebSocket defaults to binary; this client uses base64 for REST",
    )
    parser.add_argument(
        "--question",
        default="What is happening in this feed?",
        help="Sent with the first frame only.",
    )
    args = parser.parse_args()

    if args.backend == "dynamo" and args.transport != "rest":
        parser.error("Dynamo supports REST streaming only; use --transport rest.")
    if args.transport == "rest" and args.frame_format == "binary":
        parser.error(
            "This client sends REST frames as JSON/base64. Use --frame-format base64."
        )
    if args.frame_format is None:
        args.frame_format = "base64" if args.transport == "rest" else "binary"

    if not math.isfinite(args.fps) or args.fps <= 0:
        parser.error("--fps must be finite and positive")
    if args.max_frames < 1 or args.max_tokens < 1:
        parser.error("--max-frames and --max-tokens must be positive")
    if args.max_video_segments is not None and args.max_video_segments < 1:
        parser.error("--max-video-segments must be positive")
    if not args.video.is_file():
        parser.error(f"video not found: {args.video}")
    try:
        _ws_url(args.nim_url)  # Both transports require an HTTP(S) base URL.
        config = require_streaming(args.nim_url, backend=args.backend)
        create = session_request(args, config["model"])
        if args.transport == "rest":
            return run_rest(args, create)
        return asyncio.run(run_websocket(args, create))
    except (requests.RequestException, RuntimeError, TypeError, ValueError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
