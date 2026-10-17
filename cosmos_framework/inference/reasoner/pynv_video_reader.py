# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Deterministic Qwen video reader backed by NVDEC/PyNvVideoCodec."""

from __future__ import annotations

import atexit
import ctypes
import json
import os
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

from cosmos_framework.inference.reasoner.video_pixel_bounds import normalize_video_pixel_bounds


def _is_pynv_exception(error: BaseException) -> bool:
    """Return whether an exception originated in PyNvVideoCodec native code."""
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if any("pynvvc" in cls.__name__.lower() for cls in type(current).__mro__):
            return True
        current = current.__cause__ or current.__context__
    return False


def _is_nvdec_capability_error(error: BaseException) -> bool:
    """Return whether an exception reports a permanent GPU-reader limitation.

    PyNvVideoCodec exposes unsupported stream/GPU combinations through a
    dedicated exception type.  Its FFmpeg demuxer also reports a stable seek
    error when a stream has no usable random-access index; uniform frame
    sampling cannot proceed through PyNv on such a stream.  Keep narrow message
    checks for wrapper releases that preserve those native errors while
    translating the Python exception.  Do not classify ordinary decode, I/O,
    or programming errors as capability misses: those must keep failing the
    strict fast profile.
    """
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        names = {cls.__name__ for cls in type(current).__mro__}
        message = str(current).lower()
        pynv_error = any("pynvvc" in name.lower() for name in names)
        if "PyNvVCExceptionUnsupported" in names:
            return True
        if pynv_error and (
            "error code : 801" in message
            or "not supported on this gpu" in message
            or "unsupported" in message
            or ("seek target index is out of range" in message and "no matching index entry" in message)
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


def register_pynv_video_reader(
    *,
    cache_size: int = 0,
    decoder_cache_size: int = 4,
    video_override_map: str | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    """Register a Qwen NVDEC reader and optionally forbid CPU fallback."""
    if cache_size < 0:
        raise ValueError("cache_size must be non-negative")
    if decoder_cache_size < 1:
        raise ValueError("decoder_cache_size must be positive")
    # DataLoader workers use the spawn start method, so parent-process monkey
    # patches are not inherited.  Export the complete registration contract so
    # the baked qwen-vl-utils worker hook can recreate this reader in every
    # spawned worker before its first decode.
    os.environ["FORCE_QWENVL_VIDEO_READER"] = "pynvvideocodec"
    os.environ["COSMOS_PYNV_VIDEO_STRICT"] = "1" if strict else "0"
    os.environ["COSMOS_PYNV_VIDEO_CACHE_SIZE"] = str(cache_size)
    os.environ["COSMOS_PYNV_DECODER_CACHE_SIZE"] = str(decoder_cache_size)
    frame_transfer = os.environ.get("COSMOS_PYNV_FRAME_TRANSFER", "host_rgb")
    if frame_transfer not in {"host_rgb", "device_rgbp"}:
        raise ValueError("COSMOS_PYNV_FRAME_TRANSFER must be host_rgb or device_rgbp")
    os.environ["COSMOS_PYNV_FRAME_TRANSFER"] = frame_transfer
    try:
        ctypes.CDLL("libnvcuvid.so.1")
    except OSError as exc:
        raise RuntimeError("GPU video decoding requires readable libnvcuvid.so.1") from exc
    try:
        import numpy as np
        import PyNvVideoCodec as nvc
        import qwen_vl_utils.vision_process as vision_process
        import torch
        from cuda.bindings import driver as cuda_driver
    except ImportError as exc:
        raise RuntimeError("GPU video decoding requires PyNvVideoCodec, cuda.bindings, and qwen-vl-utils") from exc
    from cosmos_framework.inference.reasoner.system_pyav_video_reader import (
        _assert_software_video_decoders,
        read_video_system_pyav,
    )

    # The fast profile remains NVDEC/device-RGBP first.  Prove once at process
    # setup that its capability-only safety path is the checksum-pinned sparse
    # software reader, rather than Qwen's broad implicit fallback.
    software_codecs = _assert_software_video_decoders()

    overrides: dict[str, str] = {}
    if video_override_map:
        override_path = Path(video_override_map).expanduser().resolve(strict=True)
        value = json.loads(override_path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or not all(
            isinstance(key, str) and isinstance(item, str) for key, item in value.items()
        ):
            raise ValueError("video_override_map must be a JSON object of string paths")
        overrides = value
        os.environ["COSMOS_PYNV_VIDEO_OVERRIDE_MAP"] = str(override_path)
    else:
        os.environ.pop("COSMOS_PYNV_VIDEO_OVERRIDE_MAP", None)
    processed_cache: OrderedDict[tuple[Any, ...], Any] = OrderedDict()
    processed_inflight: dict[tuple[Any, ...], threading.Event] = {}
    cache_lock = threading.RLock()
    cache_stats = {"hits": 0, "misses": 0, "evictions": 0}
    cache_hit_attested = False
    decoder_lock = threading.RLock()
    cuda_state: dict[str, Any] = {}
    decode_attested = False
    fallback_attested = False
    software_only_paths: set[str] = set()

    def release_decoder(decoder) -> None:
        """Best-effort release across PyNvVideoCodec wrapper variants."""
        try:
            decoder.stop()
        except AttributeError:
            # PyNvVideoCodec 2.2.0's Python wrapper exposes ``stop`` even when
            # the selected native SimpleDecoder implementation does not.
            pass

    def cuda_check(result, operation):
        error, *values = result
        if int(error) != 0:
            raise RuntimeError(f"{operation} failed: {error}")
        return values[0] if len(values) == 1 else tuple(values)

    def decoder_gpu_ordinal() -> int:
        """Return the process-visible GPU ordinal for this policy rank."""
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        visible = [token.strip() for token in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if token.strip()]
        # A launcher may expose one physical GPU per policy process. CUDA then
        # remaps that device to process-local ordinal zero even when LOCAL_RANK
        # retains the node-global rank.
        return 0 if len(visible) == 1 else local_rank

    def cleanup_cuda_state() -> None:
        decoder = cuda_state.pop("decoder", None)
        cuda_state.pop("decoder_path", None)
        stream = cuda_state.pop("stream", None)
        device = cuda_state.pop("device", None)
        cuda_state.pop("context", None)
        if decoder is not None:
            release_decoder(decoder)
            del decoder
        if stream is not None:
            cuda_driver.cuStreamDestroy(stream)
        if device is not None:
            cuda_driver.cuCtxSetCurrent(None)
            cuda_driver.cuDevicePrimaryCtxRelease(device)

    def get_cuda_state() -> tuple[int, int, int]:
        """Retain one primary context and stream per spawned data worker."""
        if cuda_state:
            return (
                int(cuda_state["context"]),
                int(cuda_state["stream"]),
                int(cuda_state["gpu_id"]),
            )
        cuda_check(cuda_driver.cuInit(0), "cuInit")
        gpu_id = decoder_gpu_ordinal()
        device = cuda_check(cuda_driver.cuDeviceGet(gpu_id), "cuDeviceGet")
        context = cuda_check(
            cuda_driver.cuDevicePrimaryCtxRetain(device),
            "cuDevicePrimaryCtxRetain",
        )
        cuda_check(cuda_driver.cuCtxSetCurrent(context), "cuCtxSetCurrent")
        stream = cuda_check(cuda_driver.cuStreamCreate(0), "cuStreamCreate")
        cuda_state.update(
            device=device,
            context=context,
            stream=stream,
            gpu_id=gpu_id,
        )
        atexit.register(cleanup_cuda_state)
        return int(context), int(stream), gpu_id

    def make_decoder(
        video_path: str,
        cuda_context: int,
        cuda_stream: int,
        gpu_id: int,
        *,
        scan_stream_metadata: bool = False,
        session_cache_size: int | None = None,
    ):
        device_rgbp = frame_transfer == "device_rgbp"
        return nvc.SimpleDecoder(
            video_path,
            gpu_id=gpu_id,
            cuda_context=cuda_context,
            cuda_stream=cuda_stream,
            use_device_memory=device_rgbp,
            # Avoid the scan on ordinary inputs. If a native batch omission
            # proves that container metadata is not a decodable-frame count,
            # the recovery path below creates a scanned decoder and recomputes
            # the same uniform sampling policy from that authoritative count.
            need_scanned_stream_metadata=scan_stream_metadata,
            decoder_cache_size=(decoder_cache_size if session_cache_size is None else session_cache_size),
            output_color_type=(nvc.OutputColorType.RGBP if device_rgbp else nvc.OutputColorType.RGB),
        )

    def copy_batch_frames(frames, video_path: str, gpu_id: int):
        """Materialize one decoded batch without per-frame temporary copies."""
        if frame_transfer == "device_rgbp":
            tensors = [torch.from_dlpack(frame) for frame in frames]
            if any(tensor.device.type != "cuda" for tensor in tensors):
                raise RuntimeError(f"device RGBP decode returned a non-CUDA frame for {video_path}")
            if any(tensor.ndim != 3 or int(tensor.shape[0]) != 3 for tensor in tensors):
                raise RuntimeError(
                    "unexpected device RGBP frame layout "
                    f"{[tuple(tensor.shape) for tensor in tensors]} for {video_path}"
                )
            # DLPack shares each decoded CUDA surface without a copy. Stack
            # once in planar TCHW form, then issue one blocking transfer to
            # the CPU tensor expected by qwen-vl-utils preprocessing.
            video = torch.stack(tensors).cpu()
            torch.cuda.synchronize(gpu_id)
            return video

        video = None
        video_array = None
        for frame_index, frame in enumerate(frames):
            shape = tuple(int(value) for value in frame.shape)
            strides = tuple(int(value) for value in frame.strides)
            if len(shape) != 3 or shape[2] != 3:
                raise RuntimeError(f"unexpected RGB frame layout {shape} for {video_path}")
            if video is None:
                video = torch.empty((len(frames), 3, shape[0], shape[1]), dtype=torch.uint8)
                video_array = video.numpy()
            elif tuple(video.shape[2:]) != shape[:2]:
                raise RuntimeError(f"inconsistent RGB frame layout {shape} for {video_path}")
            raw = np.ctypeslib.as_array(
                ctypes.cast(frame.GetPtrToPlane(0), ctypes.POINTER(ctypes.c_uint8)),
                shape=(int(frame.framesize()),),
            )
            source = np.ndarray(shape=shape, dtype=np.uint8, buffer=raw, strides=strides)
            np.copyto(video_array[frame_index], source.transpose(2, 0, 1))
        if video is None:
            raise RuntimeError(f"decoder returned no frames for {video_path}")
        return video

    def discard_active_decoder() -> None:
        decoder = cuda_state.pop("decoder", None)
        cuda_state.pop("decoder_path", None)
        if decoder is not None:
            release_decoder(decoder)
            del decoder

    def read_video_capability_fallback(
        element: dict[str, Any],
        video_path: str,
        error: BaseException | None,
    ):
        """Decode one NVDEC-incompatible stream without changing profiles."""
        nonlocal fallback_attested
        software_only_paths.add(video_path)
        software_element = dict(element)
        software_element["video"] = video_path
        if not fallback_attested:
            print(
                "COSMOS_VIDEO_DECODER_CAPABILITY_FALLBACK_ATTESTATION "
                "primary=pynvvideocodec "
                "fallback=cosmos_system_pyav_sparse "
                f"software_codecs={software_codecs} "
                "scope=nvdec_unsupported_stream_only",
                flush=True,
            )
            fallback_attested = True
        print(
            "COSMOS_VIDEO_DECODER_CAPABILITY_FALLBACK "
            f"primary=pynvvideocodec video={video_path} "
            "fallback=cosmos_system_pyav_sparse "
            f"reason={type(error).__name__ if error is not None else 'cached_capability'}",
            flush=True,
        )
        return read_video_system_pyav(software_element)

    def read_video_pynv(element):
        nonlocal decode_attested
        with decoder_lock:
            video_path = str(element["video"])
            if video_path.startswith("file://"):
                video_path = video_path[7:]
            if video_path.startswith(("http://", "https://")):
                raise ValueError("PyNvVideoCodec evaluation requires compute-node-local media")
            video_path = overrides.get(video_path, video_path)
            video_path = str(Path(video_path).expanduser().resolve(strict=True))
            try:
                if video_path in software_only_paths:
                    return read_video_capability_fallback(element, video_path, None)
                cuda_context, cuda_stream, gpu_id = get_cuda_state()
                cuda_check(
                    cuda_driver.cuCtxSetCurrent(cuda_state["context"]),
                    "cuCtxSetCurrent",
                )
                decoder = cuda_state.get("decoder")
                if decoder is not None and cuda_state["decoder_path"] != video_path:
                    if decoder_cache_size > 1:
                        # The explicit fast profile keeps PyNvVideoCodec's native
                        # decoder/session cache alive across repeated media. Any
                        # omission or stale metadata is still caught by the exact
                        # batch-size gate and recovered through the scanned path.
                        try:
                            decoder.reconfigure_decoder(video_path)
                            cuda_state["decoder_path"] = video_path
                        except Exception:
                            discard_active_decoder()
                            decoder = None
                    else:
                        # The conservative default isolates decoder sessions per
                        # source while retaining the rank-local CUDA context.
                        discard_active_decoder()
                        decoder = None
                if decoder is None:
                    decoder = make_decoder(video_path, cuda_context, cuda_stream, gpu_id)
                    cuda_state["decoder"] = decoder
                    cuda_state["decoder_path"] = video_path

                metadata = decoder.get_stream_metadata()
                source_total_frames = len(decoder)
                video_fps = float(metadata.average_fps)
                start_frame, end_frame, selected_total_frames = vision_process.calculate_video_frame_range(
                    element, source_total_frames, video_fps
                )
                nframes = vision_process.smart_nframes(
                    element,
                    total_frames=selected_total_frames,
                    video_fps=video_fps,
                )
                indices = torch.linspace(start_frame, end_frame, nframes).round().long().tolist()
                try:
                    try:
                        batch_frames = decoder.get_batch_frames_by_index(indices)
                    except Exception as batch_error:
                        # At this boundary, a native PyNv exception means the
                        # stream cannot satisfy random-access batch retrieval.
                        # Route only that stream through the existing sparse
                        # software reader. Other exception types retain the
                        # scanned NVDEC recovery path below.
                        if _is_pynv_exception(batch_error):
                            discard_active_decoder()
                            return read_video_capability_fallback(element, video_path, batch_error)
                        raise
                    if len(batch_frames) != len(indices):
                        raise RuntimeError(f"NVDEC batch returned {len(batch_frames)} of {len(indices)} frames")
                    video = copy_batch_frames(batch_frames, video_path, gpu_id)
                except Exception as batch_error:
                    if _is_nvdec_capability_error(batch_error):
                        raise
                    # Some MP4 container metadata overstates the number of
                    # decodable frames. PyNvVideoCodec 2.2.0 then omits the
                    # out-of-range frame from a batch, and direct access to that
                    # index can segfault in native code. Rescan only on this
                    # proven mismatch, recompute the same uniform-frame policy
                    # from the authoritative decodable count, and retry through
                    # NVDEC. Never clamp an index or route through Qwen's broad
                    # implicit fallback.
                    unscanned_total_frames = source_total_frames
                    unscanned_indices = indices
                    discard_active_decoder()
                    print(
                        "COSMOS_GPU_VIDEO_BATCH_RETRY "
                        f"backend=pynvvideocodec video={video_path} "
                        f"indices={unscanned_indices} "
                        f"unscanned_total_frames={unscanned_total_frames} "
                        f"error={type(batch_error).__name__}",
                        flush=True,
                    )
                    video = None
                    last_error = batch_error
                    for _attempt in range(3):
                        retry_decoder = make_decoder(
                            video_path,
                            cuda_context,
                            cuda_stream,
                            gpu_id,
                            scan_stream_metadata=True,
                            session_cache_size=1,
                        )
                        try:
                            metadata = retry_decoder.get_stream_metadata()
                            source_total_frames = len(retry_decoder)
                            video_fps = float(metadata.average_fps)
                            start_frame, end_frame, selected_total_frames = vision_process.calculate_video_frame_range(
                                element, source_total_frames, video_fps
                            )
                            nframes = vision_process.smart_nframes(
                                element,
                                total_frames=selected_total_frames,
                                video_fps=video_fps,
                            )
                            indices = torch.linspace(start_frame, end_frame, nframes).round().long().tolist()
                            batch_frames = retry_decoder.get_batch_frames_by_index(indices)
                            if len(batch_frames) != len(indices):
                                raise RuntimeError(
                                    f"scanned NVDEC batch returned {len(batch_frames)} of {len(indices)} frames"
                                )
                            video = copy_batch_frames(batch_frames, video_path, gpu_id)
                            cuda_state["decoder"] = retry_decoder
                            cuda_state["decoder_path"] = video_path
                            last_error = None
                            print(
                                "COSMOS_GPU_VIDEO_BATCH_RECOVERED "
                                f"backend=pynvvideocodec video={video_path} "
                                f"unscanned_total_frames={unscanned_total_frames} "
                                f"scanned_total_frames={source_total_frames} "
                                f"indices={indices}",
                                flush=True,
                            )
                            break
                        except Exception as exc:
                            last_error = exc
                            release_decoder(retry_decoder)
                            del retry_decoder
                            if _is_nvdec_capability_error(exc):
                                break
                    if last_error is not None or video is None:
                        raise RuntimeError(
                            f"GPU-only PyNvVideoCodec scanned retry failed for {video_path} indices {indices}"
                        ) from last_error
                sample_fps = nframes / max(selected_total_frames, 1.0) * video_fps
                result = (
                    video,
                    {
                        "fps": video_fps,
                        "frames_indices": indices,
                        "total_num_frames": selected_total_frames,
                        "video_backend": "pynvvideocodec",
                    },
                    sample_fps,
                )
                if not decode_attested:
                    worker_info = torch.utils.data.get_worker_info()
                    worker_id = worker_info.id if worker_info is not None else -1
                    print(
                        "COSMOS_GPU_VIDEO_DECODE_ATTESTATION "
                        f"backend=pynvvideocodec pid={os.getpid()} "
                        f"local_rank={os.environ.get('LOCAL_RANK', '0')} "
                        f"worker_id={worker_id} gpu_ordinal={gpu_id} "
                        f"decoded_frames={len(indices)} "
                        f"frame_transfer={frame_transfer} "
                        f"decoder_cache_size={decoder_cache_size}",
                        flush=True,
                    )
                    decode_attested = True
                return result
            except Exception as error:
                if not _is_nvdec_capability_error(error):
                    raise
                discard_active_decoder()
                return read_video_capability_fallback(element, video_path, error)

    vision_process.VIDEO_READER_BACKENDS["pynvvideocodec"] = read_video_pynv

    original_fetch_video = getattr(
        vision_process,
        "_cosmos_pynv_original_fetch_video",
        vision_process.fetch_video,
    )
    vision_process._cosmos_pynv_original_fetch_video = original_fetch_video

    def fetch_video_pynv_cached(
        element,
        image_patch_size=14,
        return_video_sample_fps=False,
        return_video_metadata=False,
    ):
        """Cache the fully resized Qwen result, not full-resolution RGB frames."""
        nonlocal cache_hit_attested
        element = normalize_video_pixel_bounds(element, image_patch_size, vision_process)
        key = tuple(
            element.get(name)
            for name in (
                "video",
                "video_start",
                "video_end",
                "nframes",
                "fps",
                "min_frames",
                "max_frames",
                "min_pixels",
                "max_pixels",
                "total_pixels",
                "resized_height",
                "resized_width",
            )
        ) + (
            image_patch_size,
            return_video_sample_fps,
            return_video_metadata,
        )
        while True:
            with cache_lock:
                cached = processed_cache.get(key)
                if cached is not None:
                    cache_stats["hits"] += 1
                    processed_cache.move_to_end(key)
                    if not cache_hit_attested:
                        worker_info = torch.utils.data.get_worker_info()
                        worker_id = worker_info.id if worker_info is not None else -1
                        print(
                            "COSMOS_PYNV_VIDEO_CACHE_HIT_ATTESTATION "
                            f"pid={os.getpid()} "
                            f"local_rank={os.environ.get('LOCAL_RANK', '0')} "
                            f"worker_id={worker_id} "
                            f"capacity={cache_size} "
                            f"entries={len(processed_cache)} "
                            "cache_boundary=processed_fetch_video",
                            flush=True,
                        )
                        cache_hit_attested = True
                    return cached
                event = processed_inflight.get(key)
                if event is None:
                    event = threading.Event()
                    processed_inflight[key] = event
                    cache_stats["misses"] += 1
                    owner = True
                else:
                    owner = False
            if owner:
                break
            event.wait()

        try:
            result = original_fetch_video(
                element,
                image_patch_size=image_patch_size,
                return_video_sample_fps=return_video_sample_fps,
                return_video_metadata=return_video_metadata,
            )
            with cache_lock:
                if cache_size:
                    processed_cache[key] = result
                    processed_cache.move_to_end(key)
                    while len(processed_cache) > cache_size:
                        processed_cache.popitem(last=False)
                        cache_stats["evictions"] += 1
            return result
        finally:
            with cache_lock:
                completed = processed_inflight.pop(key, None)
                if completed is not None:
                    completed.set()

    vision_process.fetch_video = fetch_video_pynv_cached
    vision_process._cosmos_pynv_processed_cache_stats = cache_stats
    if strict:

        def reject_cpu_fallback(_element):
            raise RuntimeError("CPU video decoding fallback is disabled for this regression")

        vision_process.VIDEO_READER_BACKENDS["torchvision"] = reject_cpu_fallback
    vision_process.FORCE_QWENVL_VIDEO_READER = "pynvvideocodec"
    vision_process.get_video_reader_backend.cache_clear()
    return {
        "backend": "pynvvideocodec",
        "version": getattr(nvc, "__version__", "unknown"),
        "cache_size": cache_size,
        "decoder_cache_size": decoder_cache_size,
        "frame_transfer": frame_transfer,
        "cache_boundary": "processed_fetch_video",
        "capability_fallback": "cosmos_system_pyav_sparse",
        "capability_fallback_scope": "nvdec_unsupported_stream_only",
        "video_overrides": len(overrides),
        "strict": strict,
    }
