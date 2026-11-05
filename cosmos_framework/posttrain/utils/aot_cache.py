#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Share the AOT tokenizer cache across eval cells, without editing the tokenizer.

`Wan2pt2VAEInterface.compile_encode` compiles one ``.pt2`` package per (resolution,
aspect-ratio, cache_t) variant. By default the packages go to a PID-unique scratch dir that
is wiped on entry and deleted on exit, so every policy server recompiles all of them --
~143 s for the four 480p variants, paid once per cell. On a 48-cell fan-out that is 48
compiles of identical artifacts.

Pointing them at one mounted directory makes the first cell to reach a variant compile it and
the rest load it. This applies that as a **runtime patch** rather than a change to
``wan2pt2_vae_4x16x16.py``: that file is shared with training and generation, and a caching
policy that only the sim-eval fan-out wants does not belong in it.

Three seams, all reachable from outside:

1. ``compile_encode(output_dir=...)`` -- redirect it at the shared root, keyed so a stale hit
   cannot happen.
2. ``_synchronized_reset_dir`` -- module-level, looked up as a global by the method, so
   patching the module attribute suppresses the wipe-on-entry and delete-on-exit.
3. ``torch._inductor.aoti_compile_and_package`` -- resolved by attribute at call time, so
   wrapping it makes package publication atomic.

(3) is not optional once the directory is shared. ``_compile_variant`` treats
``os.path.exists(pkg_path)`` as proof a package is usable while the compiler writes to that
same path IN PLACE, so a second cell can load a half-written ``.pt2``. Per-process that is
impossible; across 48 cells starting together it is likely, and it only bites on a cold cache.

Reuse is otherwise already guarded: ``_compile_variant`` skips a variant whose package
exists, and ``_variant_matches_eager`` re-validates every loaded package against eager and
drops mismatches rather than trusting them.
"""

from __future__ import annotations

import functools
import os

_SHARED_DIR: str | None = None  # resolved cache dir once installed


def _cache_key() -> str:
    """Directory key covering everything a compiled package depends on.

    Shapes are already in the ``.pt2`` filenames, and ``package_constants_in_so=False`` keeps
    weights out of the package (they are bound at load), so neither needs to be in the key.
    What is left is the toolchain and the target: a package built by another torch build, CUDA
    runtime, or GPU architecture must not be picked up.
    """
    import torch

    cap = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
    return "-".join((f"torch{torch.__version__}", f"cu{torch.version.cuda or 'none'}", f"sm{cap[0]}{cap[1]}")).replace(
        "/", "_"
    )


def _atomic_compile_and_package(original):
    """Wrap ``aoti_compile_and_package`` so a package appears whole or not at all."""

    @functools.wraps(original)
    def _wrapped(*args, package_path: str, **kwargs):
        # The temp name MUST still end in .pt2: torch asserts
        # `os.fspath(package_path).endswith(".pt2")` (torch/_inductor/__init__.py). A suffix
        # appended after the extension makes every compile raise, _compile_variant swallows
        # it, and the run dies with the unhelpful "AOT compilation produced no loadable
        # functions" -- which is exactly what happened the first time.
        # Safe to keep the extension: the tokenizer never globs the directory, it tracks
        # packages by explicit path, so a temp .pt2 cannot be mistaken for a real one.
        tmp = f"{package_path[: -len('.pt2')]}.{os.getpid()}.tmp.pt2"
        try:
            result = original(*args, package_path=tmp, **kwargs)
            os.replace(tmp, package_path)  # atomic within a filesystem
            # Report the PUBLISHED path. torch returns the path it wrote, which here is the
            # temp name -- a name that no longer exists by the time this returns. The current
            # caller ignores the return value (``_compile_variant`` reuses its own pkg_path),
            # so this is latent rather than live, but a caller that loaded the returned path
            # would fail on a missing file and the cause would not be obvious.
            return package_path if result == tmp else result
        finally:
            # A failed compile must not leave its partial file behind: nothing sweeps a
            # persistent cache dir, so they would accumulate run after run.
            if os.path.exists(tmp):
                try:
                    os.unlink(tmp)
                except OSError:
                    pass

    return _wrapped


def enable_shared_aot_cache() -> str | None:
    """Redirect tokenizer AOT packages to ``$COSMOS_AOT_TOKENIZER_CACHE_DIR``.

    No-op when the variable is unset, so the stock per-process behaviour is the default and
    nothing changes for callers that have not opted in. Returns the directory in use, or None.
    Safe to call more than once.
    """
    global _SHARED_DIR
    if _SHARED_DIR is not None:
        return _SHARED_DIR  # the RESOLVED dir, not the root -- a caller must get the same
        # answer every time, and the keyed subdirectory is the one packages live in.
    root = os.environ.get("COSMOS_AOT_TOKENIZER_CACHE_DIR")
    if not root:
        return None

    import torch

    from cosmos_framework.model.generator.tokenizers import wan2pt2_vae_4x16x16 as _vae

    shared = os.path.join(root, _cache_key())
    os.makedirs(shared, exist_ok=True)

    torch._inductor.aoti_compile_and_package = _atomic_compile_and_package(torch._inductor.aoti_compile_and_package)
    # Keep the directory across runs. The method looks this up as a module global, so replacing
    # the module attribute reaches it; both call sites (wipe-on-entry, delete-on-exit) go
    # through it, and makedirs stands in for the "recreate" it would have done.
    _vae._synchronized_reset_dir = lambda path, **_kw: os.makedirs(path, exist_ok=True)

    original_compile_encode = _vae.Wan2pt2VAEInterface.compile_encode

    @functools.wraps(original_compile_encode)
    def _compile_encode(self, *args, output_dir=None, **kwargs):
        return original_compile_encode(self, *args, output_dir=shared, **kwargs)

    _vae.Wan2pt2VAEInterface.compile_encode = _compile_encode
    _SHARED_DIR = shared
    print(f"[aot] shared tokenizer AOT cache: {shared}", flush=True)
    return shared
