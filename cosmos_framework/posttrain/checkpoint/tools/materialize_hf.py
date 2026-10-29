# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Identity-bound, idempotent materialization of a consolidated checkpoint into a Hugging Face teacher directory.

``python -m cosmos_framework.posttrain.checkpoint.tools.materialize_hf (--ckpt-dir D | --artifact-uri s3://...) --base B
--out O [--stage-dir S --force --min-tensors N --expect-tensors N --keep-staged --no-verify --unified-chat-template
--credentials C]``

The conversion itself is ``conversion.hf.overlay_consolidated_onto_hf`` (CPU, needs torch + transformers + safetensors;
run inside the training container). This module owns everything around it:

* Source. ``--ckpt-dir`` holds a local ``model.pt`` (trainer or merge artifact), hashed (sha256 + bytes) once before the output
  lock as validation, then OPENED ONCE under the lock: the identity of record is hashed through that open file object, the
  skip / mismatch decision, the source record and the marker use it, and the overlay memory-maps the SAME open file (through
  ``/proc/self/fd``; ``conversion.hf.load_consolidated_state_dict``) and the default verification (``verify_with_torch``) compares
  the staged shards against that same open file (fix round T6t), so by construction the marker names the bytes the overlay
  consumed and the shards are verified against them, whatever the path names meanwhile: the merge tool (tmp + os.replace over ``model.pt``) is not excluded by the output
  lock and may replace the path while the request waits or while the overlay runs (fix rounds T6q and T6r, review thread
  524355e746; a change during the wait is adopted with a WARNING naming both hashes, since ``--ckpt-dir`` names a path; a
  replacement during the overlay is logged and irrelevant to what was built). The opened inode is fstat'd before the hash and
  after the overlay: a change of its size or mtime refuses the build by name, with no "same bytes" acceptance, because the
  digest of a file after the fact says nothing about what a reader consumed. That check is best effort against a writer INTO
  the opened inode: the supported merge tool never writes in place, and a client may report an open inode's attributes late
  (Lustre showed no fstat change for a 16-byte in-place write within 20 ms); a writer that writes in place needs a lock it
  honours. The guarantee of record is against a replaced PATH, by construction.
  ``--artifact-uri`` names an archived ``checkpoint_artifacts/iter_*``: only its small manifest object is read and
  validated (``artifact.validate_artifact_manifest``); the ``model`` descriptor's sha256 + size ARE the source identity,
  so the skip decision needs no download. When a build is needed the payload is fetched through
  ``archive.reader.fetch_archived_artifact`` into ``--stage-dir`` (default ``<out>.stage``) and re-hashed; a fetched
  copy whose digest is not the manifest's is removed and refused.
* Identity-bound idempotency. The ``.materialized`` marker in ``--out`` records the IDENTITY {conversion version,
  source sha256 + bytes, base digest}, the source and base records, the verification options and the output shard
  inventory (name, bytes, tensor count). A later invocation skips (exit 0, safe under job requeue) only when every
  identity field matches what it was asked to build and the recorded shards are on disk unchanged. A same-size
  different checkpoint, another base, another conversion version, or a marker without an identity record (an earlier
  writer) is REFUSED with exit 1 (``--force`` rebuilds, or choose another ``--out``). An output without a marker or
  with a damaged inventory is rebuilt. ``--no-verify`` skips only the torch-side probing; identity, index and sidecar
  checks always run, on the skip path too, with THIS invocation's ``--expect-tensors`` / ``--min-tensors``.
* Staging reuse. ``<out>.tmp`` left by an attempt that DIED after the overlay (a signal, a timeout) is reused when config +
  index exist, every indexed shard is a complete safetensors file with the index's tensors, and its ``.overlay_source.json``
  stamp carries exactly the identity this invocation resolved. No stamp or any difference rebuilds. A request WITHOUT
  ``--force`` only (``--force`` rebuilds from the source, fix round T6l); the reused copy goes through the same verification as a
  fresh build, and ``--no-verify`` then trusts its bytes on size and tensor count. A failure this invocation itself meets before
  the swap removes ``<out>.tmp`` (see the staged construction below).
* Preflight before anything is written. The preflight is every
  torch-free refusal that reads the BASE alone (``preflight``): the teacher files, the content identity, ``config.json``
  parsing to an object with the structural identity that ``config.json <- base`` requires after the overlay, the index's
  ``weight_map`` mapping tensor names to shard names, every base shard the index lists passing the safetensors header check
  the output gets (``conversion.hf.shard_status``: present, readable, header parsed, size = 8 + header + data, header keys =
  the index's) AND accepted by the pinned safetensors reader itself (``conversion.hf.safetensors_reader_error``:
  ``safe_open`` on the header; dtype names, string metadata, contiguous offsets, byte count per tensor, file length; fix
  round T6j, review R149G-01), every Jinja template the loaders read (``chat_template.jinja``,
  ``additional_chat_templates/*.jinja``; ``conversion.hf.loader_template_files``) decoding as strict UTF-8 (fix round T6j),
  every JSON sidecar parsing to a JSON
  object and ``chat_template.json``, when present, carrying a non-empty ``chat_template`` string (the loaders index these
  after the overlay: a ``[]`` sidecar or a ``{}`` template file fails them), in the default path an
  ``additional_chat_templates/`` of regular files only (the sync copies it verbatim, the verifier refuses a sub-directory, a
  dangling symlink fails the copy), and under ``--unified-chat-template`` the supported-base contract below; an OSError met
  while reading the base is a named refusal, never a traceback. Every refusal it raises is one message family,
  ``REFUSED before anything is published: <what was found; what the staged build would have done with it>``
  (``_refuse``). It runs first; only after it passes may ``<out>.tmp`` be created or the overlay run, so a request it
  refuses leaves an existing output byte-identical and no ``<out>.tmp`` behind. What only the output can show (the tensor
  count, the shards, the export's sidecars, the torch probe) is verified on the staged ``<out>.tmp`` BEFORE the swap
  (below), so a failure there keeps the existing output too. One JSON decoder serves the whole preflight and this module's
  readers (``conversion.hf.decode_json_bytes``:
  strict UTF-8 text, as ``model_identity``, ``_read_json_file`` and transformers' loaders decode in the staged build; a UTF-8
  byte-order mark is refused by name, review R149F-01, where ``json.loads(bytes)`` stripped it and a BOM-prefixed ``config.json``
  destroyed an existing export), and ``shard_status`` validates the safetensors header's shape (an object of tensor entries with
  ``dtype``, ``shape`` and two-integer ``data_offsets``, an optional ``__metadata__`` object; review R149F-02), so a malformed
  header is a named refusal here and a named verifier failure on the output, never a traceback; the reader's semantics (an
  unknown dtype, a non-string metadata value, an offset gap or overlap, a byte range that is not dtype size x shape) are the
  pinned reader's own verdict, asked for every base shard (fix round T6j). NOT part of the preflight, by
  design: the weights-free loader calls themselves (``AutoProcessor`` /
  ``AutoTokenizer`` / ``GenerationConfig`` ``from_pretrained`` on the base). They are not torch-free (the processor
  instantiates the image and video processors, which import torch), their verdicts depend on the installed transformers
  rather than on the files, and the synthetic bases of the tests and the review probes are not loadable by them; a base
  that passes every structural check above but fails a loader on a schema fact the files do not state (``tokenizer.json``
  = ``{}``, ``preprocessor_config.json`` without ``image_processor_type``) is still refused by the overlay, in the staged build,
  with the existing output kept (fix round T6j).
* Sidecars. After the overlay every tokenizer / processor / generation sidecar of the base is copied verbatim over
  the staging directory and every one the base lacks is removed (``conversion.hf.sync_sidecars``), then verified.
* Unified chat template (``--unified-chat-template``, default off). After the sidecar sync, BOTH template files of the
  export, ``tokenizer_config.json``'s and ``chat_template.json``'s ``chat_template``, are set to the base's processor
  template in its CANONICAL form (``template_mode`` ``processor-canonical``: the processor template with ONE change in
  its reasoning branch, the replayed ``reasoning_content`` rendered ``| trim``, row rule R1 of record), every other key
  of each file unchanged (``conversion.hf.write_unified_chat_template``), so a request with or without tools renders
  the same reasoning form; ``verify_sidecars`` then checks that rule instead of byte identity for those two files and
  ``verify_with_torch`` expects the canonical text from the loaded tokenizer and the loaded processor; the marker
  records the processor template's sha, the shipped sha of each file, the one-hunk diff and the serve assumption
  under ``chat_template`` and the option under ``options``; the supported base ships its templates in the two JSON
  files and carries no template sidecar a loader prefers over them (``chat_template.jinja``, ``additional_chat_templates/``,
  a ``chat_template`` key in ``processor_config.json``; read on transformers 4.57.6 and 5.8.1): such a base is REFUSED in
  unified mode before anything is published, with or without ``--no-verify``, naming the file, and a ``chat_template.jinja``
  equal to the canonical text is allowed and recorded (``chat_template.preferred_sidecars``); the
  ``additional_chat_templates/`` DIRECTORY is refused whatever it holds (empty, sub-directories only, or files) by the one
  helper the preflight and ``verify_sidecars`` share (``conversion.hf.unified_template_directory_conflict``); the option is part of
  the IDENTITY (present only when on), so an output built with it is refused by a request without it, and the reverse,
  until ``--force``. The default output is byte-identical to one built before the option existed (no contract bump).
* Stage-dir ownership. Only files and directories this invocation created under ``--stage-dir`` are removed after
  success (``--keep-staged`` keeps them); anything that was already there survives, including a copy a caller keeps.
* Staged construction and the swap (previously an existing ``--out`` was removed BEFORE
  the overlay, so a base the torch-free preflight could not fault, a shard the pinned safetensors reader rejects or a Jinja
  template the tokenizer loader cannot decode, cost the export under ``--force --no-verify``). The whole build happens in the
  sibling staging directory ``<out>.tmp``: the overlay (the base read by its real readers: the pinned safetensors reader on
  every shard, transformers on the sidecars), the sidecar sync, ``config.json <- base``, the unified template when asked,
  the verification (index, sidecars and, unless ``--no-verify``, the torch probe, all on the staged copy) and the
  ``.materialized`` marker, the LAST file written there (its semantics unchanged: an ``<out>`` without a marker is a partial
  build). An existing ``<out>`` is not touched until all of that has succeeded; then the swap (``publish_staging``):
  ``<out>`` -> ``<out>.previous``, ``<out>.tmp`` -> ``<out>``, ``<out>.previous`` removed (two ``os.rename`` on one file
  system; Lustre has no RENAME_NOREPLACE or RENAME_EXCHANGE, so ``io.rename_directory_noreplace`` cannot be used here). On ANY
  failure between the staging and the swap the staging directory is removed, the error names its stage and the existing
  output is byte-identical, under ``--force --no-verify`` too; a failure of the second rename puts the old output back under
  its name; only a death (a signal, a timeout) leaves ``<out>.tmp``, for the staging reuse above. Transient disk: TWO exports
  side by side from the overlay to the swap (the staged shards are written by ``save_pretrained``, not linked) plus the fetch
  stage. ``<out>.tmp`` and ``<out>.previous`` are this module's names: a leftover ``<out>.previous`` (an interrupted swap, or
  a directory parked under that name) is refused before the build unless ``--force``; on the idempotent skip it is reported
  and left. Under ``--force`` (fix round T6k, review R149H-01) no complete marked tree is discarded while it is the only one:
  with ``<out>`` absent, ``<out>.previous`` is renamed back to ``<out>`` first (the crash state between the two renames) and
  the build happens beside it; with ``<out>`` present, a leftover ``<out>.previous`` is kept until the swap and removed there,
  after the new tree is verified; ``--force`` then REBUILDS FROM THE SOURCE (fix round T6l, review R149I-01): a death-left
  ``<out>.tmp`` is discarded whatever its marker says once ``<out>`` holds a complete marked tree (a marker match covers the
  identity, the shard sizes and tensor counts, not the shard bytes, and the base digest excludes the base shards' bytes, so a
  kept copy with a flipped data byte, or one built from base shards whose bytes changed with the size kept, must not be
  published by ``--force``); a complete marked ``<out>.tmp`` that is the only complete tree is not discarded: the request is
  refused by name (a request without ``--force`` publishes it after verification, or it is removed by hand). The kept-copy
  publication without a fresh overlay is a request WITHOUT ``--force`` only (the staging reuse above, the idempotent recovery),
  after the same verification as a fresh build: index, sidecars and the torch probe unless ``--no-verify``, which trusts the
  kept copy's bytes on size and tensor count.
* One materializer per output (fix round T6k, review R149H-02). ``<out>.lock`` beside the output, held with
  ``fcntl.flock(LOCK_EX | LOCK_NB)`` (``output_lock``) from before ``<out>.previous`` / ``<out>.tmp`` / the fetch stage are
  touched to after the swap and the removal of ``<out>.previous``; a second materializer on the same ``--out`` is REFUSED by
  name (``REFUSED: <out> is being materialized by another process (<out>.lock held by pid P on host H since T)``) without
  touching anything, after WAITING for a held lock up to ``LOCK_WAIT_SECONDS`` = 120 s (``--lock-wait-seconds``; polling every 0.2 s;
  fix round T6n, review R149K-02: two simultaneous identical requests of a published source both end with 0, the second skipping
  once the holder has finished; a differing request is refused on the marker under the lock; a holder still there after the
  bound is refused by name with the bound stated). flock is released by the kernel when its holder exits (a crash, a kill), so a
  held lock means a live holder: wait, or check that pid on that host. The lock file stays beside the output (unlinking it would race a waiter that
  already opened it) and carries the last holder's record; it is not part of the export. The file system must support flock
  (Lustre: the ``flock`` mount option, as on the cluster); one that refuses it is a named refusal (fail closed).
  The marker decision (skip / mismatch / build) is taken UNDER the lock (fix round T6m): a materializer that paused before the
  lock while another published a different source sees that export's marker when it acquires and is refused without ``--force``
  instead of overwriting it with a stale "absent" decision.
* An inherited sole complete marked staging copy is never deleted (fix round T6m). When a first publication died after writing
  ``.materialized`` and before the rename, ``<out>.tmp`` is the only completed copy: ``--force`` refuses by name (above) and a
  request without ``--force`` recovers it under protection: the marker must be this request's (else refused by name), a missing
  staged ``config.json`` is repaired from the base through a temporary file and a rename, the sidecars are re-synced the same way
  (files through ``.copying`` + replace, the template directory through a staged copy and renames; never a shard), then the
  verification a fresh build gets, the marker and the swap; a failure at any of these refuses by name and KEEPS the tree: its
  shards and its marker untouched, its sidecars possibly repaired from the base before the check (fix round T6n, review R149K-01;
  under ``--unified-chat-template`` the two template files are re-synced to the base's form at the sidecar-sync stage and rewritten
  with the unified text through a ``.writing`` temporary, so a failure there leaves ``<template>.writing`` beside base-form files:
  the refusal names "a .copying or .writing temporary"; fix round T6o, review R149L-01; the refusal's third clause for the state
  at ``<out>``, a complete marked export standing there, is unreachable by construction: the skip / mismatch decision in
  ``run_request`` itself, under the lock, returns or refuses before the recovery condition is evaluated whenever a complete marked
  export stands at ``<out>``, for the CLI and for a direct caller alike; the clause is kept as a defensive branch, since the state is
  read at refusal time: only a hand change of ``<out>`` during the run could print it (fix round T6p, review R149M-01):
  the refusal says so and names the state at ``<out>``: absent, standing but not a complete marked export, or a complete marked
  export); no discard, no half-written file; the overlay is never run while the copy is the sole complete tree (no rebuild beside
  it).

Known answers live in ``tests/test_checkpoint_materialize_hf.py`` (torch-free, the overlay replaced by a fake).
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import os
import shutil
import socket
import sys
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, BinaryIO, NoReturn
from urllib.parse import urlsplit

from cosmos_framework.posttrain.checkpoint.artifact import CHECKPOINT_ARTIFACT_MANIFEST, validate_artifact_manifest
from cosmos_framework.posttrain.checkpoint.conversion import hf
from cosmos_framework.posttrain.checkpoint.conversion.hf import (
    INDEX_FILE,
    REQUIRED_FILES,
    SIDECAR_DIRS,
    SIDECAR_FILES,
    ConversionError,
    safetensors_header,
    shard_status,
    sync_sidecars,
    verify_index,
    verify_sidecars,
    verify_with_torch,
)
from cosmos_framework.posttrain.checkpoint.io import atomic_write_json
from cosmos_framework.posttrain.checkpoint.manifest import iteration_name, parse_json_object
from cosmos_framework.posttrain.utils import object_store
from cosmos_framework.posttrain.utils.log import logger

MATERIALIZED_MARKER = ".materialized"
MARKER_PROTOCOL = "posttrain.materialized-hf.v1"
# Contract of the bytes this module produces (overlay with strict accounting, experts in the checkpoint layout on
# disk, base sidecars verbatim, source proven by sha256 + bytes for archived artifacts too). Bump whenever any of that
# changes: outputs built under the old contract then stop satisfying the idempotent skip and are refused until --force.
CONVERSION_VERSION = "materialize_model_pt_to_hf/5"  # /5: restore declared embedding ties after assign=True; index single-shard HF exports
SOURCE_STAMP = ".overlay_source.json"  # written into <out>.tmp right after the overlay: the identity that built it
STAGING_SUFFIX = ".tmp"  # <out>.tmp: the staging directory the whole build happens in, beside an existing output
PREVIOUS_SUFFIX = ".previous"  # <out>.previous: the old output during the swap; a leftover is refused unless --force
LOCK_SUFFIX = ".lock"  # <out>.lock: the per-output flock held from before the staging to after the swap (fix round T6k)
LOCK_WAIT_SECONDS = (
    120.0  # how long a request waits for a held <out>.lock before refusing by name (fix round T6n; --lock-wait-seconds)
)
LOCK_POLL_SECONDS = 0.2  # the polling interval of that wait
DEFAULT_MIN_TENSORS = 800
PAYLOAD = "model.pt"


class MaterializeRefusedError(RuntimeError):
    """``--out`` holds a different or unprovable materialization; ``force`` rebuilds over it."""


def _log(msg: str) -> None:
    logger.info(f"[materialize] {msg}")


def _load_json(path: str) -> Any:
    with open(path) as handle:
        return json.load(handle)


def read_index_weight_map(index_path: str) -> dict[str, str]:
    """The ``weight_map`` of one safetensors index, validated torch-free: ``index_path`` parses to a JSON object whose
    ``weight_map`` is a non-empty JSON object mapping tensor names to shard file names (non-empty strings). ``ValueError``,
    ``<index_path> unreadable: <what was found>``, for every other shape, so no index escapes the iteration that follows as
    AttributeError / TypeError (review R149E-03: ``{"weight_map": []}`` and ``{"weight_map": {"a.weight": 1}}`` did) and an
    empty ``weight_map`` (no shard to open; ``verify_index`` refuses it on the output) is refused before the staged build too.
    Used on the base (``base_identity``, the preflight) and on ``<out>.tmp`` (``tmp_complete``). Decodes through the one decoder,
    ``conversion.hf.decode_json_bytes`` (strict UTF-8 text; a byte-order mark is refused by name, review R149F-01)."""
    try:
        with open(index_path, "rb") as handle:
            payload = hf.decode_json_bytes(handle.read())
    except (OSError, ValueError) as error:
        raise ValueError(f"{index_path} unreadable: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{index_path} unreadable: the index is {hf.describe_json_value(payload)}, not a JSON object")
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(
            f"{index_path} unreadable: weight_map must be a non-empty JSON object mapping tensor names to shard names, "
            f"found {hf.describe_json_value(weight_map)}"
        )
    bad = sorted(name for name, shard in weight_map.items() if not isinstance(shard, str) or not shard)
    if bad:
        raise ValueError(
            f"{index_path} unreadable: weight_map must map tensor names to shard names (non-empty strings); "
            f"{bad[:5]} map to {[weight_map[name] for name in bad[:5]]!r}"
        )
    return weight_map


# ---------------------------------------------------------------- hashing / identity --------------------------------
def source_digest(path: str) -> dict[str, Any]:
    """Return ``{"sha256": hexdigest, "bytes": size}`` of one file (one sequential read through the shared hasher)."""
    return {"sha256": object_store.file_sha256(path), "bytes": os.path.getsize(path)}


def base_identity(base: str) -> dict[str, Any]:
    """Content-defined, torch-free identity of the HF base.

    sha256 over config.json, the index, every present sidecar file (and every file under the sidecar directories) plus
    the (shard, bytes) inventory the index lists. ``digest`` is what reuse compares; ``path`` is provenance only, so a
    byte-identical base under another path has the same digest. Raises ``FileNotFoundError`` for an incomplete base and
    ``ValueError`` for an index whose ``weight_map`` is not a non-empty object of tensor name -> shard name
    (``read_index_weight_map``); the preflight turns both into its one refusal family. The shards are inventoried by
    size only here; the preflight runs the safetensors header check on each of them.
    """
    files: dict[str, str] = {}
    for name in ("config.json", INDEX_FILE, *SIDECAR_FILES):
        path = os.path.join(base, name)
        if os.path.isfile(path):
            files[name] = object_store.file_sha256(path)
    for directory in SIDECAR_DIRS:
        directory_path = os.path.join(base, directory)
        if os.path.isdir(directory_path):
            for name in sorted(os.listdir(directory_path)):
                path = os.path.join(directory_path, name)
                if os.path.isfile(path):
                    files[f"{directory}/{name}"] = object_store.file_sha256(path)
    if INDEX_FILE not in files or "config.json" not in files:
        raise FileNotFoundError(f"base model incomplete at {base}: config.json + {INDEX_FILE} required")
    index_path = os.path.join(base, INDEX_FILE)
    weight_map = read_index_weight_map(index_path)
    shards: list[list[Any]] = []
    for shard in sorted(set(weight_map.values())):
        path = os.path.join(base, shard)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"base model incomplete at {base}: shard {path} listed in {index_path} is missing")
        shards.append([shard, os.path.getsize(path)])
    digest = hashlib.sha256(json.dumps({"files": files, "shards": shards}, sort_keys=True).encode()).hexdigest()
    return {"path": os.path.realpath(base), "digest": digest, "n_shards": len(shards), "n_tensors": len(weight_map)}


def make_identity(
    source: dict[str, Any],
    base_id: dict[str, Any],
    *,
    unified_chat_template: bool = False,
    output_dtype: str | None = None,
) -> dict[str, Any]:
    """The fields a reuse (final output or ``<out>.tmp``) must reproduce exactly.

    Verification thresholds are NOT identity: they are re-enforced on every skip with the current invocation's values.
    ``unified_chat_template`` IS identity (the bytes of ``tokenizer_config.json`` differ) and is recorded only when on, so
    every marker written before the option existed keeps its identity.
    """
    identity = {
        "conversion": CONVERSION_VERSION,
        "source_sha256": source["sha256"],
        "source_bytes": source["bytes"],
        "base_digest": base_id["digest"],
    }
    if unified_chat_template:
        identity["unified_chat_template"] = True
    if output_dtype is not None:
        identity["output_dtype"] = output_dtype
    return identity


def identity_digest(identity: dict[str, Any]) -> str:
    """Canonical sha256 of one identity record (key order independent)."""
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def _short(identity: dict[str, Any]) -> str:
    return (
        f"{identity.get('conversion')} source sha256 {str(identity.get('source_sha256'))[:12]} "
        f"({identity.get('source_bytes')} bytes) base {str(identity.get('base_digest'))[:12]}"
        f"{' unified-chat-template' if identity.get('unified_chat_template') else ''}"
    )


# ---------------------------------------------------------------- source resolution ---------------------------------
def parse_artifact_uri(uri: str) -> tuple[str, str]:
    """Split ``s3://bucket/<...>/checkpoint_artifacts/iter_*`` into ``(bucket, prefix)``; ``ValueError`` otherwise."""
    parts = urlsplit(uri)
    prefix = parts.path.strip("/")
    if parts.scheme != "s3" or not parts.netloc or not prefix:
        raise ValueError(f"--artifact-uri must look like s3://bucket/<...>/checkpoint_artifacts/iter_*, got {uri!r}")
    iteration_name(uri)
    return parts.netloc, prefix


def probe_local_source(ckpt_dir: str) -> dict[str, Any]:
    """Identify a local consolidated checkpoint by hashing ``<ckpt_dir>/model.pt`` once."""
    path = os.path.join(ckpt_dir, PAYLOAD)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path} does not exist")
    _log(f"hashing {path} ({os.path.getsize(path) / 2**30:.1f} GiB)")
    return {"kind": "local", "path": os.path.abspath(path), **source_digest(path)}


def probe_archived_source(artifact_uri: str, credentials: str | None = None) -> dict[str, Any]:
    """Identify an archived artifact from its manifest object WITHOUT downloading the payload.

    The manifest is validated with the native rules and must describe a hashed consolidated ``model.pt`` component;
    its sha256 + size are the source identity, so the archive proof no longer depends on ETags.
    """
    bucket, prefix = parse_artifact_uri(artifact_uri)
    client = object_store.build_s3_client(credentials or object_store.DEFAULT_CREDENTIALS)
    key = f"{prefix}/{CHECKPOINT_ARTIFACT_MANIFEST}"
    try:
        payload = client.get_object(Bucket=bucket, Key=key)["Body"].read()
    except Exception as error:  # noqa: BLE001 -- any read failure means the artifact cannot be proven
        raise FileNotFoundError(
            f"archived artifact manifest s3://{bucket}/{key} is missing or unreadable ({error})"
        ) from error
    manifest = validate_artifact_manifest(
        parse_json_object(payload, context=f"archived artifact manifest at {artifact_uri!r}"),
        artifact_location=artifact_uri,
    )
    descriptor = manifest["components"].get("model")
    if (
        not isinstance(descriptor, dict)
        or descriptor.get("kind") != "file"
        or descriptor.get("path") != PAYLOAD
        or not descriptor.get("sha256")
    ):
        raise ValueError(
            f"{artifact_uri}: the archived manifest has no hashed consolidated 'model' component ({PAYLOAD} with sha256); "
            f"only consolidated artifacts with a payload digest can be materialized"
        )
    return {
        "kind": "artifact",
        "artifact_uri": artifact_uri,
        "bucket": bucket,
        "prefix": prefix,
        "sha256": descriptor["sha256"],
        "bytes": int(descriptor["size"]),
        "manifest_sha256": hashlib.sha256(payload).hexdigest(),
    }


# ---------------------------------------------------------------- staging: owned files only -------------------------
@dataclass
class StagedSource:
    """What one invocation fetched: the directory holding ``model.pt`` and exactly the paths it created."""

    stage_dir: str
    artifact_dir: str
    created_files: list[str] = field(default_factory=list)
    created_dirs: list[str] = field(default_factory=list)  # deepest first, so an empty tree unwinds bottom-up


def _tree_snapshot(root: str) -> tuple[set[str], set[str]]:
    files: set[str] = set()
    dirs: set[str] = set()
    if os.path.isdir(root):
        dirs.add(os.path.abspath(root))
        for current, names, filenames in os.walk(root):
            dirs.update(os.path.abspath(os.path.join(current, name)) for name in names)
            files.update(os.path.abspath(os.path.join(current, name)) for name in filenames)
    return files, dirs


def _fetch_archived_artifact(artifact_uri: str, dst_dir: str, *, credentials: str | None) -> str:
    # Imported lazily: the archive reader is needed only on the fetch path.
    from cosmos_framework.posttrain.checkpoint.archive.reader import fetch_archived_artifact

    return fetch_archived_artifact(artifact_uri, dst_dir, credentials=credentials)


def fetch_archived_source(probe: dict[str, Any], stage_dir: str, credentials: str | None = None) -> StagedSource:
    """Fetch the probed artifact into ``stage_dir`` and prove the payload against the manifest digest.

    Records exactly which files and directories the fetch created (the ownership ``cleanup_staged`` honours). A payload
    whose sha256 or size is not the manifest's is removed again and refused: an unproven teacher must never be built.
    """
    files_before, dirs_before = _tree_snapshot(stage_dir)
    _log(f"fetching {probe['artifact_uri']} ({probe['bytes'] / 2**30:.1f} GiB) -> {stage_dir}")
    started = time.time()
    artifact_dir = _fetch_archived_artifact(probe["artifact_uri"], stage_dir, credentials=credentials)
    files_after, dirs_after = _tree_snapshot(stage_dir)
    staged = StagedSource(
        stage_dir=stage_dir,
        artifact_dir=artifact_dir,
        created_files=sorted(files_after - files_before),
        created_dirs=sorted(dirs_after - dirs_before, key=len, reverse=True),
    )
    model_pt = os.path.join(artifact_dir, PAYLOAD)
    if not os.path.isfile(model_pt):
        cleanup_staged(staged)
        raise FileNotFoundError(f"fetch of {probe['artifact_uri']} left no {model_pt}")
    digest = source_digest(model_pt)
    if digest["sha256"] != probe["sha256"] or digest["bytes"] != probe["bytes"]:
        cleanup_staged(staged)
        raise ConversionError(
            f"fetched {model_pt} sha256 {digest['sha256'][:12]} ({digest['bytes']} bytes) != the archived manifest's "
            f"{probe['sha256'][:12]} ({probe['bytes']} bytes); refusing to materialize an unproven payload"
        )
    _log(f"fetched and verified in {time.time() - started:.0f}s ({len(staged.created_files)} files created)")
    return staged


def cleanup_staged(staged: StagedSource) -> list[str]:
    """Remove exactly what the fetch created: its files, then its directories bottom-up while they are empty."""
    removed: list[str] = []
    for path in staged.created_files:
        if os.path.lexists(path):
            os.remove(path)
            removed.append(path)
    for path in staged.created_dirs:
        if os.path.isdir(path) and not os.listdir(path):
            os.rmdir(path)
            removed.append(path + "/")
    return removed


# ---------------------------------------------------------------- idempotency: identity-bound marker -----------------
def shard_inventory(out: str) -> list[dict[str, Any]]:
    """``[{name, bytes, n_tensors}]`` for every shard the index lists (header parsed, no torch)."""
    weight_map = _load_json(os.path.join(out, INDEX_FILE))["weight_map"]
    inventory = []
    for shard in sorted(set(weight_map.values())):
        path = os.path.join(out, shard)
        header, _ = safetensors_header(path)
        inventory.append({"name": shard, "bytes": os.path.getsize(path), "n_tensors": len(header)})
    return inventory


def inventory_mismatch(out: str, want: list[dict[str, Any]]) -> str | None:
    """None when every recorded shard is on disk with the recorded size and tensor count; else the reason."""
    for entry in want:
        path = os.path.join(out, entry["name"])
        if not os.path.isfile(path):
            return f"{path} recorded in the marker is missing"
        size = os.path.getsize(path)
        if size != entry["bytes"]:
            return f"{path} is {size} bytes, the marker recorded {entry['bytes']}"
        try:
            header, _ = safetensors_header(path)
        except (OSError, ValueError) as error:
            return f"{path}: unreadable safetensors header ({error})"
        if len(header) != entry["n_tensors"]:
            return f"{path} holds {len(header)} tensors, the marker recorded {entry['n_tensors']}"
    return None


def marker_status(out: str, want: dict[str, Any], probe: dict[str, Any]) -> tuple[str, str]:
    """Early-skip decision on a final output directory: ``("absent"|"match"|"mismatch"|"damaged", reason)``.

    ``want`` = ``{"conversion", "base_digest"[, "base_path", "unified_chat_template"]}``; ``probe`` = the requested
    source's ``sha256`` + ``bytes``. A marker without the ``unified_chat_template`` key was built without the option.
      absent   -> no marker (partial output): rebuilt.
      match    -> conversion, base digest and source sha256 + bytes agree AND the recorded shards are on disk unchanged.
      mismatch -> a DIFFERENT or UNPROVABLE materialization (other source / base / conversion, a marker without an
                  identity record, an unreadable marker): refused unless forced.
      damaged  -> identity agrees but a recorded shard is missing or changed: rebuilt.
    """
    marker_path = os.path.join(out, MATERIALIZED_MARKER)
    if not os.path.isfile(marker_path):
        return "absent", f"{marker_path} missing"
    try:
        marker = _load_json(marker_path)
    except (OSError, ValueError) as error:
        return "mismatch", f"{marker_path} unreadable ({error})"
    identity = marker.get("identity") if isinstance(marker, dict) else None
    if not isinstance(identity, dict):
        return "mismatch", (
            f"{marker_path} has no identity record (written by an earlier writer); its source, base and conversion "
            f"cannot be proven"
        )
    if identity.get("conversion") != want["conversion"]:
        return (
            "mismatch",
            f"{marker_path} conversion {identity.get('conversion')!r} != this module's {want['conversion']!r}",
        )
    if identity.get("base_digest") != want["base_digest"]:
        return "mismatch", (
            f"{marker_path} base digest {str(identity.get('base_digest'))[:12]} ({(marker.get('base') or {}).get('path')}) "
            f"!= current base digest {want['base_digest'][:12]} ({want.get('base_path')})"
        )
    if bool(identity.get("unified_chat_template", False)) != bool(want.get("unified_chat_template", False)):
        built = "with" if identity.get("unified_chat_template") else "without"
        asked = "with" if want.get("unified_chat_template") else "without"
        return "mismatch", (
            f"{marker_path} was built {built} --unified-chat-template, this request is {asked} it; tokenizer_config.json's "
            f"chat_template differs between the two"
        )
    if identity.get("output_dtype") != want.get("output_dtype"):
        return "mismatch", f"{marker_path} has a different merged export output_dtype"
    source = marker.get("source") if isinstance(marker.get("source"), dict) else {}
    if identity.get("source_sha256") != probe["sha256"] or identity.get("source_bytes") != probe["bytes"]:
        requested = probe.get("path") or probe.get("artifact_uri")
        return "mismatch", (
            f"{marker_path} source sha256 {str(identity.get('source_sha256'))[:12]} ({identity.get('source_bytes')} bytes, "
            f"{source.get('path') or source.get('artifact_uri')}) != requested {requested} sha256 {probe['sha256'][:12]} "
            f"({probe['bytes']} bytes)"
        )
    inventory = marker.get("shards")
    if not isinstance(inventory, list) or not inventory:
        return "mismatch", f"{marker_path} has no output shard inventory"
    bad = inventory_mismatch(out, inventory)
    if bad:
        return "damaged", bad
    return (
        "match",
        f"identity {identity_digest(identity)[:12]} ({_short(identity)}), {len(inventory)} shards on disk as recorded",
    )


def tmp_complete(tmp_out: str, identity: dict[str, Any] | None = None) -> tuple[bool, str]:
    """Staging-reuse decision for ``<out>.tmp``: ``(reusable, reason)``.

    Reusable when config.json + index exist, every indexed shard is a complete safetensors file holding exactly the
    index's tensors for it, and (when ``identity`` is given) the stamp written after the overlay carries EXACTLY that
    identity. No stamp, a stamp without an identity record, or any identity difference is not reusable: without the
    identity the staged bytes cannot be tied to this request.
    """
    if not os.path.isdir(tmp_out):
        return False, f"{tmp_out} does not exist"
    index_path = os.path.join(tmp_out, INDEX_FILE)
    if not os.path.isfile(index_path):
        return False, f"{index_path} missing"
    if not os.path.isfile(os.path.join(tmp_out, "config.json")):
        return False, f"{tmp_out}/config.json missing"
    try:
        weight_map = read_index_weight_map(index_path)
    except ValueError as error:  # "<index_path> unreadable: <what was found>", a list-shaped weight_map included
        return False, str(error)
    shards = sorted(set(weight_map.values()))
    for shard in shards:
        reason = shard_status(os.path.join(tmp_out, shard), {k for k, v in weight_map.items() if v == shard})
        if reason:
            return False, reason
    if identity is not None:
        stamp_path = os.path.join(tmp_out, SOURCE_STAMP)
        if not os.path.isfile(stamp_path):
            return False, f"{stamp_path} missing: the staging dir carries no source/base/conversion identity"
        try:
            stamp = _load_json(stamp_path)
        except (OSError, ValueError) as error:
            return False, f"{stamp_path} unreadable ({error})"
        have = stamp.get("identity") if isinstance(stamp, dict) else None
        if not isinstance(have, dict):
            return False, f"{stamp_path} has no identity record (built by an earlier writer): {stamp}"
        if have != identity:
            diff = sorted(k for k in set(have) | set(identity) if have.get(k) != identity.get(k))
            return (
                False,
                f"{stamp_path} identity differs on {diff}: staged [{_short(have)}] vs requested [{_short(identity)}]",
            )
    return (
        True,
        f"config.json + index + {len(shards)} complete shards ({len(weight_map)} tensors), identity matches the stamp",
    )


# ---------------------------------------------------------------- request ------------------------------------------
def validate_args(a: argparse.Namespace) -> str | None:
    """Return an error message or None. Pure (no filesystem), so tests can enumerate every branch."""
    if not a.out:
        return "--out is required"
    if not a.base:
        return "--base is required"
    if bool(a.ckpt_dir) == bool(a.artifact_uri):
        return "pass exactly one of --ckpt-dir <dir with model.pt> or --artifact-uri s3://bucket/<...>/checkpoint_artifacts/iter_*"
    if a.artifact_uri:
        try:
            parse_artifact_uri(a.artifact_uri)
        except ValueError as error:
            return str(error)
    if a.min_tensors < 1:
        return "--min-tensors must be >= 1"
    if a.expect_tensors < 0:
        return "--expect-tensors must be >= 0 (0 = skip the exact count)"
    if os.path.abspath(a.out.rstrip("/")) == os.path.abspath(a.base.rstrip("/")):
        return "--out must not be the base model dir"
    wait = getattr(a, "lock_wait_seconds", LOCK_WAIT_SECONDS)
    if not math.isfinite(wait) or wait < 0:
        # argparse's float accepts "inf" (an unbounded wait) and "nan" (which max() turns into 0): both refused by name (fix round T6o)
        return f"--lock-wait-seconds must be >= 0 and finite (0 = refuse a held lock at once), got {wait!r}"
    return None


REFUSED_PREFIX = "REFUSED before anything is published: "


def _refuse(reason: str) -> NoReturn:
    """The one message family of every preflight refusal (review R149E-04): the required files, the content identity, the
    index and its shards, ``config.json``, the JSON sidecars, the template directory, the unified-mode contract and an
    unreadable base all raise through it as ``REFUSED before anything is published: <what was found; what the staged
    build would have done with it>``. A reason that already carries the prefix (the unified conflict text of
    ``conversion.hf.refuse_unified_template_conflicts``) is not prefixed twice."""
    raise ConversionError(REFUSED_PREFIX + reason.removeprefix(REFUSED_PREFIX))


def preflight(base: str, *, unified_chat_template: bool) -> dict[str, Any]:
    """Every torch-free refusal that reads the BASE alone, run by ``run_request`` before anything is written (the creation of
    ``<out>.tmp``, the overlay) and so before the swap that replaces an existing ``out``. Each check names what the
    staged build would otherwise do with the base,
    and every refusal is raised through ``_refuse`` (one message family, review R149E-04): (1) the teacher files the base
    must carry (``REQUIRED_FILES``); (2) its content identity (``base_identity``: config, index, shards and every sidecar
    hashed; a missing shard, or an index whose ``weight_map`` is not tensor name -> shard name, ``read_index_weight_map``,
    review R149E-03); (3) ``config.json`` parses to a JSON object (its own reason when it does not) carrying the structural
    identity (``model_identity_from_config``) that ``write_config_from_base`` requires after the overlay; (4) every base
    shard the index lists passes the safetensors header check the OUTPUT gets in ``verify_index`` (``shard_status``: present,
    readable, header parsed, size = 8 + header + data, header keys = the tensors the index maps to it): the overlay's
    ``from_pretrained`` opens every shard, and a shard truncated by one byte, with a garbage header or unreadable used to pass
    because ``base_identity`` records sizes only (review R149E-01), AND is accepted by the pinned safetensors reader itself
    (``conversion.hf.safetensors_reader_error``, ``safe_open`` on the header: dtype names, string metadata values, contiguous
    offsets, byte count per tensor, file length; fix round T6j, review R149G-01, where a BF16 ``[3]`` tensor over four bytes,
    an unknown dtype, a non-string metadata value and an offset gap passed the shape check); (5b) every Jinja template the
    tokenizer and processor loaders read as UTF-8 text (``conversion.hf.loader_template_files``: ``chat_template.jinja`` and
    every ``additional_chat_templates/*.jinja`` named template, the loader lines cited there) decodes as strict UTF-8
    (``conversion.hf.utf8_text_error``; fix round T6j, where a template with byte FF passed and the tokenizer loader raised
    UnicodeDecodeError in the overlay); (5) every JSON sidecar the base
    carries parses to a JSON object (``read_tokenizer_config`` for ``tokenizer_config.json``, ``json.loads`` plus the object
    rule for the rest: the loaders index them after the overlay and a ``[]`` fails them) and ``chat_template.json``, when
    present, carries a non-empty ``chat_template`` string (``read_processor_template``: the processor loader's legacy path
    indexes ``["chat_template"]``, so ``{}`` is a KeyError at load; review R149E-02); (6) without the option,
    ``additional_chat_templates/`` holds regular files only: ``sync_sidecars`` copies the directory verbatim and
    ``verify_sidecars`` refuses a sub-directory (``filecmp`` reports it as a mismatch) while a dangling symlink fails the
    copy (review R149D-01); (7) under ``--unified-chat-template`` the supported-base contract
    (``conversion.hf.preflight_unified_base``: both template files parse and carry a template string, the processor
    template canonicalizes, no loader-preferred sidecar conflicts, the directory refused whatever it holds). An ``OSError``
    met while reading the base (an unreadable directory or file; a file that vanished after its presence check) is refused
    with its class and path, never traced (review R149D-02). Returns ``{"base": base_identity(base), "preferred_sidecars":
    the allowed ones}``. What only the OUTPUT can show (the tensor count, the export's shards and sidecars, the torch probe)
    is verified on the staged ``<out>.tmp`` before the swap, so a failure there keeps an existing ``out`` too (fix round
    T6j); the weights-free loader calls stay in the overlay (module docstring: why).
    """
    allowed: list[dict[str, Any]] = []
    try:
        for name in REQUIRED_FILES:
            path = os.path.join(base, name)
            if not hf.required_file_present(base, name):
                _refuse(
                    f"the base {base} lacks {name} (no regular file at {path}), which the teacher directory must carry; the "
                    f"overlay and every loader read it in the staged build"
                )
        try:
            base_id = base_identity(base)
        except FileNotFoundError as error:
            _refuse(f"{error}; the overlay opens every shard the index lists in the staged build")
        except ValueError as error:
            _refuse(f"{error}; the overlay reads the index in the staged build")
        config_path = os.path.join(base, "config.json")
        with open(config_path, "rb") as handle:
            raw_config = handle.read()
        try:
            config = hf.decode_json_bytes(raw_config)  # the one decoder (strict UTF-8 text), review R149F-01
        except hf.ByteOrderMarkError as error:
            _refuse(
                f"{config_path} {error}; the overlay's from_pretrained and config.json <- base decode it as UTF-8 text after an "
                f"existing output is removed"
            )
        except ValueError as error:
            _refuse(
                f"{config_path} is not valid JSON ({error}); the overlay's from_pretrained and config.json <- base read it "
                f"in the staged build"
            )
        if not isinstance(config, dict):
            _refuse(
                f"{config_path} is not a JSON object ({hf.describe_json_value(config)}); the overlay's from_pretrained and "
                f"config.json <- base read it in the staged build"
            )
        if hf.model_identity_from_config(config) is None:
            _refuse(
                f"model identity mismatch: the base {base} config.json carries no structural identity (model_type, "
                f"architectures, {', '.join(hf.REQUIRED_MODEL_IDENTITY_FIELDS)}), which config.json <- base requires after "
                f"the overlay; nothing published"
            )
        # The shards (review R149E-01): the check verify_index applies to the output, on every base shard the index lists.
        weight_map = read_index_weight_map(os.path.join(base, INDEX_FILE))
        for shard in sorted(set(weight_map.values())):
            shard_path = os.path.join(base, shard)
            reason = shard_status(shard_path, {k for k, v in weight_map.items() if v == shard})
            if reason:
                _refuse(
                    f"base shard {reason}; the overlay opens every base shard the index lists (from_pretrained) in the staged "
                    f"build, so a shard that is missing, truncated, unreadable or not the index's is refused first"
                )
            # The pinned reader itself (fix round T6j, review R149G-01): the semantics the shape check does not claim.
            reason = hf.safetensors_reader_error(shard_path)
            if reason:
                _refuse(
                    f"base shard {shard_path}: refused by the pinned safetensors reader ({reason}); the overlay's "
                    f"from_pretrained reads every base shard with that reader in the staged build, so a header it rejects (an "
                    f"unknown dtype, a non-string metadata value, offsets with a gap or an overlap, a byte range that is not the "
                    f"dtype size x the shape) is refused first, by file and reader message"
                )
        # The JSON sidecars (fix round T6g; the object rule and the template string, review R149E-02).
        for name in SIDECAR_FILES:
            path = os.path.join(base, name)
            if not name.endswith(".json") or not os.path.isfile(path):
                continue
            try:
                if name == hf.TOKENIZER_CONFIG_FILE:
                    hf.read_tokenizer_config(base)
                elif name == hf.PROCESSOR_TEMPLATE_FILE:
                    hf.read_processor_template(base)
                else:
                    with open(path, "rb") as handle:
                        payload = hf.decode_json_bytes(handle.read())  # the one decoder (strict UTF-8 text), R149F-01
                    if not isinstance(payload, dict):
                        raise ConversionError(f"{path} is not a JSON object ({hf.describe_json_value(payload)})")
            except ConversionError as error:
                _refuse(
                    f"{error}; the overlay loads the base's sidecars through transformers and the template record reads "
                    f"{name} after the overlay, so a base whose JSON sidecar is not what the loaders index is refused first"
                )
            except hf.ByteOrderMarkError as error:
                _refuse(
                    f"{path} {error}; the overlay loads the base's sidecars through transformers, which decode them as UTF-8 "
                    f"text in the staged build, so such a base is refused first"
                )
            except ValueError as error:
                _refuse(
                    f"{path} is not valid JSON ({error}); the overlay loads the base's sidecars through transformers in the "
                    f"staged build, so a base whose JSON sidecar does not parse is refused first"
                )
        # The Jinja templates the loaders read as UTF-8 text: chat_template.jinja and
        # every additional_chat_templates/*.jinja named template (hf.loader_template_files cites the loader lines). The JSON
        # loop above skips them and the identity only hashes them, so a template with a byte that is not UTF-8 passed the
        # preflight and the tokenizer loader raised UnicodeDecodeError in the overlay.
        for path in hf.loader_template_files(base):
            reason = hf.utf8_text_error(path)
            if reason:
                _refuse(
                    f"{path} is not UTF-8 text ({reason}); the tokenizer and processor loaders read chat_template.jinja and "
                    f"every additional_chat_templates/*.jinja named template as UTF-8 text when the overlay loads the base, so "
                    f"such a base is refused first, by file and decode position"
                )
        if unified_chat_template:
            # Before anything is published or removed, with or without --no-verify: the supported base ships its templates
            # in the two JSON files only; a chat_template.jinja (the tokenizer loader prefers it), the additional_chat_templates/
            # directory (refused whatever it holds) or a chat_template key in processor_config.json with another text would
            # make the export render a template the option does not promise.
            # An equal chat_template.jinja is allowed and recorded. Its texts join the one family (review R149E-04).
            try:
                allowed = hf.preflight_unified_base(base)
            except ConversionError as error:
                if str(error).startswith(REFUSED_PREFIX):
                    raise
                _refuse(
                    f"{error}; --unified-chat-template rewrites both template files from the base's processor template "
                    f"after the overlay, so a base it cannot unify is refused first"
                )
        else:
            # The default path copies additional_chat_templates/ verbatim after the overlay and verifies it file by file:
            # a sub-directory (filecmp reports it as a mismatch) or a dangling symlink (the copy fails) would cost an existing
            # output, so the directory must hold regular files only (review R149D-01; the same lister as the unified rule).
            record = hf.unified_template_directory_conflict(base)
            irregular = [
                entry for entry in (record["entries"] if record is not None else []) if entry["kind"] != "file"
            ]
            if irregular:
                named = ", ".join(f"{entry['name']} [{entry['kind']}]" for entry in irregular)
                _refuse(
                    f"{os.path.join(base, SIDECAR_DIRS[0])}/ holds entries that are not regular files ({named}); the sidecar "
                    f"sync copies the directory verbatim after the overlay and the sidecar verifier refuses a sub-directory "
                    f"while a dangling symlink fails the copy, so such a base is refused first"
                )
    except OSError as error:
        # An unreadable directory or file (R149D-02), or a file that vanished between its presence check and its read: the
        # class and the errno text with the path, in the one family (every expected absence is refused by name above).
        _refuse(f"the base {base} is not readable: {type(error).__name__}: {error}")
    _log(
        f"preflight OK: base {base_id['digest'][:12]} ({base_id['n_tensors']} tensors), {len(allowed)} preferred sidecar(s) allowed"
    )
    return {"base": base_id, "preferred_sidecars": allowed}


def _remove_path(path: str) -> None:
    """Remove what is at ``path``: a directory tree, or a file / symlink as such (its target untouched). Used on the names
    this module owns, ``<out>.previous`` and the staging directory (fix round T6j)."""
    if os.path.isdir(path) and not os.path.islink(path):
        shutil.rmtree(path)
    elif os.path.lexists(path):
        os.remove(path)


def _discard_staging(tmp_out: str) -> bool:
    """Remove the staging directory after a failure before the swap (fix round T6j); True when it is gone. Errors while
    removing are swallowed: the failure that brought us here is the one to report."""
    shutil.rmtree(tmp_out, ignore_errors=True)
    if os.path.lexists(tmp_out):
        with contextlib.suppress(OSError):
            os.remove(tmp_out)
    return not os.path.lexists(tmp_out)


def complete_marked(path: str) -> tuple[bool, str]:
    """``(True, identity digest)`` when ``path`` is a complete marked tree: a ``.materialized`` marker that parses to an identity
    record and a shard inventory whose shards are on disk unchanged (``inventory_mismatch``), whatever the identity; else
    ``(False, why)``. The rule of fix round T6k (review R149H-01): such a tree is never discarded while it is the only one."""
    marker_path = os.path.join(path, MATERIALIZED_MARKER)
    if not os.path.isfile(marker_path):
        return False, f"{marker_path} missing"
    try:
        marker = _load_json(marker_path)
    except (OSError, ValueError) as error:
        return False, f"{marker_path} unreadable ({error})"
    identity = marker.get("identity") if isinstance(marker, dict) else None
    inventory = marker.get("shards") if isinstance(marker, dict) else None
    if not isinstance(identity, dict) or not isinstance(inventory, list) or not inventory:
        return False, f"{marker_path} has no identity record or no shard inventory"
    bad = inventory_mismatch(path, inventory)
    if bad:
        return False, bad
    return True, identity_digest(identity)[:12]


def _lock_holder(fd: int) -> str:
    """The holder record written into the lock file by ``output_lock`` (pid, host, start time, source), as a phrase."""
    try:
        raw = os.pread(fd, 4096, 0)
        record = json.loads(raw.decode("utf-8")) if raw.strip() else {}
    except (OSError, ValueError):
        record = {}
    if not isinstance(record, dict) or "pid" not in record:
        return " (no holder record in the lock file)"
    return (
        f" held by pid {record.get('pid')} on host {record.get('host')} since {record.get('started')}"
        f" for source {record.get('source')}"
    )


@contextlib.contextmanager
def output_lock(out: str, source: str | None = None, wait_seconds: float | None = None) -> Iterator[str]:
    """The per-output exclusive lock of fix round T6k (review R149H-02): ``<out>.lock`` beside the output, ``fcntl.flock(LOCK_EX |
    LOCK_NB)`` on it for the whole build path (from before ``<out>.previous`` / ``<out>.tmp`` / the fetch stage are touched to after
    the swap and the removal of ``<out>.previous``). A second materializer on the same ``--out`` raises ``MaterializeRefusedError``
    naming the holder (pid, host, start time, source from the record the holder wrote) and touches nothing. flock is released by the
    kernel when the holder exits, so a held lock means a live holder; the lock file is never unlinked (a waiter that already opened
    it would otherwise lock a dead inode while a newcomer creates a new one) and is not part of the export. A file system that
    refuses flock (``ENOLCK`` / ``EOPNOTSUPP`` / ``ENOSYS``: Lustre without the ``flock`` mount option) is a named refusal, fail
    closed. Fix round T6n (review R149K-02): a held lock is WAITED for, polling every ``LOCK_POLL_SECONDS`` up to ``wait_seconds``
    (default ``LOCK_WAIT_SECONDS`` = 120 s; the CLI's ``--lock-wait-seconds``; 0 = refuse at once), so two simultaneous identical
    requests of a published source both end with 0 (the second skips once the holder has finished) and a differing request is
    refused on the marker under the lock; a holder still there after the bound is refused by name, the bound stated. Yields the
    lock path."""
    lock_path = out + LOCK_SUFFIX
    wait = (
        LOCK_WAIT_SECONDS if wait_seconds is None else (float(wait_seconds) or 0.0)
    )  # -0.0 renders as 0 (fix round T6p)
    if not (math.isfinite(wait) and wait >= 0):  # inf would wait forever, nan would silently mean 0 (fix round T6o)
        raise ValueError(f"wait_seconds must be a finite number >= 0, got {wait_seconds!r}")
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        deadline = time.monotonic() + wait
        waited = False
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise MaterializeRefusedError(
                        f"REFUSED: {out} is being materialized by another process ({lock_path}{_lock_holder(fd)}) and the lock was "
                        f"not released within {wait:g} s (--lock-wait-seconds); nothing was touched; wait for it to finish (the lock "
                        f"is released when its holder exits), raise the bound, or check that process"
                    ) from None
                if not waited:
                    waited = True
                    _log(f"waiting up to {wait:g} s for {lock_path}{_lock_holder(fd)}")
                time.sleep(LOCK_POLL_SECONDS)
            except OSError as error:
                raise MaterializeRefusedError(
                    f"REFUSED: cannot take the per-output lock {lock_path} ({type(error).__name__}: {error}); the output's file system "
                    f"must support flock (Lustre: the flock mount option) so that two materializers on one --out exclude each other; "
                    f"nothing was touched"
                ) from error
        record = {
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "out": out,
            "source": source,
        }
        os.ftruncate(fd, 0)
        os.write(fd, json.dumps(record).encode("utf-8"))
        os.fsync(fd)
        yield lock_path
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _repair_staged_config(tmp_out: str, base: str) -> bool:
    """Fix round T6m: an inherited complete marked staging copy may lack ``config.json`` (the file ``write_config_from_base`` rewrites
    last before the marker; a death or a hand removal can leave the shards and the marker without it). Copy the base's ``config.json``
    in through a temporary file and a rename (the base is the one the marker's identity names: same base digest) so the recovery's
    ``write_config_from_base`` can validate the identity; nothing else is touched. Returns True when it wrote the file."""
    destination = os.path.join(tmp_out, "config.json")
    if os.path.isfile(destination):
        return False
    part = destination + ".copying"
    shutil.copy2(os.path.join(base, "config.json"), part)
    os.replace(part, destination)
    return True


def digest_open_file(payload: BinaryIO) -> dict[str, Any]:
    """``{"sha256", "bytes"}`` of the file ``payload`` has open, read sequentially through that file object from offset 0 (8 MiB
    chunks; fix round T6r, review thread 524355e746): the identity of the inode the caller holds open, not of whatever its path
    names now. ``bytes`` is what was read; the caller compares it with ``fstat`` and refuses a file that grew or shrank meanwhile."""
    digest = hashlib.sha256()
    payload.seek(0)
    total = 0
    while chunk := payload.read(8 * 1024 * 1024):
        digest.update(chunk)
        total += len(chunk)
    payload.seek(0)
    return {"sha256": digest.hexdigest(), "bytes": total}


def inode_signature(st: os.stat_result) -> tuple[int, int, int]:
    """``(inode, size, mtime_ns)`` of an fstat'd open file: what a write INTO that inode changes. Not ctime: the merge tool's rebuild
    unlinks the path before writing the new file (``merge.py``: ``os.remove``, then tmp + ``os.replace``), and unlinking the inode a
    reader holds open moves its link count and so its ctime without touching one byte of it (the first T6r candidate included ctime
    and refused the reviewer's own scenario on it, on Lustre); a replacement of the PATH never matters to a reader holding the inode
    open. Fix round T6r."""
    return (st.st_ino, st.st_size, st.st_mtime_ns)


def probe_opened_source(payload: BinaryIO, path: str) -> tuple[dict[str, Any], os.stat_result]:
    """Identify the local ``model.pt`` the caller has OPEN (fix round T6r): fstat first, then the hash through the same file object;
    a size that differs between the two means a writer is changing the inode in place and the request is refused by name."""
    opened = os.fstat(payload.fileno())
    _log(f"hashing {path} through the open file ({opened.st_size / 2**30:.1f} GiB, inode {opened.st_ino})")
    probe = {"kind": "local", "path": os.path.abspath(path), **digest_open_file(payload)}
    if probe["bytes"] != opened.st_size:
        raise ConversionError(
            f"the source {path} is being written in place: {opened.st_size} bytes when opened, {probe['bytes']} bytes read through "
            f"the open file; the request is refused; re-run it when the writer has finished"
        )
    return probe, opened


def _refuse_if_opened_inode_changed(
    payload: BinaryIO, opened: os.stat_result, model_pt: str, probe: dict[str, Any]
) -> None:
    """Fix round T6r (review thread 524355e746): the overlay memory-mapped the file object ``payload``, the same inode the identity of
    record was hashed from, so a replacement of the PATH meanwhile changed nothing of what was built (logged, with the inode that
    now stands at the path). What CAN change what a mapping reads is a write to that inode itself: ``fstat`` after the overlay
    must equal ``fstat`` before the hash (inode, size, mtime; ``inode_signature``); any difference refuses the build by name, with
    no "same bytes" acceptance, because the digest of the file after the fact does not say which bytes the mapping consumed. Best
    effort: the attributes a client reports for an open inode may lag a write into it (Lustre reported no change for a 16-byte
    in-place write within 20 ms, node-local storage moved mtime_ns); the supported merge tool never writes in place, so the
    guarantee of record is against a replaced PATH, by construction; an in-place writer needs a lock it honours."""
    now = os.fstat(payload.fileno())
    if inode_signature(now) != inode_signature(opened):
        raise ConversionError(
            f"the source {model_pt} (inode {opened.st_ino}, sha256 {probe['sha256'][:12]} when hashed under the lock) was written in "
            f"place while the overlay mapped it: size {opened.st_size} -> {now.st_size}, mtime_ns {opened.st_mtime_ns} -> "
            f"{now.st_mtime_ns}; what the overlay consumed cannot be tied to one source and the staged copy is not published; re-run "
            f"the request"
        )
    try:
        at_path = os.stat(model_pt).st_ino
    except FileNotFoundError:
        at_path = None
    if at_path != opened.st_ino:
        _log(
            f"note: {model_pt} was {'removed' if at_path is None else f'replaced (inode {at_path} stands there now)'} while the "
            f"overlay ran; the build consumed the file opened under the lock (inode {opened.st_ino}, sha256 {probe['sha256'][:12]}) "
            f"and the marker records that identity"
        )


def publish_staging(tmp_out: str, out: str) -> str | None:
    """The swap (fix round T6j): a leftover ``<out>.previous`` (kept until now under ``--force``, fix round T6k) removed first,
    ``out`` -> ``<out>.previous`` when ``out`` exists, ``tmp_out`` -> ``out``, then ``<out>.previous`` removed; returns the removed
    path, or None when there was no previous output. Two ``os.rename`` on one file system (Lustre has neither RENAME_NOREPLACE nor
    RENAME_EXCHANGE). A failure of the first rename leaves both directories as they are; a failure of the second puts the old
    output back under its name before the error propagates; a failure to remove ``<out>.previous`` after the second rename is
    raised with ``out`` already published (the next request skips on the marker and reports the leftover; ``--force`` removes it
    at its own swap). In every failure the staging directory stays, complete and marked for the request: a re-run with
    ``--force`` puts the old output back under its name when it is absent, discards the staging directory and rebuilds from
    the source (fix round T6l); without ``--force`` the request decides on ``out``'s marker (skip, or refused while it holds
    another materialization). Every failure is a ``ConversionError``
    naming the state. Called with ``<out>.lock`` held."""
    previous = out + PREVIOUS_SUFFIX
    had_output = os.path.lexists(out)
    kept_note = (
        f"the staging directory {tmp_out} is kept (complete and marked for this request); a re-run with --force puts the old "
        f"output back under its name when it is absent, discards this staging directory and rebuilds from the source; a re-run "
        f"without --force decides on the marker of {out} (skip, or refused while it holds another materialization)"
    )
    try:
        if os.path.lexists(previous):
            # A leftover of an interrupted swap, kept until the new tree was verified (R149H-01): it may go now.
            _remove_path(previous)
        if had_output:
            os.rename(out, previous)
    except OSError as error:
        raise ConversionError(
            f"the swap failed before anything moved: {type(error).__name__}: {error}; the existing output {out} is unchanged "
            f"and {kept_note}"
        ) from error
    try:
        os.rename(tmp_out, out)
    except OSError as error:
        restored = False
        if had_output:
            with contextlib.suppress(OSError):
                os.rename(previous, out)
                restored = True
        where = (
            f"the existing output is back under its name {out}"
            if restored
            else f"the existing output is at {previous} (--force renames it back to {out} before building)"
            if had_output
            else f"no output exists at {out}"
        )
        raise ConversionError(f"the swap failed: {type(error).__name__}: {error}; {where}; {kept_note}") from error
    if had_output:
        try:
            _remove_path(previous)
        except OSError as error:
            raise ConversionError(
                f"{out} is published (complete and marked); removing the previous output {previous} failed: "
                f"{type(error).__name__}: {error}; remove it by hand, or the next request with --force removes it at its swap"
            ) from error
        return previous
    return None


def run_request(
    *,
    base: str,
    out: str,
    ckpt_dir: str | None = None,
    artifact_uri: str | None = None,
    stage_dir: str | None = None,
    force: bool = False,
    min_tensors: int = DEFAULT_MIN_TENSORS,
    expect_tensors: int = 0,
    keep_staged: bool = False,
    verify: bool = True,
    credentials: str | None = None,
    unified_chat_template: bool = False,
    lock_wait_seconds: float | None = None,
    output_dtype: str | None = None,
) -> str:
    """Idempotent entry point: ``"skipped"`` (``out`` provably holds this request) or ``"built"``.

    Raises ``ConversionError`` in the preflight's one family (``REFUSED before anything is published: ...``) when the base
    is refused before anything is written, ``MaterializeRefusedError`` when ``out`` holds a different or unprovable
    materialization and ``force`` is false (decided under the output lock, fix round T6m; or a leftover ``<out>.previous`` exists
    without ``force``, or an inherited sole complete marked ``<out>.tmp`` is not this request's, or another materializer
    holds ``<out>.lock`` beyond ``lock_wait_seconds``, or a complete marked ``<out>.tmp`` is the only complete tree),
    ``ConversionError``
    naming the stage when the staged build or a verification fails (a fresh build's staging directory removed, an existing
    ``out`` untouched; fix round T6j) or when the recovery of an inherited sole complete marked ``<out>.tmp`` fails (the directory
    kept: its shards and marker untouched, its sidecars possibly repaired from the base; fix rounds T6m / T6n) or when the swap
    fails, ``RuntimeError``
    when the archive fetch refuses (a foreign staged copy, an unprovable payload), ``FileNotFoundError`` / ``ValueError``
    for an invalid request (no ``model.pt`` in ``ckpt_dir``, a malformed ``artifact_uri``).
    """
    if bool(ckpt_dir) == bool(artifact_uri):
        raise ValueError("pass exactly one of ckpt_dir or artifact_uri")
    if output_dtype not in (None, "float32"):
        raise ValueError("explicit merged export output_dtype currently supports float32 only")
    if lock_wait_seconds is not None and not (math.isfinite(lock_wait_seconds) and lock_wait_seconds >= 0):
        raise ValueError(f"lock_wait_seconds must be a finite number >= 0, got {lock_wait_seconds!r}")
    out = os.path.abspath(out.rstrip("/"))
    # Every torch-free refusal that reads the base alone, before anything destructive (fix round T6f): nothing below
    # removes or writes until it has passed.
    base_id = preflight(base, unified_chat_template=unified_chat_template)["base"]
    # Identify the requested source WITHOUT fetching: hash a local model.pt, or read the archived manifest. The skip
    # below must prove that `out` was built from THIS source, THIS base and THIS conversion; --no-verify never skips that.
    # The source is identified twice (fix rounds T6q and T6r, review thread 524355e746): once here, before the lock, so a missing
    # model.pt or an unprovable artifact is refused without waiting, and once more UNDER the lock through the file object that the
    # overlay then maps, which is the identity the decision, the source record and the marker use: the lock may wait up to
    # --lock-wait-seconds and the merge tool (tmp + os.replace over model.pt) is not excluded by it, so the path may change while
    # the request waits or while the overlay runs.
    probe_before = probe_local_source(ckpt_dir) if ckpt_dir else probe_archived_source(str(artifact_uri), credentials)
    want = {
        "conversion": CONVERSION_VERSION,
        "base_digest": base_id["digest"],
        "base_path": base_id["path"],
        "unified_chat_template": unified_chat_template,
        "output_dtype": output_dtype,
    }
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with (
        output_lock(out, source=ckpt_dir or artifact_uri, wait_seconds=lock_wait_seconds),
        contextlib.ExitStack() as held,
    ):
        # Fix rounds T6q / T6r: the identity of record is read here, under the lock. A local model.pt is OPENED ONCE (closed when
        # the lock is released): fstat, then the hash through that file object, and the overlay below maps the same file object,
        # so the marker names the bytes the overlay consumed by construction; an archived artifact's fetched copy is opened and
        # hashed the same way after the fetch and must match the manifest identity.
        payload: BinaryIO | None = None
        opened: os.stat_result | None = None
        if ckpt_dir:
            payload = held.enter_context(open(os.path.join(ckpt_dir, PAYLOAD), "rb"))
            probe, opened = probe_opened_source(payload, os.path.join(ckpt_dir, PAYLOAD))
        else:
            probe = probe_archived_source(str(artifact_uri), credentials)
        if (probe["sha256"], probe["bytes"]) != (probe_before["sha256"], probe_before["bytes"]):
            _log(
                f"WARNING: the source {probe.get('path') or probe.get('artifact_uri')} changed while this request waited for the lock: "
                f"sha256 {probe_before['sha256'][:12]} ({probe_before['bytes']} bytes) before, {probe['sha256'][:12]} "
                f"({probe['bytes']} bytes) now; the request proceeds with the bytes it reads now, and the marker records them"
            )
        # Fix round T6m (review R149J-02 / thread b15dc9ef1b): the marker decision (skip / mismatch / build) is taken UNDER the
        # per-output lock, so decision and publication happen in one lock hold; a materializer that paused before the lock
        # while another one published a different source now sees that export's marker and is refused without --force.
        status, why = marker_status(out, want, probe)
        if status == "match" and not force:
            report = verify_index(out, expect_tensors=expect_tensors, min_tensors=min_tensors)
            report.update(verify_sidecars(out, base, unified_chat_template=unified_chat_template))
            if output_dtype is not None:
                report.update(hf.verify_export_dtype(out, base, output_dtype))
            _log(
                f"already materialized: {out}: {why}; {report['n_tensors']} tensors, {report['sidecars_identical']} sidecars re-checked"
            )
            if os.path.lexists(out + PREVIOUS_SUFFIX):
                _log(
                    f"WARNING: leftover {out + PREVIOUS_SUFFIX} beside the proven output (an interrupted swap, or a directory parked "
                    f"under that name); left alone; a rebuild with --force discards it"
                )
            return "skipped"
        if status == "mismatch" and not force:
            raise MaterializeRefusedError(
                f"REFUSED: {out} holds a different or unproven materialization: {why}. Pass --force to rebuild it from the "
                f"requested source, or choose another --out"
            )
        # Nothing is removed here (fix round T6j): the build is staged in <out>.tmp beside an existing output, which the swap
        # below replaces only after the staged build and its verification have succeeded. Fix round T6k: no complete marked
        # tree is discarded while it is the only one (R149H-01), and one materializer per output (R149H-02, the lock).
        previous = out + PREVIOUS_SUFFIX
        if os.path.lexists(previous):
            if not force:
                raise MaterializeRefusedError(
                    f"REFUSED: {previous} exists (the previous output of an interrupted swap, or a directory parked under the "
                    f"swap's name); remove it, or pass --force: with {out} absent --force renames it back to {out} first and builds "
                    f"beside it; with {out} present --force removes it at the swap, after the new build is verified"
                )
            if not os.path.lexists(out):
                _log(
                    f"--force: renaming {previous} back to {out} (the old output of an interrupted swap) before building beside it"
                )
                os.rename(previous, out)
            else:
                _log(
                    f"--force: leftover {previous} kept until the swap, where it is removed after the new build is verified"
                )
        replaces = f"{'existing (--force)' if force else status} output {out}: {why}" if os.path.lexists(out) else None
        if replaces:
            _log(f"building beside the {replaces}; it is kept until the staged build is verified and swapped in")

        staged: StagedSource | None = None
        if ckpt_dir:
            source_dir = ckpt_dir
        else:
            staged = fetch_archived_source(probe, stage_dir or out + ".stage", credentials)
            source_dir = staged.artifact_dir
        model_pt = os.path.join(source_dir, PAYLOAD)
        if (
            payload is None
        ):  # an archived artifact: open the fetched copy once; its bytes must be the manifest's identity
            payload = held.enter_context(open(model_pt, "rb"))
            fetched, opened = probe_opened_source(payload, model_pt)
            if (fetched["sha256"], fetched["bytes"]) != (probe["sha256"], probe["bytes"]):
                raise ConversionError(
                    f"the fetched copy {model_pt} (sha256 {fetched['sha256'][:12]}, {fetched['bytes']} bytes) is not the artifact the "
                    f"manifest identifies (sha256 {probe['sha256'][:12]}, {probe['bytes']} bytes); not materialized"
                )
        assert opened is not None
        source = {
            "kind": probe["kind"],
            "path": os.path.abspath(model_pt),
            "sha256": probe["sha256"],
            "bytes": probe["bytes"],
            "artifact_uri": probe.get("artifact_uri"),
            "manifest_sha256": probe.get("manifest_sha256"),
        }
        identity = make_identity(
            source, base_id, unified_chat_template=unified_chat_template, output_dtype=output_dtype
        )
        _log(f"identity {identity_digest(identity)[:12]}: {_short(identity)}")
        tmp_out = out + STAGING_SUFFIX

        started = time.time()
        if force and os.path.lexists(tmp_out):
            # Fix round T6l (review R149I-01): --force means a rebuild from the source. A death-left staging directory is discarded
            # whatever its marker says once <out> holds a complete marked tree (after the restore above, or when it stood); a
            # complete marked staging directory that is the only complete tree is not discarded (fix round T6k's rule): a request
            # WITHOUT --force publishes it after verification (tmp_complete below), or it is removed by hand.
            marked, marked_why = complete_marked(tmp_out)
            if marked:
                out_marked, out_why = complete_marked(out)
                if not out_marked:
                    raise MaterializeRefusedError(
                        f"REFUSED: {tmp_out} is a complete marked export (identity {marked_why}) and no complete marked export stands "
                        f"at {out} ({out_why}); --force does not discard the only complete tree: a request without --force publishes "
                        f"it after verification, or remove it by hand, before rebuilding with --force"
                    )
                _log(
                    f"--force: discarding staging dir {tmp_out} (complete and marked, identity {marked_why}): --force rebuilds from "
                    f"the source; {out} keeps a complete marked export ({out_why})"
                )
            else:
                _log(f"--force: discarding staging dir {tmp_out} ({marked_why})")
            _remove_path(tmp_out)
        # Fix round T6m (review R149J-01 / thread 677171a12b): an INHERITED sole complete marked staging copy (a first publication
        # that died after writing .materialized and before the rename; <out> holds no complete marked export, else --force would be
        # the way and the unforced request would have skipped or been refused above) is recovered under protection: its marker must
        # be this request's (else refused by name), a missing staged config.json is repaired from the base through a temporary file
        # and a rename, the sidecars are re-synced the same way (never a shard), then verification, the marker and the swap; a
        # failure anywhere refuses by name and leaves the tree as found. The overlay is never run while it is the sole complete tree.
        recovery = False
        if not force and complete_marked(tmp_out)[0]:
            tmp_status, tmp_why = marker_status(tmp_out, want, probe)
            if tmp_status != "match":
                raise MaterializeRefusedError(
                    f"REFUSED: {tmp_out} is the only complete marked export here and it is not this request's ({tmp_why}); it is "
                    f"not discarded: publish it with a request of its identity, or remove it by hand"
                )
            recovery = True
            reuse, why = (
                True,
                f"inherited sole complete marked copy of this request ({tmp_why}); recovered, never deleted",
            )
        else:
            reuse, why = tmp_complete(tmp_out, identity)
        stage = "staging"
        try:
            if recovery:
                stage = "sidecar repair"
                repaired = _repair_staged_config(tmp_out, base)
                _log(
                    f"recovering inherited staging dir {tmp_out}: {why}; config.json {'repaired from the base' if repaired else 'present'}"
                )
            elif reuse:
                _log(f"reusing staging dir {tmp_out}: {why}; skipping the overlay")
            else:
                if os.path.exists(tmp_out):
                    _log(f"staging dir {tmp_out} not reusable ({why}); rebuilding")
                    shutil.rmtree(tmp_out)
                stage = "overlay"
                _log(
                    f"materialize {model_pt} ({probe['bytes'] / 2**30:.1f} GiB, inode {opened.st_ino}) onto {base} -> {tmp_out}"
                )
                hf.overlay_consolidated_onto_hf(source_dir, base, tmp_out, payload=payload)
                stage = "source identity after the overlay"
                _refuse_if_opened_inode_changed(payload, opened, model_pt, probe)
                transformers = sys.modules.get("transformers")
                atomic_write_json(
                    os.path.join(tmp_out, SOURCE_STAMP),
                    {
                        "identity": identity,
                        "source": source,
                        "base": base_id,
                        "transformers_version": getattr(transformers, "__version__", None),
                        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    },
                )
                _log(f"materialized in {(time.time() - started) / 60:.1f} min")
            stage = "sidecar sync"
            sync = sync_sidecars(base, tmp_out)
            # Idempotent on a reused staging dir; returns what config.json is and where the transformers-written one is kept.
            stage = "config.json <- base"
            sync["config"] = hf.write_config_from_base(
                tmp_out, base, **({"output_dtype": output_dtype} if output_dtype is not None else {})
            )
            if unified_chat_template:
                # After the sync, which re-copied the base's tokenizer_config.json (on a reused staging dir too): the one ruled
                # sidecar difference, tokenizer_config.json's chat_template = the processor template's text.
                stage = "unified chat template"
                sync["chat_template"] = hf.write_unified_chat_template(tmp_out, base)
            template_note = (
                "<- the processor template (--unified-chat-template)" if unified_chat_template else "= the base's"
            )
            _log(
                f"sidecars <- base: overwrote {sync['copied']}; removed shadowing {sync['removed'] or 'none'}; "
                f"config.json <- base ({sync['config']}); tokenizer chat_template {template_note}; verifying {tmp_out}"
            )
            stage = "template record"
            stamp = _load_json(os.path.join(tmp_out, SOURCE_STAMP))
            report: dict[str, Any] = {
                "protocol": MARKER_PROTOCOL,
                "identity": identity,
                "identity_digest": identity_digest(identity),
                "source": source,
                "base": base_id,
                "transformers_version": stamp.get("transformers_version"),
                "reused_staging": reuse,
                "sidecars": sync,
                "chat_template": hf.chat_template_record(tmp_out, base, unified_chat_template=unified_chat_template),
                "options": {
                    "expect_tensors": expect_tensors,
                    "min_tensors": min_tensors,
                    "no_verify": not verify,
                    "unified_chat_template": unified_chat_template,
                },
                "materialized_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            stage = "verification (index)"
            report["verify"] = verify_index(tmp_out, expect_tensors=expect_tensors, min_tensors=min_tensors)
            stage = "verification (sidecars)"
            report["verify"].update(verify_sidecars(tmp_out, base, unified_chat_template=unified_chat_template))
            if output_dtype is not None:
                report["verify"].update(hf.verify_export_dtype(tmp_out, base, output_dtype))
            report["verify"]["torch"] = verify
            if verify:
                stage = "verification (torch)"
                report["verify"].update(
                    verify_with_torch(
                        tmp_out, base, model_pt, unified_chat_template=unified_chat_template, payload=payload
                    )
                )
            stage = "shard inventory"
            report["shards"] = shard_inventory(tmp_out)
            stage = "marker"
            atomic_write_json(os.path.join(tmp_out, MATERIALIZED_MARKER), report)
        except Exception as error:
            if recovery:
                # Fix round T6m: the inherited sole complete marked copy is never deleted; the failed check refuses by name. Fix round
                # T6n (review R149K-01): the words say what is true: the repair and the sidecar sync ran BEFORE the check, so the
                # sidecars may differ from what was found while the shards and the marker are untouched; the state at <out> is named.
                out_marked, out_why = complete_marked(out)
                out_state = (
                    f"no output exists at {out}"
                    if not os.path.lexists(out)
                    else f"a complete marked export stands at {out} (identity {out_why})"
                    if out_marked
                    else f"{out} stands but is not a complete marked export ({out_why})"
                )
                raise ConversionError(
                    f"the recovery of the inherited staging directory {tmp_out} failed at the {stage} stage: "
                    f"{type(error).__name__}: {error}; the directory is kept: its shards and its marker are untouched, its sidecars may "
                    f"have been repaired from the base before the check (config.json rewritten from the base, a differing one kept as "
                    f"config.json{hf.TRAINER_BACKUP_SUFFIX}, a sidecar the base lacks removed, a .copying or .writing temporary left by "
                    f"an interrupted copy or unified-template rewrite, and under --unified-chat-template the two template files back "
                    f"in the base's form when that rewrite did not reach its rename); {out_state}; nothing was published; fix the "
                    f"cause and retry without --force, or remove the directory by hand"
                ) from error
            # Fix round T6j: nothing of the existing output has been touched; the staging directory goes, the stage is named.
            gone = _discard_staging(tmp_out)
            kept = f"; the existing output {out} was not touched by this request" if replaces else ""
            raise ConversionError(
                f"the staged build failed at the {stage} stage: {type(error).__name__}: {error}; the staging directory {tmp_out} "
                f"{'was removed' if gone else 'could not be removed'}{kept}; nothing was published"
            ) from error
        if replaces:
            _log(f"swap: replacing the {replaces}")
        publish_staging(tmp_out, out)
        _log(
            f"OK -> {out} ({report['verify']['bytes'] / 2**30:.1f} GiB, {report['verify']['n_tensors']} tensors in "
            f"{len(report['shards'])} shards, {report['verify']['sidecars_identical']} sidecars identical to base, identity "
            f"{report['identity_digest'][:12]})"
        )
        if staged is not None and not keep_staged:
            removed = cleanup_staged(staged)
            _log(
                f"removed staged {removed} (the archive is the copy of record); nothing else under {staged.stage_dir} was touched"
            )
        return "built"


def main(argv: Sequence[str] | None = None) -> int:
    """Materialize one consolidated checkpoint into an HF directory; 0 when built or skipped, 1 when refused or failed."""
    parser = argparse.ArgumentParser(
        prog="python -m cosmos_framework.posttrain.checkpoint.tools.materialize_hf",
        description="Identity-bound, idempotent materialization of a consolidated model.pt into a Hugging Face teacher dir.",
    )
    parser.add_argument(
        "--ckpt-dir", help="local directory holding a consolidated model.pt (trainer or merge artifact)"
    )
    parser.add_argument(
        "--artifact-uri",
        help="s3://bucket/<...>/checkpoint_artifacts/iter_* of an archived artifact (fetched only when a build is needed)",
    )
    parser.add_argument("--base", help="HF base model directory the state dict overlays")
    parser.add_argument(
        "--out",
        help="HF output dir (built as <out>.tmp beside an existing output and swapped in by renames after the staged build "
        "and its verification succeed; skipped when its marker proves the same source, base and conversion; refused when it "
        "proves a different one; a leftover <out>.previous is refused without --force)",
    )
    parser.add_argument(
        "--stage-dir",
        help="where an archived artifact is fetched (default <out>.stage); only files this invocation creates there are "
        "removed afterwards",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="rebuild from the source even if --out is already materialized or holds a different materialization; the torch-free "
        "preflight on the base runs first and the build is staged beside the existing output, which is replaced by a rename swap "
        "only after the staged build and its verification have succeeded; a failure before the swap removes the staging directory "
        "and leaves the existing output byte-identical (transient disk: two exports); a death-left <out>.tmp is discarded once <out> "
        "is complete again (a kept staged copy is published only by a request without --force); with <out> absent a leftover "
        "<out>.previous is renamed back to <out> first, with <out> present it is removed at the swap",
    )
    parser.add_argument(
        "--min-tensors", type=int, default=DEFAULT_MIN_TENSORS, help="floor on the tensor count in the output index"
    )
    parser.add_argument(
        "--expect-tensors", type=int, default=0, help="exact tensor count expected (0 = skip the exact count)"
    )
    parser.add_argument("--keep-staged", action="store_true", help="keep what this run fetched into --stage-dir")
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="skip the torch-side tensor probing only (identity, index and sidecar checks still run)",
    )
    parser.add_argument(
        "--unified-chat-template",
        action="store_true",
        help="write both template files of the export (tokenizer_config.json's and chat_template.json's chat_template) as "
        "the base's processor template in its canonical form (processor-canonical: the reasoning branch rendered trimmed), "
        "every other key unchanged, so a request with or without tools renders the history reasoning in the student's "
        "emission form; recorded in .materialized and part of the output's identity (default: the base's two templates, "
        "verbatim)",
    )
    parser.add_argument(
        "--credentials", help="object-store credentials JSON for --artifact-uri (default: the framework's)"
    )
    parser.add_argument(
        "--lock-wait-seconds",
        type=float,
        default=LOCK_WAIT_SECONDS,
        help="how long to wait for another materializer's <out>.lock before refusing by name (default 120; 0 = refuse at once; a finite "
        "number >= 0: inf and nan are refused); an "
        "identical request skips with exit 0 once the holder has finished, a differing one is refused on the output's marker",
    )
    a = parser.parse_args(argv)
    error_message = validate_args(a)
    if error_message:
        parser.error(error_message)
    out = os.path.abspath(a.out.rstrip("/"))
    try:
        result = run_request(
            base=a.base,
            out=out,
            ckpt_dir=a.ckpt_dir,
            artifact_uri=a.artifact_uri,
            stage_dir=a.stage_dir,
            force=a.force,
            min_tensors=a.min_tensors,
            expect_tensors=a.expect_tensors,
            keep_staged=a.keep_staged,
            verify=not a.no_verify,
            credentials=a.credentials,
            unified_chat_template=a.unified_chat_template,
            lock_wait_seconds=a.lock_wait_seconds,
        )
    except (ConversionError, OSError, RuntimeError, ValueError) as error:
        # OSError covers FileNotFoundError and, after the preflight, any unreadable path met later: a named FATAL line,
        # never a traceback (review R149D-02).
        print(f"[materialize] FATAL: {error}", file=sys.stderr, flush=True)
        return 1
    print("MATERIALIZE DONE", out, *(["(idempotent skip)"] if result == "skipped" else []), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
