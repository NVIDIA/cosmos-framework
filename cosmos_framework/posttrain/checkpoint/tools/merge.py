# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Streaming average / task-vector merge of consolidated ``model.pt`` experts into one native checkpoint artifact.

Every expert is a consolidated ``model.pt`` (``torch.save({"net": state_dict})``, bf16, Hugging Face key names).
The output is one ``model.pt`` of the same format next to the native ``.posttrain-checkpoint-artifact.json``
manifest, so a merged artifact reads exactly like a trainer-written ``checkpoint_artifacts/iter_*`` directory
(``artifact.read_local_artifact_manifest`` accepts it). Tensors stream one key at a time: never more than about
two fp32 tensors plus the bf16 output dict are resident.

  avg : W = mean_t W_t                                (task arithmetic with lambda = 1/N; the base cancels)
  tv  : W = W_base + lam * sum_t (W_t - W_base)       (lam default 1.0; needs ``--base``)

``--out`` must be ``<...>/iter_*/model.pt``: the native manifest binds the iteration to the directory name, and an
explicit ``--iteration`` must agree with it.

Idempotency (safe under job requeue). Every run first derives the MERGE IDENTITY of the request = {tool version,
mode, lam, iteration, ordered experts as (realpath, sha256, bytes), base as (realpath, digest of the index bytes plus
the shard inventory), expected key set (count plus sha256 of the sorted names)}. The artifact directory is written
transactionally: ``model.pt`` (tmp + rename, reload-verified, sha256), then ``--diag``, then the artifact manifest,
and LAST the completion record ``.merge-complete.json`` holding the identity, its digest, the merge request and the
artifact inventory (names, bytes, payload sha256). A later run

* SKIPS only when the record is present, its identity digest equals the request's, every recorded artifact is on
  disk with the recorded size, the payload sha256 re-verifies and a requested ``--diag`` exists;
* REBUILDS when something is on disk without a record (crash after the payload or manifest write), the record is
  unreadable or carries no identity, or the inventory is damaged (missing or truncated payload or manifest, requested
  ``--diag`` absent). The stale record and manifest are removed first, so a crash mid-rebuild leaves no false proof;
* REFUSES (``MergeRefusedError``; exit 1 from the CLI) when a completed merge of a DIFFERENT request lives at
  ``--out`` (other experts or expert contents, mode, lam, base or key set); ``--force`` rebuilds over it.

Known answers live in ``tests/test_checkpoint_merge.py`` (synthetic two-shard base plus three experts with deltas
that are exact in bf16). torch and safetensors are imported at call time only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sys
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

from cosmos_framework.posttrain.checkpoint.artifact import CHECKPOINT_ARTIFACT_MANIFEST, build_artifact_manifest
from cosmos_framework.posttrain.checkpoint.io import atomic_write_json
from cosmos_framework.posttrain.checkpoint.manifest import iteration_name, iteration_number
from cosmos_framework.posttrain.utils import object_store
from cosmos_framework.posttrain.utils.log import logger

if TYPE_CHECKING:
    import torch

TOOL_VERSION = "merge_model_pt.v3"  # bump whenever the identity schema, the emitted files or the arithmetic change
MERGE_COMPLETE = ".merge-complete.json"  # written LAST: a model.pt without it is an unfinished merge
MERGE_COMPLETE_PROTOCOL = "merge_model_pt.complete.v1"
MODES = ("avg", "tv")


class MergeRefusedError(RuntimeError):
    """A completed merge of a different request occupies the output directory; ``force`` rebuilds over it."""


def _log(msg: str) -> None:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    logger.info(f"[merge rss={rss:.1f}G] {msg}")


def load_expert(path: str) -> dict[str, torch.Tensor]:  # values: [*shape]
    """Memory-map one consolidated ``model.pt`` and return its ``net`` state dict."""
    import torch

    obj = torch.load(path, mmap=True, map_location="cpu", weights_only=False)  # net values: [*shape]
    if not isinstance(obj, dict) or set(obj) != {"net"}:
        found = set(obj) if isinstance(obj, dict) else type(obj)
        raise ValueError(f"{path}: top-level keys {found} != {{'net'}}")
    return obj["net"]  # values: [*shape]


class Base:
    """Lazy per-key access to a Hugging Face safetensors directory (``model.safetensors.index.json``)."""

    def __init__(self, hf_dir: str) -> None:
        from safetensors import safe_open

        self._safe_open: Any = safe_open
        with open(os.path.join(hf_dir, "model.safetensors.index.json")) as handle:
            self.map: dict[str, str] = json.load(handle)["weight_map"]
        self.dir: str = hf_dir
        self._handles: dict[str, Any] = {}

    def keys(self) -> set[str]:
        """Return the tensor names the index lists."""
        return set(self.map)

    def get(self, key: str) -> torch.Tensor:  # returns: [*shape]
        """Read one tensor from its shard, keeping the shard handle open for later keys."""
        shard = self.map[key]
        if shard not in self._handles:
            self._handles[shard] = self._safe_open(os.path.join(self.dir, shard), framework="pt", device="cpu")
        return self._handles[shard].get_tensor(key)  # [*shape]


# ---------------------------------------------------------------- merge identity ---------------------------------------
def payload_digest(path: str) -> dict[str, Any]:
    """Return ``{"sha256": hexdigest, "bytes": size}`` of one file (one sequential read through the shared hasher)."""
    return {"sha256": object_store.file_sha256(path), "bytes": os.path.getsize(path)}


def digest_experts(experts: Sequence[str]) -> list[dict[str, Any]]:
    """Ordered ``[{path (realpath), sha256, bytes}]``, hashed in parallel (hashlib releases the GIL on large buffers)."""
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=min(8, len(experts))) as pool:
        digests = list(pool.map(payload_digest, experts))
    out = [{"path": os.path.realpath(p), **d} for p, d in zip(experts, digests)]
    _log(f"hashed {len(out)} experts ({sum(d['bytes'] for d in out) / 2**30:.1f} GiB) in {time.time() - t0:.0f}s")
    return out


def base_identity(base_dir: str) -> dict[str, Any]:
    """Content-defined identity of the HF base: sha256 over the index bytes plus the (shard, bytes) inventory it lists.

    No shard is read; a different base, a re-sharded base and a missing or truncated shard all change the digest.
    """
    idx_path = os.path.join(base_dir, "model.safetensors.index.json")
    with open(idx_path, "rb") as handle:
        idx_bytes = handle.read()
    shards = sorted(set(json.loads(idx_bytes)["weight_map"].values()))
    inventory = []
    for shard in shards:
        path = os.path.join(base_dir, shard)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"base shard missing: {path}")
        inventory.append([shard, os.path.getsize(path)])
    digest = hashlib.sha256(idx_bytes)
    digest.update(json.dumps(inventory).encode())
    return {"path": os.path.realpath(base_dir), "digest": digest.hexdigest(), "n_shards": len(shards)}


def expert_keys(path: str) -> list[str]:
    """Sorted tensor names of one expert."""
    state = load_expert(path)  # values: [*shape]
    keys = sorted(state)
    del state
    return keys


def resolve_iteration(out_path: str, iteration: int | None = None) -> int:
    """Bind the request iteration to the ``iter_*`` directory ``out_path`` lives in.

    The native artifact manifest requires the directory name and the recorded iteration to agree, so an explicit
    ``iteration`` is accepted only when it equals the directory's number.
    """
    name = iteration_name(os.path.dirname(os.path.abspath(out_path)))
    number = iteration_number(name)
    if iteration is not None and int(iteration) != number:
        raise ValueError(f"--iteration {iteration} does not match the output directory {name!r}")
    return number


def request_identity(
    experts: Sequence[str], mode: str, lam: float, base_dir: str | None, iteration: int, keys: Sequence[str]
) -> dict[str, Any]:
    """Everything a skip must reproduce exactly; ``--diag`` is a by-product whose presence is checked separately."""
    return {
        "tool": TOOL_VERSION,
        "mode": mode,
        "lam": float(lam),
        "iteration": int(iteration),
        "experts": digest_experts(experts),
        "base": base_identity(base_dir) if base_dir else None,
        "n_tensors": len(keys),
        "keys_sha256": hashlib.sha256("\n".join(keys).encode()).hexdigest(),
    }


def identity_digest(identity: dict[str, Any]) -> str:
    """Canonical sha256 of one identity record."""
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def identity_diff(want: dict[str, Any], have: dict[str, Any]) -> list[str]:
    """Human-readable field differences between the requested identity and the one recorded on disk."""
    diffs = []
    for k in ("tool", "mode", "lam", "iteration", "n_tensors", "keys_sha256", "base"):
        if want.get(k) != have.get(k):
            diffs.append(f"{k}: requested {want.get(k)!r}, artifact {have.get(k)!r}")
    we, he = want.get("experts") or [], have.get("experts") or []
    if len(we) != len(he):
        diffs.append(f"experts: requested {len(we)}, artifact {len(he)}")
    for i, (a, b) in enumerate(zip(we, he)):
        if a != b:
            diffs.append(
                f"expert[{i}]: requested {a['path']} sha256 {a['sha256'][:12]} ({a['bytes']} B), "
                f"artifact {b.get('path')} sha256 {str(b.get('sha256'))[:12]} ({b.get('bytes')} B)"
            )
    return diffs


def artifact_status(out_path: str, identity: dict[str, Any], diag_path: str | None) -> tuple[str, str]:
    """Decide what the directory next to ``out_path`` proves about THIS request.

    complete   record present, identity digest equal, payload + manifest on disk with the recorded bytes, payload
               sha256 re-verified, requested ``--diag`` present                            -> skip
    incomplete something on disk but no proof the merge finished, or a damaged inventory  -> rebuild
    mismatch   a completed merge of a DIFFERENT request                                    -> refuse (or force)
    absent     nothing on disk                                                             -> build
    """
    d = os.path.dirname(out_path) or "."
    record_path, manifest_path = os.path.join(d, MERGE_COMPLETE), os.path.join(d, CHECKPOINT_ARTIFACT_MANIFEST)
    present = [os.path.basename(p) for p in (out_path, manifest_path, record_path) if os.path.exists(p)]
    if not present:
        return "absent", "nothing on disk"
    if not os.path.exists(record_path):
        return "incomplete", f"{' + '.join(present)} present without {MERGE_COMPLETE}: an earlier merge did not finish"
    try:
        with open(record_path) as handle:
            record = json.load(handle)
    except (OSError, ValueError) as error:
        return "incomplete", f"{MERGE_COMPLETE} unreadable ({error})"
    if not isinstance(record, dict) or record.get("protocol") != MERGE_COMPLETE_PROTOCOL or "identity" not in record:
        protocol = record.get("protocol") if isinstance(record, dict) else None
        return "incomplete", f"{MERGE_COMPLETE} carries no identity record (protocol {protocol!r})"
    want = identity_digest(identity)
    if record.get("identity_digest") != want or identity_digest(record["identity"]) != want:
        return "mismatch", "; ".join(identity_diff(identity, record["identity"])) or "identity digest differs"
    arts = record.get("artifacts") or {}
    payload, manifest = arts.get("payload") or {}, arts.get("manifest") or {}
    for rec, p in ((payload, out_path), (manifest, manifest_path)):
        if not rec or not os.path.isfile(p):
            return "incomplete", f"{os.path.basename(p)} recorded in {MERGE_COMPLETE} is missing"
        if os.path.getsize(p) != rec.get("bytes"):
            return (
                "incomplete",
                f"{os.path.basename(p)} is {os.path.getsize(p)} bytes, {MERGE_COMPLETE} recorded {rec.get('bytes')}",
            )
    got = payload_digest(out_path)["sha256"]
    if got != payload.get("sha256"):
        return (
            "incomplete",
            f"{os.path.basename(out_path)} sha256 {got[:12]} != recorded {str(payload.get('sha256'))[:12]}",
        )
    if diag_path and not os.path.isfile(diag_path):
        return "incomplete", f"--diag {diag_path} requested but absent (only a re-merge produces it)"
    return "complete", f"{MERGE_COMPLETE} matches the request (identity {want[:12]}), payload sha256 verified"


def run_request(
    experts: Sequence[str],
    mode: str,
    lam: float,
    base_dir: str | None,
    out_path: str,
    diag_path: str | None,
    iteration: int | None = None,
    force: bool = False,
) -> str:
    """Idempotent entry point: ``"skipped"`` (a completed merge of exactly this request is on disk) or ``"built"``.

    Raises ``MergeRefusedError`` when a completed merge of a different request lives at ``out_path`` and ``force``
    is false; ``ValueError`` / ``FileNotFoundError`` for an invalid request.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if len(experts) < 2:
        raise ValueError("need at least two experts")
    for p in experts:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"expert missing: {p}")
    if mode == "tv" and not base_dir:
        raise ValueError("--mode tv needs --base")
    out_path = os.path.abspath(out_path)
    diag_path = os.path.abspath(diag_path) if diag_path else None
    iteration = resolve_iteration(out_path, iteration)
    identity = request_identity(experts, mode, lam, base_dir, iteration, expert_keys(experts[0]))
    status, reason = artifact_status(out_path, identity, diag_path)
    if status == "complete":
        _log(f"{out_path}: {reason} -- skipping merge (idempotent)")
        return "skipped"
    if status == "mismatch":
        if not force:
            raise MergeRefusedError(
                f"REFUSED: {out_path} holds a COMPLETED merge of a different request ({reason}). "
                "Choose another --out, or pass --force to rebuild over it."
            )
        _log(f"{out_path}: completed merge of a different request ({reason}); --force -> rebuilding")
    elif status == "incomplete":
        _log(f"{out_path}: {reason} -- rebuilding")
    d = os.path.dirname(out_path)
    # record first: never leave a completion record next to a payload that is being rebuilt
    for p in (
        os.path.join(d, MERGE_COMPLETE),
        os.path.join(d, CHECKPOINT_ARTIFACT_MANIFEST),
        out_path + ".tmp",
        out_path,
    ):
        if os.path.exists(p):
            os.remove(p)
    merge(experts, mode, lam, base_dir, out_path, diag_path, iteration, identity=identity)
    return "built"


# ---------------------------------------------------------------- merge -------------------------------------------------
def merge(
    experts: Sequence[str],
    mode: str,
    lam: float,
    base_dir: str | None,
    out_path: str,
    diag_path: str | None,
    iteration: int | None = None,
    identity: dict[str, Any] | None = None,
) -> None:
    """Merge unconditionally and write the artifact transactionally (payload, diag, manifest, completion record).

    ``run_request`` is the idempotent front door; call this directly only when the output directory is known to hold
    nothing of value.
    """
    import torch

    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    version = tuple(int(part) for part in torch.__version__.split("+")[0].split(".")[:2])
    if version < (2, 1):
        raise RuntimeError(f"torch {torch.__version__} lacks torch.load(mmap=True)")
    _log(f"torch {torch.__version__}")
    n = len(experts)
    if n < 2:
        raise ValueError("need at least two experts")
    out_path = os.path.abspath(out_path)
    diag_path = os.path.abspath(diag_path) if diag_path else None
    iteration = resolve_iteration(out_path, iteration)
    states = [load_expert(p) for p in experts]  # per-expert values: [*shape]
    keys = sorted(states[0])
    for p, state in zip(experts, states):
        if set(state) != set(keys):
            raise ValueError(f"{p}: key set differs from {experts[0]}")
    base = None
    if base_dir:
        base = Base(base_dir)
        if base.keys() != set(keys):
            raise ValueError(f"base key set differs: {len(base.keys() ^ set(keys))} keys")
    if mode == "tv" and base is None:
        raise ValueError("--mode tv needs --base")
    if identity is None:
        identity = request_identity(experts, mode, lam, base_dir, iteration, keys)
    if identity["n_tensors"] != len(keys):
        raise ValueError("identity was derived from a different key set")
    ident_digest = identity_digest(identity)
    _log(f"{n} experts, {len(keys)} tensors, mode={mode} lam={lam} identity={ident_digest[:12]}")
    out: dict[str, torch.Tensor] = {}  # values: [*shape]
    diag: dict[str, Any] = {
        "mode": mode,
        "lam": lam,
        "experts": list(experts),
        "base": base_dir,
        "n_tensors": len(keys),
        "per_tensor": {},
    }
    big = [k for k in keys if states[0][k].numel() >= 1_000_000]
    diag_keys = set(big[::8]) | set(sorted(big, key=lambda k: -states[0][k].numel())[:8])
    t0 = time.time()
    for i, k in enumerate(keys):
        ref = states[0][k]  # [*shape]
        for p, state in zip(experts, states):
            if state[k].shape != ref.shape or state[k].dtype != ref.dtype:
                raise ValueError(f"{p}:{k} shape/dtype mismatch")
        if mode == "avg":
            acc = states[0][k].float()  # [*shape]
            for state in states[1:]:
                acc = acc + state[k].float()  # [*shape]
            merged = acc / n  # [*shape]
            wb = base.get(k).float() if (base is not None and k in diag_keys) else None  # [*shape] or None
        else:
            assert base is not None  # guaranteed above; narrows the type
            wb = base.get(k).float()  # [*shape]
            if wb.shape != ref.shape:
                raise ValueError(f"base:{k} shape {tuple(wb.shape)} != {tuple(ref.shape)}")
            acc = wb.clone()  # [*shape]
            for state in states:
                acc.add_(state[k].float() - wb, alpha=lam)  # [*shape]
            merged = acc  # [*shape]
        out[k] = merged.to(ref.dtype)  # [*shape]
        if k in diag_keys and wb is not None:
            deltas = [(state[k].float() - wb).flatten() for state in states]  # each: [numel]
            wn = wb.norm().item() + 1e-12  # norm: []
            rel = [float(delta.norm().item() / wn) for delta in deltas]  # norms: []
            cos = []
            for a in range(n):
                for b in range(a + 1, n):
                    cos.append(
                        float(torch.dot(deltas[a], deltas[b]) / (deltas[a].norm() * deltas[b].norm() + 1e-12))
                    )  # []
            diag["per_tensor"][k] = {
                "numel": ref.numel(),
                "rel_delta_norm": rel,
                "pairwise_cos_mean": sum(cos) / max(1, len(cos)),
                "pairwise_cos_min": min(cos),
                "pairwise_cos_max": max(cos),
            }
            del deltas
        del acc, merged
        if (i + 1) % 50 == 0 or i + 1 == len(keys):
            _log(f"{i + 1}/{len(keys)} tensors, {time.time() - t0:.0f}s")
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)
    # transactional order: payload -> (verify, sha256) -> diag -> manifest -> completion record LAST
    tmp = out_path + ".tmp"
    torch.save({"net": out}, tmp)
    os.replace(tmp, out_path)
    size = os.path.getsize(out_path)
    _log(f"saved {out_path} ({size / 2**30:.1f} GiB)")
    # verify: reload via mmap, key set, three tensors bitwise, and (with a base) lm_head differs from the base
    check = load_expert(out_path)  # values: [*shape]
    if set(check) != set(keys):
        raise RuntimeError("reloaded key set differs")
    for k in (keys[0], keys[len(keys) // 2], keys[-1]):
        if not torch.equal(check[k], out[k]):
            raise RuntimeError(f"reloaded {k} differs")
    if base is not None and "lm_head.weight" in check:
        if torch.equal(check["lm_head.weight"], base.get("lm_head.weight")):
            raise RuntimeError("lm_head identical to base -- merge did nothing")
    del check
    payload = payload_digest(out_path)
    if payload["bytes"] != size:
        raise RuntimeError(f"{out_path} changed size while being hashed")
    diag_rec = None
    if diag_path:
        big_avg = [v["pairwise_cos_mean"] for v in diag["per_tensor"].values()]
        diag["summary"] = {
            "tensors_probed": len(diag["per_tensor"]),
            "mean_pairwise_cos": (sum(big_avg) / len(big_avg)) if big_avg else None,
            "mean_rel_delta_norm": [
                sum(v["rel_delta_norm"][t] for v in diag["per_tensor"].values()) / max(1, len(diag["per_tensor"]))
                for t in range(n)
            ]
            if diag["per_tensor"]
            else None,
        }
        diag["identity_digest"] = ident_digest
        os.makedirs(os.path.dirname(diag_path), exist_ok=True)
        atomic_write_json(diag_path, diag)
        diag_rec = {"path": os.path.realpath(diag_path), "bytes": os.path.getsize(diag_path)}
        _log(f"diag -> {diag_path}: {diag['summary']}")
    manifest = build_artifact_manifest(
        out_dir,
        "consolidated",
        ("model",),
        {"model": {"kind": "file", "path": os.path.basename(out_path), "size": size, "sha256": payload["sha256"]}},
    )
    manifest_path = os.path.join(out_dir, CHECKPOINT_ARTIFACT_MANIFEST)
    atomic_write_json(manifest_path, manifest)
    record = {
        "protocol": MERGE_COMPLETE_PROTOCOL,
        "tool": TOOL_VERSION,
        "identity": identity,
        "identity_digest": ident_digest,
        "merge": {
            "mode": mode,
            "lam": float(lam),
            "iteration": iteration,
            "experts": list(experts),
            "base": base_dir,
            "experts_sha256": [e["sha256"] for e in identity["experts"]],
        },
        "artifacts": {
            "payload": {"name": os.path.basename(out_path), **payload},
            "manifest": {"name": CHECKPOINT_ARTIFACT_MANIFEST, "bytes": os.path.getsize(manifest_path)},
            "diag": diag_rec,
        },
        "torch": torch.__version__,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    atomic_write_json(os.path.join(out_dir, MERGE_COMPLETE), record)
    _log(f"done (identity {ident_digest[:12]}, record {MERGE_COMPLETE})")


def main(argv: Sequence[str] | None = None) -> int:
    """Merge consolidated experts into one native checkpoint artifact; 0 when built or skipped, 1 when refused."""
    parser = argparse.ArgumentParser(
        prog="python -m cosmos_framework.posttrain.checkpoint.tools.merge",
        description="Streaming avg / task-vector merge of consolidated model.pt experts into one checkpoint artifact.",
    )
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--lam", type=float, default=1.0, help="task-vector scale (tv only)")
    parser.add_argument(
        "--experts", required=True, help="comma-separated consolidated model.pt paths (order is part of the identity)"
    )
    parser.add_argument(
        "--base", help="HF directory with model.safetensors.index.json (required for tv; enables --diag for avg)"
    )
    parser.add_argument("--out", required=True, help="<run>/<ts>/checkpoint_artifacts/iter_*/model.pt")
    parser.add_argument("--diag", help="JSON path for per-tensor delta diagnostics (needs --base)")
    parser.add_argument("--iteration", type=int, help="must equal the iter_* number of --out (default: derived)")
    parser.add_argument(
        "--force",
        action="store_true",
        help="rebuild over a COMPLETED merge of a different request at --out (default: refuse with exit 1)",
    )
    args = parser.parse_args(argv)
    experts = [p for p in args.experts.split(",") if p]
    try:
        result = run_request(
            experts, args.mode, args.lam, args.base, args.out, args.diag, args.iteration, force=args.force
        )
    except MergeRefusedError as error:
        print(str(error), file=sys.stderr, flush=True)
        return 1
    _log(f"{result}: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
