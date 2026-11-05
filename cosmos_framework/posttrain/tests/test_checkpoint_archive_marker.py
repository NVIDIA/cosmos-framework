# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""CPU known answers for the destination- and digest-bound ``.uploaded`` archive marker.

Every ``marker_covers(...) is False`` case below was accepted as "uploaded" by the earlier
existence-only marker: another destination's marker, a partial archive, a resized file, a legacy
text marker, and a torn marker file.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from cosmos_framework.posttrain.checkpoint.archive import marker as archive_marker
from cosmos_framework.posttrain.checkpoint.archive.marker import (
    MARKER_NAME,
    PROTOCOL,
    Marker,
    marker_covers,
    payload_inventory,
    read_marker,
    write_marker,
)
from cosmos_framework.posttrain.checkpoint.provenance import checkpoint_inventory_signature

WEIGHTS = b"weights" * 1000
STATE = b"trainer-state" * 10
OPTIM = b"optim" * 100
DESTINATION = {
    "kind": "object_store",
    "endpoint_authority": "storage.example.test",
    "bucket": "archive",
    "prefix": "project/run/checkpoints/iter_000100",
}
OTHER_DESTINATION = {**DESTINATION, "prefix": "project/other/checkpoints/iter_000100"}


def _iteration(tmp_path: Path) -> Path:
    iter_dir = tmp_path / "run" / "checkpoints" / "iter_000100"
    (iter_dir / "trainer").mkdir(parents=True)
    (iter_dir / "model.pt").write_bytes(WEIGHTS)
    (iter_dir / "trainer" / "state.pt").write_bytes(STATE)
    (iter_dir / "optim.pt").write_bytes(OPTIM)
    return iter_dir


def _inventory_files(iter_dir: Path) -> dict[str, dict[str, object]]:
    files: dict[str, dict[str, object]] = {
        relative: {"size": size, "sha256": None} for relative, size in payload_inventory(str(iter_dir)).items()
    }
    files["trainer/state.pt"]["sha256"] = hashlib.sha256(STATE).hexdigest()
    return files


@pytest.mark.L0
@pytest.mark.CPU
def test_write_marker_round_trips_atomically(tmp_path: Path) -> None:
    iter_dir = _iteration(tmp_path)
    files = _inventory_files(iter_dir)
    manifest_sha256 = hashlib.sha256(b"manifest").hexdigest()

    path = write_marker(str(iter_dir), destination=DESTINATION, files=files, manifest_sha256=manifest_sha256)

    assert path == str(iter_dir / MARKER_NAME) and Path(path).is_file()
    assert not list(iter_dir.glob(f"{MARKER_NAME}.tmp*")), "atomic write leaves no temporary residue"
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    assert payload["protocol"] == PROTOCOL
    assert set(payload) == {"protocol", "destination", "files", "manifest_sha256"}
    marker = read_marker(str(iter_dir))
    assert marker == Marker(
        protocol=PROTOCOL,
        destination=DESTINATION,
        files={
            "model.pt": {"size": len(WEIGHTS), "sha256": None},
            "optim.pt": {"size": len(OPTIM), "sha256": None},
            "trainer/state.pt": {"size": len(STATE), "sha256": hashlib.sha256(STATE).hexdigest()},
        },
        manifest_sha256=manifest_sha256,
        legacy=False,
    )
    assert marker_covers(str(iter_dir), DESTINATION)


@pytest.mark.L0
@pytest.mark.CPU
def test_payload_inventory_excludes_marker_and_its_temporary(tmp_path: Path) -> None:
    iter_dir = _iteration(tmp_path)
    (iter_dir / MARKER_NAME).write_text("{}", encoding="utf-8")
    (iter_dir / f"{MARKER_NAME}.tmp").write_text("{", encoding="utf-8")

    assert payload_inventory(str(iter_dir)) == {
        "model.pt": len(WEIGHTS),
        "optim.pt": len(OPTIM),
        "trainer/state.pt": len(STATE),
    }


@pytest.mark.L0
@pytest.mark.CPU
def test_marker_covers_known_answers(tmp_path: Path) -> None:
    iter_dir = _iteration(tmp_path)
    assert not marker_covers(str(iter_dir), DESTINATION), "no marker yet"

    write_marker(str(iter_dir), destination=DESTINATION, files=_inventory_files(iter_dir))
    assert marker_covers(str(iter_dir), DESTINATION)
    assert not marker_covers(str(iter_dir), OTHER_DESTINATION), "another destination's marker"
    assert not marker_covers(str(iter_dir), {**DESTINATION, "bucket": "other"}), "another bucket's marker"

    (iter_dir / "extra.pt").write_bytes(b"x")
    assert not marker_covers(str(iter_dir), DESTINATION), "a file the archive does not cover (partial archive)"
    (iter_dir / "extra.pt").unlink()
    assert marker_covers(str(iter_dir), DESTINATION)

    (iter_dir / "model.pt").write_bytes(b"w" * (len(WEIGHTS) + 1))
    assert not marker_covers(str(iter_dir), DESTINATION), "same name, different size"
    (iter_dir / "model.pt").write_bytes(WEIGHTS)
    assert marker_covers(str(iter_dir), DESTINATION)

    # Optimizer state pruned after the archive: the inventory still covers what remains on disk.
    (iter_dir / "optim.pt").unlink()
    assert marker_covers(str(iter_dir), DESTINATION)


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("text", ["archive\n", "bucket\n", "publication side effect\n", "<checkpoint-bucket>"])
def test_legacy_text_marker_reads_existence_only(tmp_path: Path, text: str) -> None:
    iter_dir = _iteration(tmp_path)
    (iter_dir / MARKER_NAME).write_text(text, encoding="utf-8")

    marker = read_marker(str(iter_dir))
    assert marker is not None and marker.legacy
    assert marker.destination == {} and marker.files == {} and marker.manifest_sha256 is None
    assert (iter_dir / MARKER_NAME).is_file(), "the existence-only retention gate still sees the file"
    assert not marker_covers(str(iter_dir), DESTINATION), "a legacy marker never digest- or destination-verifies"


@pytest.mark.L0
@pytest.mark.CPU
def test_other_writer_json_marker_reads_as_legacy(tmp_path: Path) -> None:
    iter_dir = _iteration(tmp_path)
    sidecar = {"v": 2, "bucket": "archive", "dest": "users/x/run/ts/iter_000100", "files": {}, "n_files": 0}
    (iter_dir / MARKER_NAME).write_text(json.dumps(sidecar), encoding="utf-8")

    marker = read_marker(str(iter_dir))
    assert marker is not None and marker.legacy and marker.files == {} and marker.destination == {}
    assert not marker_covers(str(iter_dir), DESTINATION)


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    "content",
    [
        b"",
        b"   \n",
        b"{not json",
        b'{"protocol": "posttrain.archive-mar',
        b"\xff\xfe\x00binary",
        b'["archive"]',
        b"line one\nline two\n",
        json.dumps({"protocol": PROTOCOL}).encode(),
        json.dumps({"protocol": PROTOCOL, "destination": {}, "files": {}, "manifest_sha256": None}).encode(),
        json.dumps(
            {
                "protocol": PROTOCOL,
                "destination": DESTINATION,
                "files": {"model.pt": {"size": -1}},
                "manifest_sha256": None,
            }
        ).encode(),
        json.dumps(
            {
                "protocol": PROTOCOL,
                "destination": DESTINATION,
                "files": {"../model.pt": {"size": 1, "sha256": None}},
                "manifest_sha256": None,
            }
        ).encode(),
        json.dumps(
            {"protocol": PROTOCOL, "destination": DESTINATION, "files": {}, "manifest_sha256": "not-a-digest"}
        ).encode(),
    ],
)
def test_torn_or_malformed_marker_is_unusable(tmp_path: Path, content: bytes) -> None:
    iter_dir = _iteration(tmp_path)
    (iter_dir / MARKER_NAME).write_bytes(content)

    assert read_marker(str(iter_dir)) is None
    assert not marker_covers(str(iter_dir), DESTINATION)


@pytest.mark.L0
@pytest.mark.CPU
def test_read_marker_absent(tmp_path: Path) -> None:
    iter_dir = _iteration(tmp_path)
    assert read_marker(str(iter_dir)) is None
    assert read_marker(str(tmp_path / "missing")) is None
    assert read_marker(str(iter_dir / "model.pt")) is None


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    ("destination", "files", "manifest_sha256"),
    [
        ({}, {"model.pt": {"size": 1, "sha256": None}}, None),
        ({"bucket": 3}, {"model.pt": {"size": 1, "sha256": None}}, None),
        (DESTINATION, {"../model.pt": {"size": 1, "sha256": None}}, None),
        (DESTINATION, {"model.pt": {"size": True, "sha256": None}}, None),
        (DESTINATION, {"model.pt": {"size": -1, "sha256": None}}, None),
        (DESTINATION, {"model.pt": {"size": 1}}, None),
        (DESTINATION, {"model.pt": {"size": 1, "sha256": "ABC"}}, None),
        (DESTINATION, {"model.pt": {"size": 1, "sha256": None}}, "not-a-digest"),
    ],
)
def test_write_marker_rejects_invalid_inputs(
    tmp_path: Path,
    destination: dict[str, str],
    files: dict[str, dict[str, object]],
    manifest_sha256: str | None,
) -> None:
    iter_dir = _iteration(tmp_path)

    with pytest.raises(ValueError):
        write_marker(str(iter_dir), destination=destination, files=files, manifest_sha256=manifest_sha256)
    assert not (iter_dir / MARKER_NAME).exists()
    assert not list(iter_dir.glob(f"{MARKER_NAME}.tmp*"))


@pytest.mark.L0
@pytest.mark.CPU
def test_marker_does_not_change_checkpoint_inventory_signature(tmp_path: Path) -> None:
    iter_dir = _iteration(tmp_path)
    before = checkpoint_inventory_signature(str(iter_dir))

    write_marker(str(iter_dir), destination=DESTINATION, files=_inventory_files(iter_dir))

    assert checkpoint_inventory_signature(str(iter_dir)) == before
    assert archive_marker.MARKER_NAME == ".uploaded", "retention gates and the provenance skip list name this file"
