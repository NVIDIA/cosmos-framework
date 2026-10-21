# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Exercise receipt integrity without importing Framework or downloading weights."""

import fnmatch
import hashlib
import importlib.util
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

SCRIPT = Path(__file__).resolve().with_name("prepare_nano_dcp.py")
SPEC = importlib.util.spec_from_file_location("prepare_nano_dcp", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
prep = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prep)
FIXTURE_REVISION = "a" * 40
FIXTURE_RUNTIME = {"framework_revision": FIXTURE_REVISION, "python": "fixture"}


class ConverterCheckoutTest(unittest.TestCase):
    """Exercise checkout and import checks with real Git and a tiny CPU-only module."""

    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.framework = Path(temporary.name).resolve()
        self.module = self.framework / "cosmos_framework/scripts/convert_model_to_dcp.py"
        self.module.parent.mkdir(parents=True)
        for directory in (self.module.parent, self.module.parent.parent):
            (directory / "__init__.py").write_text("")
        self.module.write_text('"""Fixture converter, without model dependencies."""\n')
        self.git("init", "-q")
        self.git("config", "user.name", "Conversion Test")
        self.git("config", "user.email", "conversion-test@example.com")
        self.commit()
        supported_hash = patch.object(prep, "CONVERTER_SHA256", prep.fingerprint(self.module)["sha256"])
        supported_hash.start()
        self.addCleanup(supported_hash.stop)

    def git(self, *args: str) -> str:
        return subprocess.check_output(["git", "-C", str(self.framework), *args], text=True).strip()

    def commit(self) -> str:
        self.git("add", ".")
        self.git("-c", "commit.gpgsign=false", "commit", "-qm", "Converter fixture")
        return self.git("rev-parse", "HEAD")

    def test_supported_converter_in_another_commit_records_actual_revision(self) -> None:
        # This checkout has no historical Framework objects and is not the caller's cwd.
        with patch.object(prep.importlib.metadata, "version", return_value="fixture"):
            first = prep.check_environment(self.framework)
            (self.framework / "notes.md").write_text("Unrelated documentation change\n")
            next_revision = self.commit()
            second = prep.check_environment(self.framework)
        self.assertNotEqual(first["framework_revision"], next_revision)
        self.assertEqual(second["framework_revision"], next_revision)

    def test_unreviewed_converter_stops_before_download(self) -> None:
        self.module.write_text("# Different converter implementation\n")
        self.commit()
        with patch.object(prep, "download_source") as download:
            with self.assertRaisesRegex(ValueError, "supported SHA-256"):
                prep.convert(self.framework, self.framework / "source", self.framework / "output")
            download.assert_not_called()

    def test_dirty_checkout_is_rejected(self) -> None:
        self.module.write_text("# Local edit\n")
        for staged in (False, True):
            with self.subTest(staged=staged):
                if staged:
                    self.git("add", ".")
                with self.assertRaises(subprocess.CalledProcessError):
                    prep.check_checkout(self.framework)

    def test_wrong_import_location_is_rejected(self) -> None:
        self.module.write_text('__file__ = "/wrong-checkout/converter.py"\n')
        self.commit()
        with patch.object(prep, "CONVERTER_SHA256", prep.fingerprint(self.module)["sha256"]):
            with self.assertRaises(subprocess.CalledProcessError):
                prep.check_environment(self.framework)


class ConversionReceiptTest(unittest.TestCase):
    temporary: tempfile.TemporaryDirectory[str]
    root: Path
    source: Path
    output: Path
    framework: Path
    hub: MagicMock
    payloads: dict[str, bytes]

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.output = self.root / "dcp"
        self.framework = self.root / "framework"
        self.payloads = {"config.json": b'{"model": "fixture"}\n', "model.safetensors": b"source-weight-fixture"}
        siblings = [
            SimpleNamespace(
                rfilename="config.json",
                size=len(self.payloads["config.json"]),
                lfs=None,
                blob_id=hashlib.sha1(b"blob 21\0" + self.payloads["config.json"]).hexdigest(),
            ),
            SimpleNamespace(
                rfilename="model.safetensors",
                size=len(self.payloads["model.safetensors"]),
                lfs=SimpleNamespace(sha256=hashlib.sha256(self.payloads["model.safetensors"]).hexdigest()),
            ),
        ]
        self.hub = MagicMock()
        self.hub.HfApi.return_value.model_info.return_value = SimpleNamespace(
            sha=prep.MODEL_REVISION, siblings=siblings
        )
        self.hub.snapshot_download.side_effect = self.snapshot
        self.hub.hf_hub_download.side_effect = self.video_vae

    def snapshot(self, repo: str, **kwargs: object) -> str:
        self.assertEqual(repo, prep.MODEL_REPO)
        patterns = kwargs["allow_patterns"]
        assert isinstance(patterns, list)
        if "local_dir" in kwargs:
            self.assertEqual(kwargs["revision"], prep.MODEL_REVISION)
            self.source.mkdir()
            destination = self.source
        else:
            self.assertEqual(kwargs["revision"], prep.MODEL_REVISION)
            destination = (
                Path(str(kwargs["cache_dir"])) / "models--nvidia--Cosmos3-Nano/snapshots" / prep.MODEL_REVISION
            )
            destination.mkdir(parents=True)
        for name, data in self.payloads.items():
            if any(fnmatch.fnmatch(name, pattern) for pattern in patterns):
                path = destination / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
        return str(destination)

    def video_vae(self, repo: str, filename: str, **kwargs: object) -> str:
        self.assertEqual((repo, filename, kwargs["revision"]), (prep.WAN_REPO, prep.WAN_FILE, prep.WAN_REVISION))
        destination = Path(str(kwargs["cache_dir"])) / filename
        destination.write_bytes(b"video-vae-fixture")
        return str(destination)

    def native_conversion(self, command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        if command[0] == "git":
            return subprocess.CompletedProcess(command, 0)
        self.assertEqual(
            command[1:], ["-m", prep.CONVERTER_MODULE, "--checkpoint-path", str(self.source), "-o", str(self.output)]
        )
        self.assertEqual(kwargs["cwd"], self.framework)
        environment = kwargs["env"]
        assert isinstance(environment, dict)
        for key, value in prep.CONVERTER_ENV.items():
            self.assertEqual(environment[key], value)
        model = self.output / "model"
        model.mkdir(parents=True)
        (model / ".metadata").write_bytes(b"DCP metadata fixture")
        (model / "__0_0.distcp").write_bytes(b"converted-weight-fixture")
        (model / "config.json").write_text("{}\n")
        for name in prep.DERIVED_AUDIO:
            path = self.source / name
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(b"derived-audio-fixture")
        return subprocess.CompletedProcess(command, 0)

    def run_fixture_conversion(self) -> None:
        with (
            patch.object(prep, "check_environment", return_value=FIXTURE_RUNTIME),
            patch.object(prep, "check_checkout", return_value=FIXTURE_REVISION),
            patch.object(prep.importlib, "import_module", return_value=self.hub),
            patch.object(prep.subprocess, "run", side_effect=self.native_conversion),
        ):
            prep.convert(self.framework, self.source, self.output)

    def test_fresh_pinned_conversion_and_portable_copy(self) -> None:
        self.run_fixture_conversion()
        receipt = json.loads((self.output / prep.RECEIPT).read_text())
        self.assertEqual(receipt["identity"], prep.identity(FIXTURE_REVISION))
        self.assertEqual(set(receipt["source_files"]), set(self.payloads))
        self.assertEqual(set(receipt["derived_source_files"]), prep.DERIVED_AUDIO)
        self.assertNotIn(str(self.root), json.dumps(receipt))
        copied = self.root / "copied-checkpoint"
        shutil.copytree(self.output, copied)
        shutil.rmtree(self.source)
        shutil.rmtree(self.output)
        prep.verify(copied)
        # Intact payloads are insufficient when a loadable model's provenance
        # points to another source, converter or set of conversion options.
        for field in ("source_revision", "converter_revision", "converter_hash", "options"):
            with self.subTest(field=field):
                changed = json.loads(json.dumps(receipt))
                if field == "source_revision":
                    changed["identity"]["source"]["revision"] = "a" * 40
                elif field == "converter_revision":
                    changed["identity"]["converter"]["revision"] = "b" * 40
                elif field == "converter_hash":
                    changed["identity"]["converter"]["sha256"] = "b" * 64
                else:
                    changed["identity"]["converter"]["environment"]["COSMOS_SMOKE"] = "1"
                (copied / prep.RECEIPT).write_text(json.dumps(changed))
                with self.assertRaisesRegex(ValueError, "does not match"):
                    prep.verify(copied)
        (copied / prep.RECEIPT).write_text(json.dumps(receipt))
        (copied / "model/__0_0.distcp").write_bytes(b"changed-but-same-size----")
        with self.assertRaisesRegex(ValueError, "content is incomplete or changed"):
            prep.verify(copied)

    def test_legacy_receipt_needs_no_historical_checkout(self) -> None:
        self.run_fixture_conversion()
        receipt_path = self.output / prep.RECEIPT
        receipt = json.loads(receipt_path.read_text())
        receipt["identity"]["converter"]["revision"] = "573d46507dcd91155d7f371b366da760a7a5f718"
        del receipt["runtime"]["framework_revision"]
        receipt_path.write_text(json.dumps(receipt))
        prep.verify(self.output)

    def test_missing_or_malformed_checkout_provenance_is_rejected(self) -> None:
        self.run_fixture_conversion()
        receipt_path = self.output / prep.RECEIPT
        receipt = json.loads(receipt_path.read_text())
        for revision in (None, "main", "1234567", "g" * 40):
            with self.subTest(revision=revision):
                receipt["identity"]["converter"]["revision"] = revision
                receipt_path.write_text(json.dumps(receipt))
                with self.assertRaisesRegex(ValueError, "does not match"):
                    prep.verify(self.output)

    def test_checkout_change_during_conversion_writes_no_receipt(self) -> None:
        with (
            patch.object(prep, "check_environment", return_value=FIXTURE_RUNTIME),
            patch.object(prep, "check_checkout", return_value="b" * 40),
            patch.object(prep.importlib, "import_module", return_value=self.hub),
            patch.object(prep.subprocess, "run", side_effect=self.native_conversion),
        ):
            with self.assertRaisesRegex(ValueError, "checkout changed during conversion"):
                prep.convert(self.framework, self.source, self.output)
        self.assertFalse((self.output / prep.RECEIPT).exists())

    def test_missing_or_invalid_auxiliary_evidence_fails(self) -> None:
        self.run_fixture_conversion()
        receipt_path = self.output / prep.RECEIPT
        receipt = json.loads(receipt_path.read_text())
        auxiliary = receipt["auxiliary_files"]
        processor, video_vae = auxiliary["processor"], auxiliary["video_vae"]
        invalid = {
            "missing inventory": None,
            "empty inventory": {},
            "missing processor": {"video_vae": video_vae},
            "empty processor": {"processor": {}, "video_vae": video_vae},
            "missing video VAE": {"processor": processor},
            "invalid processor record": {"processor": {"tokenizer_config.json": "lost hashes"}, "video_vae": video_vae},
            "invalid processor hash": {
                "processor": {"tokenizer_config.json": {"bytes": 3, "sha256": "g" * 64}},
                "video_vae": video_vae,
            },
            "invalid video VAE byte count": {
                "processor": processor,
                "video_vae": {**video_vae, "bytes": -1},
            },
        }
        for label, value in invalid.items():
            with self.subTest(case=label):
                changed = {**receipt, "auxiliary_files": value}
                if value is None:
                    del changed["auxiliary_files"]
                receipt_path.write_text(json.dumps(changed))
                with self.assertRaisesRegex(ValueError, "auxiliary"):
                    prep.verify(self.output)

    def test_missing_receipt_and_unrecorded_files_fail(self) -> None:
        self.run_fixture_conversion()
        added = self.output / "extra.distcp"
        added.write_bytes(b"extra")
        with self.assertRaisesRegex(ValueError, "content is incomplete or changed"):
            prep.verify(self.output)
        added.unlink()
        (self.output / prep.RECEIPT).unlink()
        with self.assertRaisesRegex(ValueError, "Missing conversion receipt"):
            prep.verify(self.output)

    def test_cosmos3_only_environment_stops_before_download(self) -> None:
        def missing_framework(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
            if "-c" in command:
                raise subprocess.CalledProcessError(
                    1, command, stderr="ModuleNotFoundError: No module named cosmos_framework"
                )
            return subprocess.CompletedProcess(command, 0)

        with (
            patch.object(prep.subprocess, "check_output", return_value=FIXTURE_REVISION),
            patch.object(prep.subprocess, "run", side_effect=missing_framework),
            patch.object(prep, "fingerprint", return_value={"sha256": prep.CONVERTER_SHA256}),
            patch.object(prep, "download_source") as download,
        ):
            with self.assertRaises(subprocess.CalledProcessError):
                prep.convert(self.framework, self.source, self.output)
            download.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_bad_source_bytes_do_not_get_a_receipt(self) -> None:
        self.hub.HfApi.return_value.model_info.return_value.siblings[1].lfs.sha256 = "0" * 64
        with (
            patch.object(prep, "check_environment", return_value=FIXTURE_RUNTIME),
            patch.object(prep.importlib, "import_module", return_value=self.hub),
            patch.object(prep.subprocess, "run") as native,
        ):
            with self.assertRaisesRegex(ValueError, "differs from pinned Hub"):
                prep.convert(self.framework, self.source, self.output)
            native.assert_not_called()
        self.assertFalse((self.output / prep.RECEIPT).exists())

    def test_processor_cache_remains_pinned_when_remote_main_moves(self) -> None:
        original = self.hub.HfApi.return_value.model_info.return_value
        self.hub.HfApi.return_value.model_info.side_effect = (
            lambda repo, *, revision, **kwargs: original
            if revision == prep.MODEL_REVISION
            else SimpleNamespace(sha="a" * 40)
        )
        self.run_fixture_conversion()
        cache = self.source / ".cache/conversion-hub/models--nvidia--Cosmos3-Nano"
        revision = (cache / "refs/main").read_text()
        self.assertEqual((cache / "snapshots" / revision / "config.json").read_bytes(), self.payloads["config.json"])
        self.hub.HfApi.return_value.model_info.assert_called_once_with(
            prep.MODEL_REPO, revision=prep.MODEL_REVISION, files_metadata=True
        )

    def test_processor_cache_excludes_unselected_example_assets(self) -> None:
        name = "assets/example_t2v_prompt.json"
        payload = b'{"prompt": "example"}\n'
        self.payloads[name] = payload
        self.hub.HfApi.return_value.model_info.return_value.siblings.append(
            SimpleNamespace(
                rfilename=name,
                size=len(payload),
                lfs=None,
                blob_id=hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest(),
            )
        )
        self.run_fixture_conversion()
        receipt = json.loads((self.output / prep.RECEIPT).read_text())
        self.assertNotIn(name, receipt["source_files"])
        self.assertEqual(set(receipt["auxiliary_files"]["processor"]), {"config.json"})

    def test_failed_conversion_writes_no_receipt(self) -> None:
        with (
            patch.object(prep, "check_environment", return_value=FIXTURE_RUNTIME),
            patch.object(prep.importlib, "import_module", return_value=self.hub),
            patch.object(prep.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "converter")),
        ):
            with self.assertRaises(subprocess.CalledProcessError):
                prep.convert(self.framework, self.source, self.output)
        self.assertFalse((self.output / prep.RECEIPT).exists())

    def test_existing_source_is_not_relabelled(self) -> None:
        self.source.mkdir()
        with patch.object(prep, "check_environment") as check:
            with self.assertRaisesRegex(ValueError, "new source and output"):
                prep.convert(self.framework, self.source, self.output)
            check.assert_not_called()


if __name__ == "__main__":
    unittest.main()
