# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import contextlib
import http.server
import json
import os
import sys
import threading
import types
import urllib.parse
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import pytest

from cosmos_framework.inference.args import DEFAULT_CHECKPOINT, DEFAULT_CHECKPOINT_NAME, OmniSetupOverrides
from cosmos_framework.inference.common import args as common_args
from cosmos_framework.inference.common.args import (
    CheckpointConfig,
    CheckpointOverrides,
    CheckpointType,
    ConfigArgs,
    ConfigFileType,
    _HFRedirectHandler,
    _is_hf_url,
    _may_send_hf_token,
    download_file,
)
from cosmos_framework.inference.common.config import deserialize_config_dict

CHECKPOINTS: dict[str, CheckpointConfig] = {
    DEFAULT_CHECKPOINT_NAME: DEFAULT_CHECKPOINT,
}

# Enough of a Cosmos3 'model' section for config resolution; not a loadable architecture.
MODEL_SECTION = {"_target": "omni_mot_model", "config": {"_type": "omni_mot_model_config"}}


def write_local_hf_checkpoint(root: Path, *, config: dict, repo_id: str | None = None) -> Path:
    """Write a minimal local Hugging Face checkpoint directory."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (root / "model.safetensors").touch()
    if repo_id is not None:
        # Mirrors the layout '_write_modular_model_index' emits.
        (root / "modular_model_index.json").write_text(
            json.dumps(
                {
                    "_class_name": "Cosmos3OmniModularPipeline",
                    "transformer": [
                        "diffusers",
                        "Cosmos3OmniTransformer",
                        {
                            "pretrained_model_name_or_path": repo_id,
                            "subfolder": "transformer",
                            "type_hint": ["diffusers", "Cosmos3OmniTransformer"],
                            "variant": None,
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
    return root


def test_num_iterations_requires_benchmark() -> None:
    overrides = OmniSetupOverrides.model_construct(benchmark=False, num_iterations=2)

    with pytest.raises(ValueError, match="num_iterations > 1 requires benchmark=True"):
        overrides._build_setup()


def test_checkpoint_type_from_diffusers_layout(tmp_path: Path) -> None:
    transformer_path = tmp_path / "transformer"
    transformer_path.mkdir()
    (tmp_path / "model_index.json").write_text("{}", encoding="utf-8")
    (transformer_path / "config.json").write_text("{}", encoding="utf-8")
    (transformer_path / "diffusion_pytorch_model.safetensors.index.json").write_text("{}", encoding="utf-8")

    assert CheckpointType.from_path(tmp_path) == CheckpointType.HF


def test_download_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Disable the URL cache; this test asserts each download is independent.
    monkeypatch.delenv("COSMOS_DOWNLOAD_CACHE_DIR", raising=False)

    download_url_1 = (
        "https://github.com/nvidia-cosmos/cosmos-dependencies/raw/2b17a2413bd86b2cf9b03823637108851e4ddf2d/inputs/vision/robot_153.jpg"
    )
    file_size_1 = 279410

    download_url_2 = "https://github.com/nvidia-cosmos/cosmos-dependencies/raw/2b17a2413bd86b2cf9b03823637108851e4ddf2d/inputs/vision/bus_terminal.jpg"
    file_size_2 = 1283715

    # Download file
    download_path = Path(
        download_file(
            download_url_1,
            tmp_path,
            "robot_welding",
        )
    )
    assert download_path.stat().st_size == file_size_1
    meta_path = Path(f"{download_path}.meta")
    assert json.loads(meta_path.read_text()) == {
        "url": download_url_1,
    }
    cache_path = download_path.resolve()

    # Same file should be noop
    download_path = Path(download_file(str(download_path), tmp_path, "robot_welding"))
    assert download_path.resolve() == cache_path

    # Copy file
    copy_path = Path(download_file(str(download_path), tmp_path, "robot_welding_copy"))
    assert copy_path.stat().st_size == file_size_1
    assert copy_path.resolve() == cache_path

    # Re-download should be noop
    copy_path = Path(download_file(str(download_path), tmp_path, "robot_welding_copy"))
    assert copy_path.resolve() == cache_path

    # Force re-download
    os.remove(meta_path)
    download_path = Path(download_file(download_url_1, tmp_path, "robot_welding"))
    assert download_path.resolve() != cache_path
    assert download_path.stat().st_size == file_size_1

    # Different file should overwrite
    download_path = Path(
        download_file(
            download_url_2,
            tmp_path,
            "robot_welding",
        )
    )
    assert download_path.stat().st_size == file_size_2
    assert json.loads(Path(f"{download_path}.meta").read_text()) == {
        "url": download_url_2,
    }


@pytest.mark.parametrize(
    ("url", "is_hf", "may_send_token"),
    [
        ("https://huggingface.co/a/b/resolve/main/x", True, True),
        ("https://cdn-lfs.huggingface.co/x", True, True),
        ("HTTPS://HuggingFace.CO/x", True, True),
        ("http://huggingface.co/x", True, False),
        ("s3://huggingface.co/x", True, False),
        ("https://other.example/huggingface.co/x", False, False),
        ("https://huggingface.co.other.example/x", False, False),
        ("https://otherhuggingface.co/x", False, False),
        ("https://huggingface.co@other.example/x", False, False),
    ],
)
def test_hf_url_matches_parsed_host(url: str, is_hf: bool, may_send_token: bool):
    assert _is_hf_url(url) is is_hf
    assert _may_send_hf_token(url) is may_send_token


@pytest.mark.parametrize(
    ("url", "branch"),
    [
        ("https://huggingface.co/a/b/resolve/main/x.bin", "hf"),
        ("https://other.example/huggingface.co/x.bin", "obstore"),
        ("https://huggingface.co.other.example/x.bin", "obstore"),
    ],
)
def test_download_dispatches_on_hf_host(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, url: str, branch: str):
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(common_args, "_download_file_hf", lambda url, path: calls.append(("hf", url)))
    fake_obstore = types.SimpleNamespace(
        store=types.SimpleNamespace(from_url=lambda base_url: base_url),
        get=lambda store, file_name: calls.append(("obstore", f"{store}/{file_name}")) or [b"data"],
    )
    monkeypatch.setitem(sys.modules, "obstore", fake_obstore)

    common_args._download_file_url_once(url, tmp_path / "out.bin")

    assert calls == [(branch, url)]


def test_hf_download_upgrades_http_to_https(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    calls: list[str] = []
    monkeypatch.setattr(common_args, "_download_file_hf", lambda url, path: calls.append(url))

    common_args._download_file_url_once("http://huggingface.co/a/b/resolve/main/x.bin?download=true", tmp_path / "x")

    assert calls == ["https://huggingface.co/a/b/resolve/main/x.bin?download=true"]


@pytest.mark.parametrize(
    ("newurl", "keeps_token"),
    [
        ("https://huggingface.co/a/b/resolve/main/y", True),
        ("https://cdn-lfs.huggingface.co/x", True),
        ("https://other.example/x", False),
        ("http://huggingface.co/x", False),
    ],
)
def test_hf_redirect_handler_drops_token_off_hf(newurl: str, keeps_token: bool):
    req = urllib.request.Request(
        "https://huggingface.co/a/b/resolve/main/x", headers={"Authorization": "Bearer test-token"}
    )
    # 'Request' normalizes header keys with 'str.capitalize', which leaves 'Authorization' unchanged.
    assert req.has_header("Authorization")

    new_req = _HFRedirectHandler().redirect_request(req, None, 302, "Found", {}, newurl)

    assert new_req is not None
    assert new_req.full_url == newurl
    assert new_req.get_header("Authorization") == ("Bearer test-token" if keeps_token else None)


@contextlib.contextmanager
def _local_http_server(host: str, *, redirect_to: str | None = None, body: bytes = b"") -> Iterator[tuple[str, list]]:
    """Serve GET requests on ``host`` at an ephemeral port.

    Yields the base URL and the ``Authorization`` header seen on each request. With ``redirect_to``, every request
    gets a 302 to the same path on that base URL; otherwise it gets ``body``.
    """
    auth_headers: list[str | None] = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            auth_headers.append(self.headers.get("Authorization"))
            if redirect_to is not None:
                self.send_response(302)
                self.send_header("Location", f"{redirect_to}{self.path}")
                self.send_header("Content-Length", "0")
                self.end_headers()
            else:
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        def log_message(self, format, *args) -> None:
            pass

    server = http.server.ThreadingHTTPServer((host, 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://{host}:{server.server_address[1]}", auth_headers
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.fixture
def _no_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep requests to the local test servers off any configured proxy."""
    for name in ("no_proxy", "NO_PROXY"):
        monkeypatch.setenv(name, "*")


@pytest.mark.usefixtures("_no_proxy")
def test_hf_token_dropped_on_redirect_to_other_host(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Treat 'localhost' as the Hugging Face host; the redirect goes to '127.0.0.1', a different host.
    monkeypatch.setattr(common_args, "_is_hf_url", lambda url: urllib.parse.urlsplit(url).hostname == "localhost")
    monkeypatch.setattr(common_args, "_may_send_hf_token", common_args._is_hf_url)
    monkeypatch.setenv("HF_TOKEN", "test-token")
    path = tmp_path / "out.bin"

    with _local_http_server("127.0.0.1", body=b"file from B") as (url_b, auth_b):
        with _local_http_server("localhost", redirect_to=url_b) as (url_a, auth_a):
            common_args._download_file_hf(f"{url_a}/a/b/resolve/main/x.bin", path)

    assert auth_a == ["Bearer test-token"]
    assert auth_b == [None]
    assert path.read_bytes() == b"file from B"


@pytest.mark.usefixtures("_no_proxy")
def test_hf_token_not_sent_over_http(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Treat the local server as a Hugging Face host; the real '_may_send_hf_token' still requires https.
    monkeypatch.setattr(common_args, "_is_hf_url", lambda url: True)
    # Call the downloader directly: the dispatcher would upgrade the URL to https first.
    monkeypatch.setenv("HF_TOKEN", "test-token")
    path = tmp_path / "out.bin"

    with _local_http_server("127.0.0.1", body=b"data") as (url, auth_headers):
        common_args._download_file_hf(f"{url}/x.bin", path)

    assert auth_headers == [None]
    assert path.read_bytes() == b"data"


def test_parse_checkpoint_path(tmp_path: Path):
    # Named checkpoint
    args = CheckpointOverrides(
        checkpoint_path=DEFAULT_CHECKPOINT_NAME,
    ).build_checkpoint(checkpoints=CHECKPOINTS)
    assert args.checkpoint_type == "hf"
    assert args.experiment == ""

    # Local HF checkpoint: a copy of the registered repo resolves to its config.
    hf_path = write_local_hf_checkpoint(
        tmp_path / "hf", config={"model": MODEL_SECTION}, repo_id=DEFAULT_CHECKPOINT.hf.repository
    )
    args = CheckpointOverrides(
        checkpoint_path=str(hf_path),
    ).build_checkpoint(checkpoints=CHECKPOINTS)
    assert args.checkpoint_type == "hf"
    assert args.experiment == ""
    assert args.config_file == DEFAULT_CHECKPOINT.config_file

    # Local DCP checkpoint
    dcp_path = (
        tmp_path
        / "project/group/t2i_mot_exp000_015_qwen3_0p6b_256res_frozen_llm_lr_4e4_large_seq_no_llm_qknorm_gcp_bs8k_baseline_long_run_1e4/checkpoints/iter_000700000/model"
    )
    dcp_path.mkdir(parents=True, exist_ok=True)
    (dcp_path / ".metadata").touch()
    (dcp_path / "__0_0.distcp").touch()
    args = CheckpointOverrides(
        checkpoint_path=str(dcp_path),
    ).build_checkpoint(checkpoints=CHECKPOINTS)
    assert args.checkpoint_type == "dcp"
    assert (
        args.experiment
        == "t2i_mot_exp000_015_qwen3_0p6b_256res_frozen_llm_lr_4e4_large_seq_no_llm_qknorm_gcp_bs8k_baseline_long_run_1e4"
    )

    # S3 DCP checkpoint
    args = CheckpointOverrides(
        checkpoint_path=f"s3://bucket/project/group/t2i_mot_exp000_015_qwen3_0p6b_256res_frozen_llm_lr_4e4_large_seq_no_llm_qknorm_gcp_bs8k_baseline_long_run_1e4/checkpoints/iter_000700000/model",
    ).build_checkpoint(checkpoints=CHECKPOINTS)
    assert args.checkpoint_type == "dcp"
    assert (
        args.experiment
        == "t2i_mot_exp000_015_qwen3_0p6b_256res_frozen_llm_lr_4e4_large_seq_no_llm_qknorm_gcp_bs8k_baseline_long_run_1e4"
    )
    assert args.config_file == "cosmos_framework/configs/base/config.py"


def test_local_edge_checkpoint_resolves_registry_config(tmp_path: Path):
    """A local copy of nvidia/Cosmos3-Edge gets its architecture from the registry.

    The published Edge repo's top-level config.json is the reasoner HF config and
    carries no 'model' section, so the architecture is recovered from the repo id
    the checkpoint records for itself rather than from the directory.
    """
    checkpoints = OmniSetupOverrides.CHECKPOINTS
    edge = checkpoints["Cosmos3-Edge"]
    # A renamed copy: the directory name must not be what makes this work.
    checkpoint_dir = write_local_hf_checkpoint(
        tmp_path / "Cosmos-Edge",
        config={"architectures": ["Cosmos3EdgeForConditionalGeneration"], "model_type": "cosmos3_edge"},
        repo_id=edge.hf.repository,
    )

    args = CheckpointOverrides(checkpoint_path=str(checkpoint_dir)).build_checkpoint(checkpoints=checkpoints)

    assert args.checkpoint_type == "hf"
    assert args.config_file == edge.config_file
    assert args.model_memory_bytes == edge.model_memory_bytes
    # The weights still come from the local directory, not a hub download.
    assert args.checkpoint_path == str(checkpoint_dir)
    # The resolved config really does describe the architecture.
    assert "ema" in args.load_model_config_dict()["config"]


def test_local_checkpoint_without_architecture_raises(tmp_path: Path):
    """An unidentifiable local HF dir fails fast instead of building a bogus config."""
    checkpoint_dir = write_local_hf_checkpoint(tmp_path / "mystery", config={"model_type": "cosmos3_edge"})

    with pytest.raises(ValueError, match="records no source repository"):
        CheckpointOverrides(checkpoint_path=str(checkpoint_dir)).build_checkpoint(
            checkpoints=OmniSetupOverrides.CHECKPOINTS
        )


def test_registered_repository_wins_over_the_directory_config(tmp_path: Path):
    """A registered repository's config wins over the copy's own 'config.json'.

    A published repo's 'config.json' can be an older snapshot of the architecture
    than the registered config: nvidia/Cosmos3-Nano omits 'include_visual' and
    names the text-only processor, which made a local copy load a text-only
    Reasoner. The registered name and a local copy must resolve identically.
    """
    checkpoints = OmniSetupOverrides.CHECKPOINTS
    nano = checkpoints["Cosmos3-Nano"]
    checkpoint_dir = write_local_hf_checkpoint(
        tmp_path / "snapshot", config={"model": MODEL_SECTION}, repo_id=nano.hf.repository
    )

    args = CheckpointOverrides(checkpoint_path=str(checkpoint_dir)).build_checkpoint(checkpoints=checkpoints)

    assert args.config_file == nano.config_file
    vlm_config = args.load_model_config_dict()["config"]["vlm_config"]
    assert vlm_config["model_instance"]["config"]["include_visual"] is True
    assert vlm_config["tokenizer"]["_target_"].endswith("build_processor_lazy")


def test_unregistered_local_checkpoint_uses_its_own_config(tmp_path: Path):
    """A checkpoint we do not publish keeps describing itself.

    Its own 'config.json' is the only description available, so it must keep
    working without '--config-file' (e.g. the ModelOpt FP8 Nano checkpoint, which
    ships from a subdirectory of an unregistered repository).
    """
    checkpoint_dir = write_local_hf_checkpoint(tmp_path / "self-described", config={"model": MODEL_SECTION})

    args = CheckpointOverrides(checkpoint_path=str(checkpoint_dir)).build_checkpoint(
        checkpoints=OmniSetupOverrides.CHECKPOINTS
    )

    assert args.config_file == str(checkpoint_dir / "config.json")


def test_explicit_config_file_bypasses_resolution(tmp_path: Path):
    """An explicit '--config-file' is honoured for any local directory."""
    checkpoint_dir = write_local_hf_checkpoint(tmp_path / "self-described", config={"model": MODEL_SECTION})
    edge_config = OmniSetupOverrides.CHECKPOINTS["Cosmos3-Edge"].config_file

    args = CheckpointOverrides(checkpoint_path=str(checkpoint_dir), config_file=edge_config).build_checkpoint(
        checkpoints=OmniSetupOverrides.CHECKPOINTS
    )

    assert args.config_file == edge_config
    assert args.checkpoint_path == str(checkpoint_dir)


def test_local_checkpoint_with_unregistered_repository_raises(tmp_path: Path):
    """An unknown source repository is named, and an empty registry still reads cleanly.

    'export_model' / 'export_train_config' pass an empty registry, so the message
    must not degrade into an empty list there.
    """
    checkpoint_dir = write_local_hf_checkpoint(
        tmp_path / "unknown", config={"model_type": "cosmos3_edge"}, repo_id="acme/Not-Registered"
    )

    with pytest.raises(ValueError, match=r"acme/Not-Registered.*Registered checkpoints: <none>"):
        CheckpointOverrides(checkpoint_path=str(checkpoint_dir)).build_checkpoint(checkpoints={})


def test_load_model_config_dict_requires_model_section(tmp_path: Path):
    """'experiment_overrides' must not be able to synthesize a 'model' section.

    The overrides are dotlists rooted at 'model.'. Merging them into a config
    without a 'model' section used to produce a two-key stub that only failed
    much later, deep in model construction, as 'Missing key ema'.
    """
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"model_type": "cosmos3_edge"}), encoding="utf-8")
    args = ConfigArgs(
        config_file=str(config_path),
        config_file_type=ConfigFileType.JSON,
        experiment="",
        experiment_overrides=["model.config.vlm_config.pretrained_weights.enabled=False"],
    )

    with pytest.raises(KeyError, match="no 'model' section"):
        args.load_model_config_dict()


def test_registry_model_configs_describe_an_architecture():
    """Every registered checkpoint's config file carries a usable 'model' section."""
    for name, checkpoint in OmniSetupOverrides.CHECKPOINTS.items():
        config_dict = deserialize_config_dict(Path(checkpoint.config_file))
        assert "model" in config_dict, f"{name}: {checkpoint.config_file} has no 'model' section"
        assert "ema" in config_dict["model"]["config"], f"{name}: {checkpoint.config_file} has no 'model.config.ema'"
