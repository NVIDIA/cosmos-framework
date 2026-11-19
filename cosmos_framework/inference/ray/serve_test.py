# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import asyncio
import functools
import http.server
import importlib.metadata
import json
import pickle
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import fastapi
import pydantic
import pytest
import ray
import ray.serve
import requests
import yaml
from starlette.testclient import TestClient

from cosmos_framework.inference.args import OmniSampleOverrides, _load_modality_defaults
from cosmos_framework.inference.common.args import SampleOutput, SampleOutputs
from cosmos_framework.inference.common.init import get_free_port
from cosmos_framework.inference.ray import serve
from cosmos_framework.inference.ray.serve import (
    OmniRouterDeployment,
    OmniRouterDeploymentArgs,
    OmniServeSampleOverrides,
)
from cosmos_framework.inference.ray.submit import handle_response

_CONFIGS_DIR = Path(__file__).parent / "configs"


@pytest.mark.parametrize("config_path", sorted(_CONFIGS_DIR.glob("*.yaml")), ids=lambda p: p.stem)
def test_config(monkeypatch: pytest.MonkeyPatch, config_path: Path):
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    config = yaml.safe_load(config_path.read_text())
    args = OmniRouterDeploymentArgs.model_validate(config["applications"][0]["args"])
    assert "Cosmos3-Nano" in args.models


def _mock_response(status_code: int, json_body: dict | None = None) -> requests.Response:
    resp = requests.Response()
    resp.status_code = status_code
    if json_body is not None:
        resp._content = json.dumps(json_body).encode()
    return resp


def test_handle_response_ok():
    handle_response(_mock_response(200))


def test_handle_response_error_with_detail():
    with pytest.raises(ValueError, match="Something went wrong"):
        handle_response(_mock_response(400, {"detail": "Something went wrong"}))


def test_handle_response_error_without_detail():
    with pytest.raises(requests.exceptions.HTTPError):
        handle_response(_mock_response(500))


def _make_generate_app(mock_handle: AsyncMock) -> fastapi.FastAPI:
    app = fastapi.FastAPI()

    @app.post("/generate")
    async def generate(sample_overrides: OmniServeSampleOverrides) -> SampleOutputs:
        return await mock_handle.generate.remote(sample_overrides)

    return app


def test_generate_e2e():
    mock_handle = AsyncMock()
    sample_kwargs = {"name": "test", "prompt": "a city"}
    mock_handle.generate.remote.return_value = SampleOutputs(
        args=sample_kwargs, status="success", outputs=[SampleOutput(content={}, files=[])]
    )
    client = TestClient(_make_generate_app(mock_handle))
    response = client.post("/generate", json=sample_kwargs)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["args"] == sample_kwargs
    mock_handle.generate.remote.assert_called_once()


def test_generate_error_response():
    mock_handle = AsyncMock()
    sample_kwargs = {"name": "test", "prompt": "a city"}
    mock_handle.generate.remote.return_value = SampleOutputs(
        args=sample_kwargs, status="error", message="Out of memory", stack_trace="Traceback ..."
    )
    client = TestClient(_make_generate_app(mock_handle))
    response = client.post("/generate", json=sample_kwargs)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "error"
    assert data["message"] == "Out of memory"
    assert data["stack_trace"] is not None


def _make_success_handle() -> AsyncMock:
    mock_handle = AsyncMock()
    mock_handle.generate.remote.return_value = SampleOutputs(
        args={"name": "test"}, status="success", outputs=[SampleOutput(content={}, files=[])]
    )
    return mock_handle


@pytest.mark.parametrize("field", ["pickle_file", "tensors_file"])
def test_generate_rejects_cli_only_field(tmp_path: Path, field: str):
    data_file = tmp_path / "sample_data.bin"
    data_file.write_bytes(b"")
    mock_handle = _make_success_handle()
    client = TestClient(_make_generate_app(mock_handle))
    response = client.post("/generate", json={"name": "test", "prompt": "a city", field: str(data_file)})
    assert response.status_code == 422
    assert f"{field} cannot be set in a server request" in response.text
    mock_handle.generate.remote.assert_not_called()


def test_generate_rejects_both_cli_only_fields(tmp_path: Path):
    data_file = tmp_path / "sample_data.bin"
    data_file.write_bytes(b"")
    mock_handle = _make_success_handle()
    client = TestClient(_make_generate_app(mock_handle))
    response = client.post(
        "/generate",
        json={"name": "test", "prompt": "a city", "pickle_file": str(data_file), "tensors_file": str(data_file)},
    )
    assert response.status_code == 422
    assert len(response.json()["detail"]) == 1
    assert "pickle_file, tensors_file cannot be set in a server request" in response.text
    mock_handle.generate.remote.assert_not_called()


def test_generate_rejects_missing_pickle_file(tmp_path: Path):
    # Rejected before the file is looked up, so the response does not reveal whether the path exists.
    mock_handle = _make_success_handle()
    client = TestClient(_make_generate_app(mock_handle))
    response = client.post(
        "/generate", json={"name": "test", "prompt": "a city", "pickle_file": str(tmp_path / "missing.pickle")}
    )
    assert response.status_code == 422
    assert len(response.json()["detail"]) == 1
    assert "pickle_file cannot be set in a server request" in response.text
    assert "does not point to a file" not in response.text
    mock_handle.generate.remote.assert_not_called()


def test_generate_accepts_full_sample_dump():
    # ``submit.py`` posts every field, with the CLI-only ones set to null.
    body = OmniSampleOverrides(name="test", prompt="a city").model_dump(mode="json")
    assert body["pickle_file"] is None
    assert body["tensors_file"] is None
    mock_handle = _make_success_handle()
    client = TestClient(_make_generate_app(mock_handle))
    response = client.post("/generate", json=body)
    assert response.status_code == 200
    mock_handle.generate.remote.assert_called_once()
    (sample_overrides,) = mock_handle.generate.remote.call_args.args
    assert sample_overrides.model_dump(mode="json") == body


def test_generate_accepts_defaults_file(tmp_path: Path):
    defaults_file = tmp_path / "defaults.json"
    defaults_file.write_text("{}")
    mock_handle = _make_success_handle()
    client = TestClient(_make_generate_app(mock_handle))
    response = client.post("/generate", json={"name": "test", "prompt": "a city", "defaults_file": str(defaults_file)})
    assert response.status_code == 200
    (sample_overrides,) = mock_handle.generate.remote.call_args.args
    assert sample_overrides.defaults_file == defaults_file.resolve()


def test_cli_only_sample_fields_exist():
    # A renamed field would otherwise stop being rejected without any error.
    assert set(serve._CLI_ONLY_SAMPLE_FIELDS) <= set(OmniSampleOverrides.model_fields)


def test_serve_sample_overrides_pickle_round_trip():
    # Ray pickles the request model to send it to the model deployment and its workers.
    assert issubclass(OmniServeSampleOverrides, OmniSampleOverrides)
    sample_overrides = OmniServeSampleOverrides(name="test", prompt="a city")
    restored = pickle.loads(pickle.dumps(sample_overrides))
    assert type(restored) is OmniServeSampleOverrides
    assert restored.model_dump() == sample_overrides.model_dump()


def _fake_model_config() -> SimpleNamespace:
    return SimpleNamespace(
        action_gen=False,
        sound_gen=False,
        resolution="480",
        tokenizer=SimpleNamespace(temporal_compression_factor=4),
        video_temporal_causal=False,
        vlm_config=SimpleNamespace(model_name="Qwen/Qwen3-VL-8B-Instruct"),
    )


def test_defaults_file_cannot_set_cli_only_fields(tmp_path: Path):
    # ``build_sample`` merges the defaults file and re-validates with ``type(self)``, so a defaults file
    # sent to the server cannot set the fields the request itself may not set.
    replay_file = tmp_path / "sample_data.pickle"
    replay_file.write_bytes(pickle.dumps({}))
    kwargs = dict(name="test", output_dir=tmp_path / "output", prompt="a city")
    model_mode = OmniSampleOverrides(**kwargs).sample_meta.model_mode
    defaults_file = tmp_path / "defaults.json"
    defaults_file.write_text(json.dumps(_load_modality_defaults(model_mode) | {"pickle_file": str(replay_file)}))
    kwargs["defaults_file"] = defaults_file

    # The CLI still accepts it.
    sample = OmniSampleOverrides(**kwargs).build_sample(model_config=_fake_model_config())
    assert sample.pickle_file == replay_file

    with pytest.raises(pydantic.ValidationError, match="pickle_file cannot be set in a server request"):
        OmniServeSampleOverrides(**kwargs).build_sample(model_config=_fake_model_config())


def test_environment_info_reports_package_version():
    assert serve._get_environment_info()["cosmos3_version"] == importlib.metadata.version("cosmos-framework")


def test_info_omits_output_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    # Ray's ingress wrapper copies the route functions, so patch what ``_get_environment_info`` calls.
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "0.0.0")
    router_cls = cast(ray.serve.Deployment, OmniRouterDeployment).func_or_class
    router = SimpleNamespace(models={"test-model": None}, output_dir=tmp_path)
    info = asyncio.run(router_cls.info(router))
    assert info["models"] == ["test-model"]
    assert "output_dir" not in info


@ray.serve.deployment
class _FakeModelDeployment:
    async def generate(self, sample_overrides: OmniSampleOverrides) -> SampleOutputs:
        return SampleOutputs(
            args=sample_overrides.model_dump(mode="json"),
            status="success",
            outputs=[SampleOutput(content={}, files=[])],
        )


@ray.serve.deployment
class _ReplayFakeModelDeployment:
    """Fake model that loads ``pickle_file`` the way ``SampleArgs.get_data`` does."""

    async def generate(self, sample_overrides: OmniSampleOverrides) -> SampleOutputs:
        if sample_overrides.pickle_file is not None:
            with sample_overrides.pickle_file.open("rb") as f:
                dict(pickle.load(f))
        return SampleOutputs(
            args=sample_overrides.model_dump(mode="json"),
            status="success",
            outputs=[SampleOutput(content={"type": type(sample_overrides).__name__}, files=[])],
        )


class _WriteMarkerOnLoad:
    """Unpickles by writing a marker file, so loading a replay file leaves a visible side effect."""

    def __init__(self, path: Path):
        self.path = path

    def __reduce__(self):
        return (Path.write_text, (self.path, "loaded"))


@pytest.mark.manual
@pytest.mark.level(1)
@pytest.mark.gpus(0)
def test_ray_serve_integration(tmp_path: Path):
    ray.init(num_cpus=2, num_gpus=0)  # type: ignore
    try:
        port = get_free_port()
        ray.serve.start(http_options={"host": "127.0.0.1", "port": port})

        app = cast(ray.serve.Deployment, OmniRouterDeployment).bind(
            {"test-model": cast(ray.serve.Deployment, _FakeModelDeployment).bind()},
            output_dir=tmp_path,
        )
        ray.serve.run(app, name="test")
        base_url = f"http://127.0.0.1:{port}"

        response = requests.get(f"{base_url}/models")
        assert response.status_code == 200
        assert "test-model" in response.json()

        sample_kwargs = {"name": "test", "prompt": "A city at night"}
        response = requests.post(
            f"{base_url}/generate",
            json=sample_kwargs,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        for k, v in sample_kwargs.items():
            assert data["args"][k] == v

        response = requests.post(
            f"{base_url}/generate",
            json={"name": "test", "prompt": "A city", "model": "nonexistent"},
        )
        assert response.status_code == 404
    finally:
        ray.serve.shutdown()
        ray.shutdown()  # type: ignore


@pytest.mark.manual
@pytest.mark.level(1)
@pytest.mark.gpus(0)
def test_ray_serve_rejects_replay_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    marker = tmp_path / "marker"
    replay_data = pickle.dumps({"data": _WriteMarkerOnLoad(marker)})
    output_dir = tmp_path / "outputs"

    # A replay file already on the server, where a URL download for ``vision_path`` would put it.
    planted_file = output_dir / "generate_x" / "inputs" / "vision.pkl"
    planted_file.parent.mkdir(parents=True)
    planted_file.write_bytes(replay_data)

    # The same file, served over HTTP for a ``vision_path`` download.
    served_dir = tmp_path / "served"
    (served_dir / "huggingface.co").mkdir(parents=True)
    (served_dir / "huggingface.co" / "vision.pkl").write_bytes(replay_data)
    file_server = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(served_dir))
    )
    threading.Thread(target=file_server.serve_forever, daemon=True).start()

    # Ray workers inherit the environment: keep downloads under tmp_path and fail fast.
    monkeypatch.setenv("COSMOS_DOWNLOAD_CACHE_DIR", str(tmp_path / "download_cache"))
    monkeypatch.setenv("COSMOS_DOWNLOAD_MAX_ATTEMPTS", "1")
    # ``ray.init`` sets these; register them so they are removed again after the test.
    for name in ("PYTHONBREAKPOINT", "RAY_USAGE_STATS_ENABLED"):
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    ray.init(num_cpus=2, num_gpus=0)  # type: ignore
    try:
        port = get_free_port()
        ray.serve.start(http_options={"host": "127.0.0.1", "port": port})

        app = cast(ray.serve.Deployment, OmniRouterDeployment).bind(
            {"test-model": cast(ray.serve.Deployment, _ReplayFakeModelDeployment).bind()},
            output_dir=output_dir,
        )
        ray.serve.run(app, name="test")
        base_url = f"http://127.0.0.1:{port}"

        replay_files = [planted_file]
        vision_url = f"http://127.0.0.1:{file_server.server_port}/huggingface.co/vision.pkl"
        response = requests.post(
            f"{base_url}/generate",
            json={"name": "test", "prompt": "A city", "vision_path": vision_url},
        )
        if response.status_code == 200:
            replay_files.append(Path(response.json()["args"]["vision_path"]))
        else:
            print(f"Download of {vision_url} failed ({response.status_code}); only the planted file is checked.")

        for replay_file in replay_files:
            assert replay_file.is_file()
            response = requests.post(
                f"{base_url}/generate",
                json={"name": "test", "prompt": "A city", "pickle_file": str(replay_file)},
            )
            assert response.status_code == 422, response.text
            assert "pickle_file cannot be set in a server request" in response.text
            assert not marker.exists()

        sample_kwargs = {"name": "test", "prompt": "A city at night"}
        response = requests.post(f"{base_url}/generate", json=sample_kwargs)
        assert response.status_code == 200
        data = response.json()
        assert data["outputs"][0]["content"]["type"] == "OmniServeSampleOverrides"
        # The router sets ``output_dir`` and ``seed``.
        expected_args = OmniServeSampleOverrides.model_validate(sample_kwargs).model_dump(mode="json")
        for key in ("output_dir", "seed"):
            expected_args.pop(key)
            data["args"].pop(key)
        assert data["args"] == expected_args
    finally:
        ray.serve.shutdown()
        ray.shutdown()  # type: ignore
        file_server.shutdown()
        file_server.server_close()
    assert not marker.exists()
