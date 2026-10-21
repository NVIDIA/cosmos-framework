"""Exercise installed resources and CLI resolution away from a source checkout."""

import importlib.resources
import os
from pathlib import Path
import subprocess
import sys

from hydra import compose, initialize_config_module


def test_packaged_configuration_and_palettes() -> None:
    with initialize_config_module(config_module="cosmos_auto.wm.conf", version_base=None):
        config = compose(config_name="render", overrides=["input_root=scene", "output_root=output"])
    assert config.version == "v4"
    assert config.s3_spec.enable_upload is False
    assert config.s3_spec.cred_path_or_profile is None
    resources = importlib.resources.files("cosmos_auto.wm")
    assert resources.joinpath("assets/road_marking_arrows/arrow_shape.json").is_file()
    assert resources.joinpath("assets/default_ftheta_intrinsic.tar").is_file()
    assert resources.joinpath("color_scheme/config_color_traffic_light.json").is_file()


def test_cli_from_unrelated_directory(tmp_path: Path) -> None:
    environment = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    result = subprocess.run(
        [sys.executable, "-I", "-m", "cosmos_auto.wm.cli", "--cfg", "job", "--resolve",
         "input_root=scene", "output_root=output"],
        cwd=tmp_path, env=environment, capture_output=True, text=True, check=True,
    )
    assert "version: v4" in result.stdout
    assert "enable_upload: false" in result.stdout
