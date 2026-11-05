"""Run the documented scene example from a freshly built source distribution."""

import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path


def test_lidar_scene_from_source_distribution(tmp_path: Path) -> None:
    source = tmp_path / "source"
    shutil.copytree(
        Path(__file__).resolve().parents[1],
        source,
        ignore=shutil.ignore_patterns(
            ".venv",
            "build",
            "dist",
            "*.egg-info",
            "__pycache__",
            ".pytest_cache",
            ".ruff_cache",
        ),
    )
    distribution = tmp_path / "dist"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--sdist",
            "--outdir",
            str(distribution),
            str(source),
        ],
        check=True,
    )
    [sdist] = distribution.glob("cosmos_auto_wsm-*.tar.gz")
    extracted = tmp_path / "extracted"
    with tarfile.open(sdist) as archive:
        archive.extractall(extracted, filter="data")
    [package] = extracted.iterdir()
    assert (package / "examples/lidar_scene.py").is_file()
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "examples/lidar_scene.py",
            "--output-root",
            "outputs/scene",
        ],
        cwd=package,
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads(result.stdout)
    assert report["pose_frames"] == 202
    assert report["object_frames"] == 202
    assert report["cameras"] == ["camera_front_wide_120fov"]
