# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Build a focused MkDocs fixture and verify the application and guide routes."""

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    target = output / "docs/camera-rig"
    target.mkdir(parents=True, exist_ok=True)
    for name in (
        "index.html",
        "guide.md",
        "styles.css",
        "app.js",
        "geometry.js",
        "example-rigs.js",
        "layouts.js",
        "preview.png",
    ):
        shutil.copyfile(source / name, target / name)
    config = output / "mkdocs.yml"
    config.write_text(
        "site_name: Camera rig documentation test\ndocs_dir: docs\nsite_dir: site\nuse_directory_urls: true\n"
    )
    subprocess.run([sys.executable, "-m", "mkdocs", "build", "--strict", "--config-file", str(config)], check=True)
    app = output / "site/camera-rig/index.html"
    guide = output / "site/camera-rig/guide/index.html"
    assert app.read_bytes() == (source / "index.html").read_bytes()
    link = re.search(r'<a href="([^"]+)">Open it locally</a>', guide.read_text())
    assert link is not None
    destination = (guide.parent / link.group(1)).resolve()
    if destination.is_dir():
        destination /= "index.html"
    assert destination == app.resolve()
    assert (output / "site/camera-rig/app.js").read_bytes() == (source / "app.js").read_bytes()
    result = {"status": "PASS", "application": str(app), "guide": str(guide), "guide_links_to_application": True}
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
