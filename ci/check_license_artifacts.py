# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Check the third-party license files in built distributions."""

import argparse
import tarfile
import tomllib
from collections.abc import Callable
from pathlib import Path
from zipfile import ZipFile


def verify_license_files(project: Path, dist: Path) -> None:
    metadata = tomllib.loads((project / "pyproject.toml").read_text())["project"]
    expected = {}
    for pattern in metadata["license-files"]:
        matches = sorted(path for path in project.glob(pattern) if path.is_file())
        if not matches:
            raise SystemExit(f"{project}: license pattern matches no files: {pattern}")
        for path in matches:
            expected[path.relative_to(project).as_posix()] = path.read_bytes()

    notice = expected.get("NOTICE", b"").decode()
    if "THIRD_PARTY_LICENSES" in notice and "THIRD_PARTY_LICENSES" not in expected:
        raise SystemExit(f"{project}: NOTICE references an unbundled license file: THIRD_PARTY_LICENSES")

    wheels = sorted(dist.glob("*.whl"))
    sdists = sorted(dist.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise SystemExit(f"{dist}: expected exactly one wheel and one source distribution")

    def verify(artifact: Path, names: list[str], read_file: Callable[[str], bytes], suffix: str) -> None:
        for relative, content in expected.items():
            matches = [name for name in names if name.split("/", 1)[-1] == suffix + relative]
            if len(matches) != 1 or read_file(matches[0]) != content:
                raise SystemExit(f"{artifact}: missing or changed license file: {relative}")

    wheel = wheels[0]
    with ZipFile(wheel) as archive:
        names = [name for name in archive.namelist() if name.split("/", 1)[0].endswith(".dist-info")]
        verify(wheel, names, archive.read, "licenses/")

    sdist = sdists[0]
    with tarfile.open(sdist, "r:gz") as archive:
        names = [member.name for member in archive.getmembers() if member.isfile()]

        def read_source_file(name: str) -> bytes:
            stream = archive.extractfile(name)
            if stream is None:
                raise SystemExit(f"{sdist}: unable to read license file: {name}")
            with stream:
                return stream.read()

        verify(sdist, names, read_source_file, "")

    print(f"Verified {len(expected)} license files in {wheel.name} and {sdist.name}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path)
    parser.add_argument("dist", type=Path)
    args = parser.parse_args()
    verify_license_files(args.project, args.dist)
