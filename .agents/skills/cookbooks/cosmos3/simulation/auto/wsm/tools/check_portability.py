"""Audit standalone source, wheels, source archives, and bundled scene archives."""

import argparse
import ast
import io
import re
import tarfile
import zipfile
from pathlib import Path, PurePosixPath


FORBIDDEN_IMPORTS = {"imaginaire", "projects", "packages", "cosmos_framework"}
# Assemble patterns so that this checker is itself scanned like every other file.
FORBIDDEN_TEXT = re.compile(
    rb"/(?:" + b"lustre|home|mnt|workspace|nfs|raid|scratch" + rb")/"
    + rb"|" + b"gitlab" + rb"[\w-]*\.nvidia\.com"
    + rb"|" + rb"[\w-]+\.slack\.com"
    + rb"|" + b"packages/" + rb"cosmos[\w-]*"
    + rb"|" + b"credentials/" + rb"[^\s\"']+\.secret"
    # Any concrete bucket; tests use the placeholder s3://bucket/.
    + rb"|" + b"s3://" + rb"(?!bucket/)[a-z0-9]"
)
LFS_POINTER = b"version https://git-lfs.github.com/spec/" + b"v1\n"
SKIP = {".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache", "build", "dist"}


def inspect_bytes(name: str, data: bytes, errors: list[str]) -> int:
    if data.startswith(LFS_POINTER):
        errors.append(f"{name}: unhydrated Git LFS resource")
        return 1
    match = FORBIDDEN_TEXT.search(data)
    if match:
        errors.append(f"{name}: machine-specific path or internal service reference: {match.group()!r}")
    if name.endswith(".py"):
        try:
            tree = ast.parse(data, filename=name)
        except (SyntaxError, UnicodeError) as exc:
            errors.append(f"{name}: {exc}")
        else:
            for node in ast.walk(tree):
                modules = []
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    modules = [node.module]
                for module in modules:
                    if module.split(".")[0] in FORBIDDEN_IMPORTS:
                        errors.append(f"{name}:{node.lineno}: external repository import {module}")
    count = 1
    if name.endswith((".whl", ".zip", ".npz")):
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            for entry in archive.infolist():
                if not entry.is_dir():
                    count += inspect_bytes(f"{name}!{entry.filename}", archive.read(entry), errors)
    elif name.endswith((".tar", ".tar.gz", ".tgz")):
        with tarfile.open(fileobj=io.BytesIO(data)) as archive:
            for entry in archive:
                path = PurePosixPath(entry.name)
                if path.is_absolute() or ".." in path.parts or entry.issym() or entry.islnk():
                    errors.append(f"{name}!{entry.name}: nonportable archive member")
                if entry.isfile():
                    stream = archive.extractfile(entry)
                    assert stream is not None
                    count += inspect_bytes(f"{name}!{entry.name}", stream.read(), errors)
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, default=[Path(__file__).resolve().parents[1]])
    args = parser.parse_args()
    errors: list[str] = []
    count = 0
    for target in args.paths:
        if not target.exists():
            errors.append(f"Missing audit target: {target}")
            continue
        for path in sorted(target.rglob("*")) if target.is_dir() else [target]:
            relative = path.relative_to(target) if target.is_dir() else Path(path.name)
            if any(part in SKIP or part.endswith(".egg-info") for part in relative.parts):
                continue
            if path.is_symlink():
                errors.append(f"{relative}: symbolic link; standalone resources must be local files")
            elif path.is_file():
                count += inspect_bytes(str(relative), path.read_bytes(), errors)
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Portability audit passed: {count} files and archive members")


if __name__ == "__main__":
    main()
