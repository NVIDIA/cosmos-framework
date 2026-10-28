# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Advisory, offline release-hygiene review of added lines or a release tree."""

import argparse
import datetime
import hashlib
import html
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

DEFAULT_RULES = Path(__file__).with_name("public_hygiene_rules.json")
MAX_FILE_BYTES = 2 * 1024 * 1024
MAX_ANNOTATIONS = 20
MAX_SUMMARY_ROWS = 100


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    rule: str
    context: str
    message: str
    line_sha256: str


def fingerprint(line: str) -> str:
    """Keep exceptions stable across line-number and indentation changes."""
    return hashlib.sha256(line.strip().encode("utf-8")).hexdigest()


def git(root: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.PIPE)


def commit(root: Path, revision: str) -> str:
    return git(root, "rev-parse", "--verify", "--end-of-options", f"{revision}^{{commit}}").decode().strip()


def load_rules(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    rules = config["rules"]
    ids = {rule["id"] for rule in rules}
    if len(ids) != len(rules):
        raise ValueError("Duplicate hygiene rule id")
    for rule in rules:
        re.compile(rule["pattern"])
        if not rule["message"].strip():
            raise ValueError("Rules must include a review message")
    for exception in config.get("exceptions", []):
        if exception["rule"] not in ids or not re.fullmatch(r"[0-9a-f]{64}", exception["line_sha256"]):
            raise ValueError("Exceptions require a known rule and a SHA-256 line fingerprint")
        if not all(exception.get(key, "").strip() for key in ("path", "owner", "reason")):
            raise ValueError("Exceptions require an exact path, owner, and reason")
        if exception.get("expires"):
            datetime.date.fromisoformat(exception["expires"])
    return config


def context(path: str, text: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix == ".ipynb":
        return "notebook text"
    if suffix in {".md", ".rst", ".txt"}:
        return "documentation/text"
    if text.lstrip().startswith(("#", "//", "<!--")):
        return "comment"
    if suffix in {".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".lock"}:
        return "configuration/data"
    return "code/text (review context)"


def scan_line(path: str, number: int, text: str, config: dict) -> list[Finding]:
    findings = []
    location = context(path, text)
    digest = fingerprint(text)
    for rule in config["rules"]:
        # Release wording is not a ban on identifiers, licenses, or GA=gradient accumulation.
        if rule.get("prose_only") and (
            location not in {"documentation/text", "notebook text", "comment"}
            or Path(path).name.upper().startswith(("LICENSE", "COPYING", "ATTRIBUTIONS", "NOTICE"))
        ):
            continue
        matches = list(re.finditer(rule["pattern"], text))
        if rule.get("dependency_source"):
            approved = {owner.casefold() for owner in config["dependency_approved_owners"]}
            matches = [
                match
                for match in matches
                if (match.group("owner") or match.group("https_owner")).casefold() not in approved
            ]
        if not matches:
            continue
        if any(
            item["rule"] == rule["id"]
            and item["path"] == path
            and item["line_sha256"] == digest
            and (
                not item.get("expires")
                or datetime.date.fromisoformat(item["expires"]) >= datetime.datetime.now(datetime.timezone.utc).date()
            )
            for item in config.get("exceptions", [])
        ):
            continue
        findings.append(Finding(path, number, rule["id"], location, rule["message"], digest))
    return findings


def added_lines(patch: str):
    """Read new-side line numbers; filenames are obtained separately with -z."""
    number = None
    for line in patch.splitlines():
        hunk = re.match(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@", line)
        if hunk:
            number = int(hunk.group(1))
        elif number is not None:
            if line.startswith("+"):
                yield number, line[1:]
                number += 1
            elif line.startswith(" "):
                number += 1


def read_text(data: bytes) -> str | None:
    if len(data) > MAX_FILE_BYTES or b"\0" in data:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def changed_paths(data: bytes):
    entries = iter(data.split(b"\0"))
    for status in entries:
        if not status:
            continue
        old_path = os.fsdecode(next(entries))
        if status.startswith(b"R"):
            yield os.fsdecode(next(entries)), old_path
        else:
            yield old_path, old_path


def scan_diff(root: Path, base: str, head: str | None, config: dict) -> tuple[list[Finding], int]:
    current = commit(root, head or "HEAD")
    ancestor = git(root, "merge-base", commit(root, base), current).decode().strip()
    revisions = [ancestor, current] if head else [ancestor]
    paths = git(root, "diff", "--name-status", "-z", "--find-renames", "--diff-filter=AMRT", *revisions, "--")
    findings = []
    skipped = 0
    for path, previous in changed_paths(paths):
        if head:
            entry = git(root, "ls-tree", current, "--", path)
            if not entry.startswith((b"100644 ", b"100755 ")):
                skipped += 1
                continue
            # Check size before reading a large blob into the scanner.
            if int(git(root, "cat-file", "-s", f"{current}:{path}")) > MAX_FILE_BYTES:
                skipped += 1
                continue
            data = git(root, "show", f"{current}:{path}")
        else:
            target = root / path
            if target.is_symlink() or not target.is_file() or not target.resolve().is_relative_to(root):
                skipped += 1
                continue
            if target.stat().st_size > MAX_FILE_BYTES:
                skipped += 1
                continue
            data = target.read_bytes()
        if read_text(data) is None:
            skipped += 1
            continue
        patch = git(
            root,
            "diff",
            "--no-ext-diff",
            "--no-textconv",
            "--find-renames",
            "--unified=0",
            *revisions,
            "--",
            previous,
            path,
        )
        for number, text in added_lines(patch.decode("utf-8", errors="replace")):
            findings.extend(scan_line(path, number, text, config))
    return findings, skipped


def scan_tree(root: Path, config: dict) -> tuple[list[Finding], int]:
    if (root / ".git").exists():
        # Never inspect private/untracked worktree files during a repository-wide scan.
        paths = [root / os.fsdecode(path) for path in git(root, "ls-files", "-z").split(b"\0") if path]
    else:
        paths = []
        for directory, dirs, files in os.walk(root, followlinks=False):
            dirs[:] = [name for name in dirs if name != ".git" and not (Path(directory) / name).is_symlink()]
            paths.extend(Path(directory) / name for name in files)
    findings = []
    skipped = 0
    for path in paths:
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
            skipped += 1
            continue
        text = read_text(path.read_bytes()) if path.stat().st_size <= MAX_FILE_BYTES else None
        if text is None:
            skipped += 1
            continue
        relative = path.relative_to(root).as_posix()
        for number, line in enumerate(text.splitlines(), 1):
            findings.extend(scan_line(relative, number, line, config))
    return findings, skipped


def escape_command(value: str, *, property_value: bool = False) -> str:
    value = value.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    if property_value:
        value = value.replace(":", "%3A").replace(",", "%2C")
    return value


def report(findings: list[Finding], skipped: int, github: bool) -> None:
    print(f"Public repo hygiene: {len(findings)} review hints; {skipped} non-text/large/symlink files skipped.")
    print(
        "Advisory only: hints are not proof of a private resource or a broken public workflow. No source excerpts emitted."
    )
    for index, item in enumerate(findings):
        if github and index < MAX_ANNOTATIONS:
            print(
                f"::notice file={escape_command(item.path, property_value=True)},line={item.line},"
                f"title=Public repo hygiene::{escape_command(item.rule + ': ' + item.message)}"
            )
        elif not github:
            # JSON quoting prevents control characters in filenames/messages becoming log commands.
            print(json.dumps(item.__dict__, ensure_ascii=True))
    if github and (summary := os.environ.get("GITHUB_STEP_SUMMARY")):
        rows = [
            "## Public repo hygiene (advisory)",
            "",
            f"{len(findings)} review hints; {skipped} non-text/large/symlink files skipped.",
            "Hints do not fail this check. They do not establish public-workflow reachability or resource privacy.",
            "No source excerpts or matched resource values are included. Existing Gitleaks checks remain separate.",
            "",
            "| File:line | Rule | Context | Review hint | Line SHA-256 for a scoped exception |",
            "| --- | --- | --- | --- | --- |",
        ]
        for item in findings[:MAX_SUMMARY_ROWS]:
            cells = [f"{item.path}:{item.line}", item.rule, item.context, item.message, item.line_sha256]
            rows.append(
                "| "
                + " | ".join(
                    html.escape(cell).replace("|", "&#124;").replace("\n", " ").replace("\r", " ") for cell in cells
                )
                + " |"
            )
        if len(findings) > MAX_SUMMARY_ROWS:
            rows.extend(["", f"Showing the first {MAX_SUMMARY_ROWS} hints. Run locally to review the complete list."])
        with Path(summary).open("a", encoding="utf-8") as stream:
            stream.write("\n".join(rows) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--base", help="Compare added lines against the merge base with this Git revision")
    mode.add_argument(
        "--root", type=Path, help="Scan tracked repository files, or all files in an exported release tree"
    )
    parser.add_argument(
        "--head", help="Read this committed PR head instead of the local working tree (requires --base)"
    )
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES)
    parser.add_argument("--github", action="store_true", help="Emit bounded notices and a GitHub step summary")
    args = parser.parse_args()
    if args.head and not args.base:
        parser.error("--head requires --base")
    try:
        config = load_rules(args.rules)
        if args.root:
            root = args.root.resolve(strict=True)
            if not root.is_dir():
                raise ValueError("--root must be a directory")
            findings, skipped = scan_tree(root, config)
        else:
            root = Path(os.fsdecode(git(Path.cwd(), "rev-parse", "--show-toplevel")).strip()).resolve()
            findings, skipped = scan_diff(root, args.base, args.head, config)
        report(findings, skipped, args.github)
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError):
        # Do not echo subprocess output or config content into public CI logs.
        print("Hygiene scan could not complete; check the revision, root, and rules configuration.", file=sys.stderr)
        return 2
    return 0  # Review hints never change the exit status.


if __name__ == "__main__":
    sys.exit(main())
