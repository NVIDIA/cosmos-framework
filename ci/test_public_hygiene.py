# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Small, standalone checks: no pytest collection, Cosmos imports, or GPU."""

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import check_public_hygiene as hygiene


def internal_link() -> str:
    return "https://" + "gitlab-master" + ".nvidia.com/team/project/-/merge_requests/123"


class RuleTests(unittest.TestCase):
    def setUp(self):
        self.config = hygiene.load_rules(hygiene.DEFAULT_RULES)

    def rules(self, text, path="example.py"):
        return {item.rule for item in hygiene.scan_line(path, 7, text, self.config)}

    def test_internal_link_and_resources(self):
        self.assertIn("internal-link", self.rules(internal_link()))
        self.assertIn("internal-resource", self.rules("/" + "lustre/fsw/portfolios/team/data"))
        self.assertIn("internal-resource", self.rules("https://" + "pdx.s8k.io/data"))
        self.assertIn("internal-resource", self.rules("s3://" + "cosmos_understanding/benchmark/data"))

    def test_public_and_placeholder_resources(self):
        for text in (
            "https://github.com/NVIDIA/cosmos-framework/pull/1",
            "s3://bucket/data",
            "https://docs.nvidia.com",
        ):
            self.assertEqual(self.rules(text), set())

    def test_dependency_sources_are_not_account_classification(self):
        for prefix in ("git+https://", "https://", "git+ssh://git@"):
            self.assertIn("dependency-source", self.rules(prefix + "github.com/example-user/library.git"))
            self.assertNotIn("dependency-source", self.rules(prefix + "github.com/NVIDIA/library.git"))
        self.assertNotIn("dependency-source", self.rules("https://github.com/example-user/library"))
        self.assertNotIn(
            "dependency-source", self.rules("https://github.com/NVIDIA/library.git git+https://github.com/nvidia/other")
        )
        self.assertIn(
            "dependency-source", self.rules("git+https://github.com/NVIDIA/a git+https://github.com/example-user/b")
        )

    def test_known_model_pattern_only(self):
        self.assertIn("model-access-review", self.rules("nvidia/" + "Cosmos3-Experimental"))
        self.assertNotIn("model-access-review", self.rules("nvidia/Cosmos3-Nano"))

    def test_release_wording_is_contextual(self):
        wording = "before " + "GA"
        self.assertIn("release-wording", self.rules(wording, "guide.md"))
        self.assertIn("release-wording", self.rules("# " + wording))
        self.assertIn("release-wording", self.rules(json.dumps(wording), "recipe.ipynb"))
        self.assertNotIn("release-wording", self.rules("GA = gradient_accumulation"))
        self.assertNotIn("release-wording", self.rules(wording))
        self.assertNotIn("release-wording", self.rules(wording, "ATTRIBUTIONS.md"))

    def test_development_provenance(self):
        self.assertIn("internal-provenance", self.rules("# " + "internal" + " review reference"))
        self.assertEqual(self.rules("# Copyright NVIDIA; MR #123; ordinary technical terminology"), set())

    def test_exact_exception_and_expiration(self):
        text = internal_link()
        self.config["exceptions"] = [
            {
                "rule": "internal-link",
                "path": "example.py",
                "line_sha256": hygiene.fingerprint(text),
                "owner": "release-maintainers",
                "reason": "Intentional test input",
                "expires": "2999-01-01",
            }
        ]
        self.assertEqual(self.rules("  " + text), set())
        self.assertIn("internal-link", self.rules(text + "/changed"))
        self.assertIn("internal-link", self.rules(text, "other.py"))
        self.config["exceptions"][0]["expires"] = "2000-01-01"
        self.assertIn("internal-link", self.rules(text))

    def test_added_line_numbers(self):
        diff = "--- a/file\n+++ b/file\n@@ -2,2 +2,3 @@\n-old\n+new\n same\n+extra\n@@ -10 +11 @@\n-x\n+y\n"
        self.assertEqual(list(hygiene.added_lines(diff)), [(2, "new"), (4, "extra"), (11, "y")])

    def test_binary_and_large_files(self):
        self.assertIsNone(hygiene.read_text(b"binary\0data"))
        self.assertIsNone(hygiene.read_text(b"\xff"))
        self.assertIsNone(hygiene.read_text(b"x" * (hygiene.MAX_FILE_BYTES + 1)))


class GitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.git("init", "-q")
        self.git("config", "user.name", "Scanner test")
        self.git("config", "user.email", "scanner@example.invalid")
        self.config = hygiene.load_rules(hygiene.DEFAULT_RULES)
        (self.root / "existing.md").write_text(internal_link() + "\nkeep\n", encoding="utf-8")
        self.base = self.save()

    def git(self, *args):
        return hygiene.git(self.root, *args).decode().strip()

    def save(self):
        self.git("add", "--all")
        self.git("-c", "core.hooksPath=/dev/null", "commit", "--no-gpg-sign", "-qm", "fixture")
        return self.git("rev-parse", "HEAD")

    def test_only_added_lines_and_odd_filenames(self):
        path = 'sp ace,quote"-文.md'
        (self.root / path).write_text("safe\n" + internal_link() + "\n", encoding="utf-8")
        head = self.save()
        found, skipped = hygiene.scan_diff(self.root, self.base, head, self.config)
        self.assertEqual([(item.path, item.line) for item in found], [(path, 2)])
        self.assertEqual(skipped, 0)

    def test_renamed_history_does_not_become_new(self):
        self.git("mv", "existing.md", "renamed.md")
        head = self.save()
        self.assertEqual(hygiene.scan_diff(self.root, self.base, head, self.config)[0], [])
        with (self.root / "renamed.md").open("a", encoding="utf-8") as stream:
            stream.write(internal_link() + "\n")
        updated = self.save()
        found, _ = hygiene.scan_diff(self.root, self.base, updated, self.config)
        self.assertEqual([(item.path, item.line) for item in found], [("renamed.md", 3)])

    def test_explicit_head_ignores_worktree(self):
        (self.root / "existing.md").write_text(internal_link() + "\nkeep\n" + internal_link(), encoding="utf-8")
        self.assertEqual(hygiene.scan_diff(self.root, self.base, self.base, self.config)[0], [])
        self.assertEqual(len(hygiene.scan_diff(self.root, self.base, None, self.config)[0]), 1)

    def test_merge_base_ignores_target_branch_only_changes(self):
        self.git("checkout", "-qb", "target")
        (self.root / "target-only.md").write_text(internal_link(), encoding="utf-8")
        target = self.save()
        self.git("checkout", "-qb", "topic", self.base)
        (self.root / "safe.md").write_text("safe", encoding="utf-8")
        head = self.save()
        self.assertEqual(hygiene.scan_diff(self.root, target, head, self.config)[0], [])

    def test_agents_fixtures_and_notebook_are_included(self):
        paths = [".agents/skills/cookbooks/guide.md", "tests/data/fixture.json", "example.ipynb"]
        for name in paths:
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(internal_link()), encoding="utf-8")
        head = self.save()
        self.assertEqual(
            {item.path for item in hygiene.scan_diff(self.root, self.base, head, self.config)[0]}, set(paths)
        )

    def test_full_scan_omits_untracked_files_and_symlinks(self):
        (self.root / "private-untracked.md").write_text(internal_link(), encoding="utf-8")
        (self.root / "link.md").symlink_to(self.root / "existing.md")
        self.git("add", "link.md")
        found, skipped = hygiene.scan_tree(self.root, self.config)
        self.assertEqual([item.path for item in found], ["existing.md"])
        self.assertEqual(skipped, 1)

    def test_cli_findings_do_not_fail(self):
        result = subprocess.run(
            [sys.executable, str(Path(hygiene.__file__)), "--root", str(self.root)],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("1 review hints", result.stdout)
        self.assertNotIn(internal_link(), result.stdout)

    def test_bad_revision_is_a_tooling_error_not_a_green_scan(self):
        result = subprocess.run(
            [sys.executable, str(Path(hygiene.__file__)), "--base", "missing-revision"],
            cwd=self.root,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertNotIn("0 review hints", result.stdout)


class OutputTests(unittest.TestCase):
    def test_export_directory_does_not_follow_symlink_directories(self):
        with tempfile.TemporaryDirectory() as export, tempfile.TemporaryDirectory() as outside:
            root = Path(export).resolve()
            (Path(outside) / "private.md").write_text(internal_link(), encoding="utf-8")
            (root / "linked").symlink_to(outside, target_is_directory=True)
            (root / "guide.md").write_text(internal_link(), encoding="utf-8")
            found, _ = hygiene.scan_tree(root, hygiene.load_rules(hygiene.DEFAULT_RULES))
            self.assertEqual([item.path for item in found], ["guide.md"])

    def test_output_escaping_bounds_and_no_source_excerpts(self):
        found = hygiene.Finding("a,\n::error::bad|<b>.md", 2, "rule", "text", "review\n::error::bad", "0" * 64)
        with tempfile.TemporaryDirectory() as directory:
            summary = Path(directory) / "summary.md"
            output = io.StringIO()
            with patch.dict(os.environ, {"GITHUB_STEP_SUMMARY": str(summary)}), contextlib.redirect_stdout(output):
                hygiene.report([found] * 105, 0, github=True)
            self.assertEqual(output.getvalue().count("::notice file="), hygiene.MAX_ANNOTATIONS)
            self.assertNotIn("\n::error::", output.getvalue())
            self.assertIn("%2C%0A%3A%3Aerror", output.getvalue())
            self.assertIn("&lt;b&gt;", summary.read_text())
            self.assertIn("Showing the first 100", summary.read_text())

    def test_invalid_exception_is_rejected(self):
        config = hygiene.load_rules(hygiene.DEFAULT_RULES)
        config["exceptions"] = [{"rule": "internal-link", "line_sha256": "0" * 64, "path": "x.py"}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rules.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaises(ValueError):
                hygiene.load_rules(path)


if __name__ == "__main__":
    unittest.main()
