# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CPU-only checks of the launcher's path resolution, before setup side effects."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name("run_paibench_c.sh")


class LauncherPathsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="paibench paths ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    def framework(self, path):
        (path / "cosmos_framework").mkdir(parents=True)
        (path / "pyproject.toml").touch()
        return path

    def resolve(self, relative_script, override=None, full_script=False):
        script = self.root / relative_script
        script.parent.mkdir(parents=True, exist_ok=True)
        source = SCRIPT.read_text()
        if not full_script:
            # Exercise the real bootstrap section without installs or downloads.
            source = source[: source.index("# Cache dirs")]
            source += '\nprintf "ROOT=%s\\nREPO=%s\\n" "$COSMOS_ROOT" "$COSMOS3_REPO"\n'
        script.write_text(source)
        env = dict(os.environ)
        env.pop("COSMOS3_REPO", None)
        if override is not None:
            env["COSMOS3_REPO"] = str(override)
        # Run outside the checkout to catch accidental dependence on cwd.
        return subprocess.run(["bash", str(script)], cwd="/", env=env, capture_output=True, text=True, timeout=5)

    def assert_paths(self, result, root, repo):
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"ROOT={root}\nREPO={repo}\n", result.stdout)

    def test_framework_checkout_with_spaces(self):
        self.framework(self.root)
        result = self.resolve(".agents/skills/evaluation/cosmos3/generator/paibench_c/run_paibench_c.sh")
        self.assert_paths(result, self.root, self.root)

    def test_containing_checkout_precedes_nested_checkout(self):
        self.framework(self.root)
        self.framework(self.root / "packages/cosmos3")
        self.assert_paths(self.resolve(".agents/skills/evaluation/run.sh"), self.root, self.root)

    def test_explicit_checkout_takes_precedence(self):
        self.framework(self.root)
        other = self.framework(self.root / "other framework")
        result = self.resolve(".agents/skills/evaluation/run.sh", override=other)
        self.assert_paths(result, self.root, other)

    def test_explicit_clone_destination_is_preserved(self):
        self.framework(self.root)
        destination = self.root / "new checkout"
        result = self.resolve(".agents/skills/evaluation/run.sh", override=destination)
        self.assert_paths(result, self.root, destination)
        self.assertFalse(destination.exists())

    def test_legacy_nested_checkouts(self):
        (self.root / "cookbooks/cosmos3").mkdir(parents=True)
        for name in ("cosmos3", "cosmos-framework"):
            with self.subTest(name=name):
                checkout = self.framework(self.root / "packages" / name)
                result = self.resolve("evaluation/cosmos3/generator/paibench_c/run.sh")
                self.assert_paths(result, self.root, checkout)
                (checkout / "pyproject.toml").unlink()

    def test_legacy_clone_destination_is_preserved(self):
        (self.root / "cookbooks/cosmos3").mkdir(parents=True)
        result = self.resolve("evaluation/run.sh")
        self.assert_paths(result, self.root, self.root / "packages/cosmos-framework")

    def test_invalid_root_fails_before_setup(self):
        # The complete script must reject an unrelated pyproject before setup.
        (self.root / "pyproject.toml").touch()
        result = self.resolve("standalone/run.sh", full_script=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Could not locate", result.stderr)
        self.assertNotIn("Step 1", result.stdout)
        self.assertFalse((self.root / "standalone/.cache").exists())
        self.assertFalse((self.root / "standalone/outputs").exists())


if __name__ == "__main__":
    unittest.main()
