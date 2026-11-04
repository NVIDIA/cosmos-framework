# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only boundary tests; no model imports, network, or public media downloads."""
import hashlib
import importlib.util
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch
import zipfile

spec = importlib.util.spec_from_file_location('launchable_mmsivideo', Path(__file__).with_name('mmsivideo.py'))
mmsivideo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mmsivideo)


class PublicStagingTests(unittest.TestCase):
    def test_rejects_traversal_and_absolute_archive_members(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('../escape', '/absolute', 'frames/../../escape', 'frames\\escape'):
                with self.subTest(name=name):
                    with zipfile.ZipFile(root / 'frames.zip', 'w') as archive:
                        archive.writestr(name, b'image')
                    with self.assertRaises(ValueError):
                        mmsivideo._extract(root, 'frames.zip')

    def test_rejects_archive_and_existing_directory_symlinks(self):
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as external:
            root = Path(directory)
            info = zipfile.ZipInfo('frames/link')
            info.create_system = 3
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            with zipfile.ZipFile(root / 'frames.zip', 'w') as archive:
                archive.writestr(info, '../../escape')
            with self.assertRaises(ValueError):
                mmsivideo._extract(root, 'frames.zip')
            (root / 'frames').symlink_to(external, target_is_directory=True)
            with zipfile.ZipFile(root / 'frames.zip', 'w') as archive:
                archive.writestr('frames/image.jpg', b'image')
            with self.assertRaises(ValueError):
                mmsivideo._extract(root, 'frames.zip')
            self.assertFalse((Path(external) / 'image.jpg').exists())

    def test_extracts_nested_canonical_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with zipfile.ZipFile(root / 'frames.zip', 'w') as archive:
                archive.writestr('frames/clip/001.jpg', b'canonical image')
            result = mmsivideo._extract(root, 'frames.zip')
            self.assertEqual(result['files'], 1)
            self.assertEqual((root / 'frames/clip/001.jpg').read_bytes(), b'canonical image')

    def test_download_checksum_failure_never_promotes_partial(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'public.bin'
            source.write_bytes(b'changed public data')
            with self.assertRaises(ValueError):
                mmsivideo._fetch(root, 'download.bin', source.as_uri(), hashlib.sha256(b'pinned').hexdigest())
            self.assertFalse((root / 'download.bin').exists())

    def test_missing_assets_fail_before_dataset_or_endpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(mmsivideo, '_dataset') as dataset:
                with self.assertRaises(ValueError):
                    mmsivideo.run(root / 'data', 'full', root / 'work')
                dataset.assert_not_called()
            failure = root / 'work' / 'failure.json'
            self.assertTrue(failure.is_file())
            self.assertFalse((root / 'work' / 'summary.json').exists())

    def test_refuses_reusing_work_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            work = root / 'work'
            work.mkdir()
            with patch.object(mmsivideo, '_verify_sources') as verify:
                with self.assertRaises(FileExistsError):
                    mmsivideo.run(root / 'data', 'smoke', work)
                verify.assert_not_called()

    def test_caps_endpoint_concurrency_before_creating_work(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for workers in (0, 5, -1, 1.5, True):
                with self.subTest(workers=workers), self.assertRaises(ValueError):
                    mmsivideo.run(root / 'data', 'smoke', root / 'work', workers)
            self.assertFalse((root / 'work').exists())

    def test_refuses_output_inside_source_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(mmsivideo.__file__).parent / 'forbidden-mmsi-test-output'
            with patch.object(mmsivideo, '_verify_sources') as verify:
                with self.assertRaises(ValueError):
                    mmsivideo.run(Path(directory), 'smoke', work)
                verify.assert_not_called()
            self.assertFalse(work.exists())

    def test_truncation_preserves_http_success_but_not_http_failure(self):
        self.assertTrue(mmsivideo._successful_transport(0, 200, None))
        self.assertTrue(mmsivideo._successful_transport(1, 200, 'truncated'))
        self.assertFalse(mmsivideo._successful_transport(1, 200, 'http_error'))
        self.assertFalse(mmsivideo._successful_transport(500, 500, 'truncated'))
        self.assertFalse(mmsivideo._successful_transport(0, 500, None))


if __name__ == '__main__':
    unittest.main()
