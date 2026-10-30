# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Synthetic consumer-boundary tests; no network, GPU, or private media.

The fixture trust root is patched to hashes of synthetic annotations, never to
bypass authentication. Counts, native task ordering, preparation, image reading,
native exact matching, and extraction all execute their real implementations.
"""
import contextlib
import io
import json
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import pandas as pd
from PIL import Image

from vlmeval import mvbench_local as local
from vlmeval.dataset.mvbench import MVBench
from vlmeval.smp import get_intermediate_file_path


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


class MVBenchTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data = self.root / 'data'
        self.hashes = {}
        self.sources = {}
        for task, (name, prefix, data_type, bound) in MVBench.TASKS.items():
            rows = []
            for index in range(200):
                row = {
                    'video': 'fixture' if data_type == 'frame' else 'fixture.mp4',
                    'question': f'Choose the shape in synthetic {task} question {index}.',
                    'candidates': ['red square', 'blue circle', 'green triangle'],
                    'answer': ['red square', 'blue circle', 'green triangle'][index % 3],
                }
                if bound:
                    row.update(start=0, end=2)
                rows.append(row)
            self.sources[name] = rows
            path = self.data / 'json' / name
            _write_json(path, rows)
            self.hashes[f'json/{name}'] = local._sha256(path)
        for name in ('README.md', 'video/MVBench_videos_ntu.txt'):
            path = self.data / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('Synthetic public-input fixture, no actual dataset media.\n')
            self.hashes[name] = local._sha256(path)
        self.trust = patch.dict(local.METADATA_SHA256, self.hashes, clear=True)
        self.trust.start()
        self.addCleanup(self.trust.stop)
        self._media(local.SMOKE_TASK)

    def _media(self, task):
        name, prefix, data_type, bound = MVBench.TASKS[task]
        path = self.data / prefix.replace('your_data_path', 'video') / self.sources[name][0]['video']
        if data_type == 'frame':
            path.mkdir(parents=True, exist_ok=True)
            for index in range(1, 10):
                color = (255, 0, 0) if index <= 3 else (0, 0, 255)
                Image.new('RGB', (320, 240), color).save(path / f'{index:05d}.jpg')
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Scoring/coverage fixtures need identity and nonempty media, not a
            # decoder. The prompt test below uses real numbered JPEG inputs.
            path.write_bytes(b'Synthetic scoring-only media identity; not decoded.')
        return path

    def _repin(self, name):
        path = self.data / 'json' / name
        _write_json(path, self.sources[name])
        local.METADATA_SHA256[f'json/{name}'] = local._sha256(path)

    def _table(self):
        prepared = local.prepare_dataset(self.data, smoke=True)
        table = pd.read_csv(prepared['data_file'], sep='\t')
        table['prediction'] = ['(A)', '(B)', '(C)', '(A)', '(B)']
        return table

    def _predictions(self, table=None, name='predictions.tsv'):
        path = self.root / name
        (self._table() if table is None else table).to_csv(path, sep='\t', index=False)
        return path

    def _score(self, predictions=None, name='results'):
        path = self._predictions() if predictions is None else predictions
        with contextlib.redirect_stdout(io.StringIO()):
            return local.score_predictions(path, self.data, self.root / name, smoke=True)

    def test_smoke_is_fixed_subset_and_never_a_full_score(self):
        prepared = local.prepare_dataset(self.data, smoke=True)
        self.assertEqual(prepared['selected_indices'], list(range(3400, 3405)))
        self.assertEqual(prepared['coverage']['expected_rows'], 4000)
        self.assertEqual(prepared['coverage']['available_rows'], 200)
        self.assertEqual(prepared['coverage']['selected_rows'], 5)
        self.assertEqual(prepared['coverage']['selected_tasks'], ['Egocentric Navigation'])
        self.assertTrue(prepared['coverage']['subset'])
        with self.assertRaises(local.MVBenchCoverageError) as caught:
            local.prepare_dataset(self.data)
        self.assertIn('4000', str(caught.exception))
        self.assertIn('video/nturgbd', str(caught.exception))
        self.assertIn('Fine-grained Pose', caught.exception.report['missing_media_by_task'])
        self.assertTrue((self.data / '.mvbench/coverage-full.json').is_file())
        summary = self._score()
        self.assertEqual(summary['status'], 'smoke_only')
        self.assertEqual(summary['primary_metric']['value'], 1.0)
        self.assertEqual(summary['primary_metric']['total'], 5)
        self.assertFalse(summary['provenance']['paper_parity'])
        self.assertEqual(json.loads((self.root / 'results/summary.json').read_text()), summary)

    def test_full_requires_manual_ntu_and_does_not_drop_its_rows(self):
        for task in MVBench.TASKS:
            if task != 'Fine-grained Pose':
                self._media(task)
        with self.assertRaises(local.MVBenchCoverageError) as caught:
            local.prepare_dataset(self.data)
        self.assertEqual(caught.exception.report['coverage']['available_rows'], 3800)
        self.assertEqual(set(caught.exception.report['missing_media_by_task']), {'Fine-grained Pose'})
        self._media('Fine-grained Pose')
        full = local.prepare_dataset(self.data)
        self.assertEqual(full['coverage']['available_rows'], 4000)
        self.assertEqual(full['coverage']['selected_rows'], 4000)
        self.assertEqual(len(full['coverage']['selected_tasks']), 20)
        self.assertFalse(full['coverage']['subset'])

    def test_source_metadata_and_prebuilt_tsv_cannot_replace_ground_truth(self):
        prepared = local.prepare_dataset(self.data, smoke=True)
        canonical = Path(prepared['data_file'])
        canonical.write_text('index\tanswer\n3400\tforged\n')
        repaired = local.prepare_dataset(self.data, smoke=True)
        self.assertEqual(len(pd.read_csv(repaired['data_file'], sep='\t')), 5)
        source = self.data / 'json/egocentric_navigation.json'
        source.write_text(source.read_text().replace('red square', 'forged square'))
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            local.prepare_dataset(self.data, smoke=True)

    def test_media_content_mutation_changes_manifest_not_just_path(self):
        first = local.prepare_dataset(self.data, smoke=True)
        video = self.data / next(iter(first['media_sha256']))
        video.write_bytes(b'Changed synthetic media at the same path.')
        second = local.prepare_dataset(self.data, smoke=True)
        self.assertNotEqual(first['media_manifest_sha256'], second['media_manifest_sha256'])
        self.assertNotEqual(first['manifest_sha256'], second['manifest_sha256'])
        self.assertEqual(first['annotation_manifest_sha256'], second['annotation_manifest_sha256'])

    def test_missing_bad_answer_and_changed_official_row_count_are_rejected(self):
        name = 'egocentric_navigation.json'
        for answer in (None, 'not a candidate'):
            with self.subTest(answer=answer):
                self.sources[name][0]['answer'] = answer
                self._repin(name)
                with self.assertRaisesRegex(ValueError, 'Answer must identify'):
                    local.prepare_dataset(self.data, smoke=True)
        self.sources[name][0]['answer'] = 'red square'
        self.sources[name].pop()
        self._repin(name)
        with self.assertRaisesRegex(ValueError, 'exactly 200'):
            local.prepare_dataset(self.data, smoke=True)

    def test_empty_and_missing_media_cannot_enter_scoring(self):
        video = self._media(local.SMOKE_TASK)
        for remove in (False, True):
            with self.subTest(remove=remove):
                if remove:
                    video.unlink()
                else:
                    video.write_bytes(b'')
                with self.assertRaises(local.MVBenchCoverageError):
                    local.prepare_dataset(self.data, smoke=True)

    def test_legitimate_one_and_five_candidate_questions_are_not_rewritten(self):
        name = 'egocentric_navigation.json'
        self.sources[name][0].update(candidates=['red square'], answer='red square')
        self.sources[name][1].update(
            candidates=['red square', 'blue circle', 'green triangle', 'yellow star', 'purple pentagon'],
            answer='purple pentagon',
        )
        self._repin(name)
        table = self._table()
        table.loc[1, 'prediction'] = '(E)'
        summary = self._score(self._predictions(table))
        self.assertEqual(summary['primary_metric']['correct'], 5)

    def test_prediction_identity_coverage_is_exact_not_just_row_count(self):
        original = self._table()
        cases = {
            'missing': original.iloc[:-1].copy(),
            'duplicate': pd.concat([original.iloc[:-1], original.iloc[:1]], ignore_index=True),
            'extra': pd.concat([original, original.iloc[:1].assign(index=99999)], ignore_index=True),
            'substituted': original.assign(index=[3400, 3401, 3402, 3403, 99999]),
        }
        for name, table in cases.items():
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, 'coverage must match'):
                    self._score(self._predictions(table, name + '.tsv'), name)
                self.assertFalse((self.root / name / 'summary.json').exists())
        reordered = original.iloc[::-1]
        summary = self._score(self._predictions(reordered, 'reordered.tsv'))
        self.assertEqual(summary['primary_metric']['correct'], 5)

    def test_tampered_identity_labels_candidates_and_prompt_are_rejected(self):
        marker = self.root / 'executed'
        changes = {
            'answer': 'forged answer', 'video': '../elsewhere.mp4',
            'task_type': 'Action Sequence', 'question': 'forged question',
            'candidates': f'__import__("pathlib").Path({str(marker)!r}).touch()',
        }
        for column, value in changes.items():
            with self.subTest(column=column):
                table = self._table()
                table.loc[0, column] = value
                with self.assertRaisesRegex(ValueError, 'Untrusted'):
                    self._score(self._predictions(table, f'{column}.tsv'), column)
        self.assertFalse(marker.exists())
        # Only index/prediction is also valid; all answer identities are rebuilt.
        summary = self._score(self._predictions(self._table()[['index', 'prediction']], 'minimal.tsv'))
        self.assertEqual(summary['primary_metric']['correct'], 5)

    def test_failure_rows_never_receive_truthy_minus_one_credit(self):
        for prediction in ('', '  ', 'Failed to obtain answer via API.', 'VIDEO_LOAD_FAILED',
                           'Failed to obtain answer: RuntimeError synthetic failure'):
            with self.subTest(prediction=prediction):
                table = self._table()
                table.loc[0, 'prediction'] = prediction
                with self.assertRaisesRegex(ValueError, 'Missing or failed prediction'):
                    self._score(self._predictions(table), 'failed')
                self.assertFalse((self.root / 'failed/summary.json').exists())

    def test_existing_score_output_is_preserved(self):
        output = self.root / 'existing'
        output.mkdir()
        sentinel = output / 'summary.json'
        sentinel.write_text('previous result')
        with self.assertRaisesRegex(ValueError, 'fresh output'):
            local.score_predictions(self._predictions(), self.data, output, smoke=True)
        self.assertEqual(sentinel.read_text(), 'previous result')

    def test_nonempty_malformed_predictions_are_native_wrong_answers_offline(self):
        table = self._table()
        # These strings must not become pandas NA or numeric scalars in native
        # scoring. Native model=None deterministically gives unknown option Z.
        table['prediction'] = ['nan', '1', '.', '(', ')']
        with patch('vlmeval.dataset.mvbench.build_judge', side_effect=AssertionError('network judge forbidden')):
            summary = self._score(self._predictions(table))
        self.assertEqual(summary['primary_metric']['correct'], 0)
        self.assertEqual(summary['primary_metric']['total'], 5)
        self.assertFalse(summary['provenance']['network_judge'])
        scores = pd.read_csv(self.root / 'results/scored_predictions.tsv', sep='\t')
        self.assertEqual(scores['score'].tolist(), [0] * 5)
        self.assertEqual(scores['index'].tolist(), list(range(3400, 3405)))

    def test_native_micro_accuracy_is_not_task_macro_for_unequal_sizes(self):
        table = pd.DataFrame([
            {'index': index, 'task_type': 'large' if index < 3 else 'small',
             'question': 'Choose a shape.', 'answer': 'red square',
             'candidates': repr(['red square', 'blue circle']),
             'prediction': '(A)' if index < 3 else '(B)'}
            for index in range(4)
        ])
        native_file = str(self.root / 'unequal.tsv')
        table.to_csv(native_file, sep='\t', index=False)
        with contextlib.redirect_stdout(io.StringIO()):
            rating = MVBench.evaluate(native_file, trusted_data=table, model='exact_matching')
        summary = local._summarize_rating(rating)
        self.assertEqual(summary['primary_metric']['value'], 0.75)
        self.assertEqual(summary['task_macro_accuracy'], 0.5)
        self.assertEqual(summary['per_task']['large']['total'], 3)
        self.assertTrue(Path(get_intermediate_file_path(native_file, '_score')).exists())

    def test_native_image_prompt_respects_clip_bounds_cache_and_reference_crop(self):
        frames = self._media('Episodic Reasoning')
        dataset = local.build_dataset(self.data, nframe=4, smoke=True)
        line = dict(dataset.data.iloc[0])
        line.update(prefix=str(frames.parent.relative_to(self.data)), video=frames.name,
                    data_type='frame', bound=True, start=0, end=1)
        first = dataset.build_prompt(line, video_llm=True)
        first_images = [part['value'] for part in first if part['type'] == 'image']
        self.assertEqual(len(first_images), 4)
        self.assertFalse(any(part['type'] == 'video' for part in first))
        self.assertIn('Question: Choose the shape', first[-3]['value'])
        self.assertEqual(first[-1]['value'], 'Best option:(')
        for path in first_images:
            with Image.open(path) as image:
                self.assertEqual(image.size, (224, 224))
        line.update(start=1, end=2)
        second = dataset.build_prompt(line, video_llm=True)
        second_images = [part['value'] for part in second if part['type'] == 'image']
        self.assertTrue(set(first_images).isdisjoint(second_images))
        with Image.open(first_images[-1]) as left, Image.open(second_images[-1]) as right:
            self.assertNotEqual(left.getpixel((100, 100)), right.getpixel((100, 100)))
        self.assertEqual(dataset.save_video_into_images(line), second_images)
        key = dataset.clip_cache_key(line)
        dataset.nframe = 8
        self.assertNotEqual(dataset.clip_cache_key(line), key)
        dataset.nframe = 4
        # Mutate a numbered source frame, not the path or clip bounds.
        Image.new('RGB', (320, 240), (0, 255, 0)).save(frames / '00005.jpg')
        self.assertNotEqual(dataset.clip_cache_key(line), key)
        changed_images = dataset.save_video_into_images(line)
        self.assertTrue(set(changed_images).isdisjoint(second_images))
        alternate = frames.parent / 'different-source' / frames.name
        alternate.mkdir(parents=True)
        for frame in frames.glob('*.jpg'):
            (alternate / frame.name).write_bytes(frame.read_bytes())
        line['prefix'] += '/different-source'
        self.assertNotEqual(dataset.clip_cache_key(line), dataset.clip_cache_key({
            **line, 'prefix': str(frames.parent.relative_to(self.data)),
        }))

    def test_extraction_validates_every_member_before_writing(self):
        for bad_name in ('../escape.txt', '/absolute.txt', 'C:/drive.txt', 'folder\\escape.txt'):
            with self.subTest(name=bad_name):
                archive = self.root / 'unsafe.zip'
                with zipfile.ZipFile(archive, 'w') as source:
                    source.writestr('innocent.txt', 'must not be extracted either')
                    source.writestr(bad_name, 'unsafe')
                destination = self.root / 'extract'
                with self.assertRaisesRegex(ValueError, 'Unsafe relative path'):
                    local._safe_extract(archive, destination)
                self.assertFalse((destination / 'innocent.txt').exists())
                self.assertFalse((self.root / 'escape.txt').exists())

    def test_zip_symlinks_and_preexisting_escape_links_are_rejected(self):
        archive = self.root / 'symlink.zip'
        link = zipfile.ZipInfo('link')
        link.create_system = 3
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        with zipfile.ZipFile(archive, 'w') as source:
            source.writestr(link, '../escape')
        with self.assertRaisesRegex(ValueError, 'member type'):
            local._safe_extract(archive, self.root / 'extract')
        destination = self.root / 'extract'
        outside = self.root / 'outside'
        outside.mkdir()
        (destination / 'linked').symlink_to(outside, target_is_directory=True)
        with zipfile.ZipFile(archive, 'w') as source:
            source.writestr('linked/escape.txt', 'unsafe')
        with self.assertRaisesRegex(ValueError, 'escapes dataset root'):
            local._safe_extract(archive, destination)
        self.assertFalse((outside / 'escape.txt').exists())

    def test_june_overlay_overwrites_only_its_unwrapped_media_path(self):
        destination = self.root / 'video'
        target = destination / 'scene_qa/video/fixture.mp4'
        target.parent.mkdir(parents=True)
        target.write_bytes(b'old')
        archive = self.root / 'data0613.zip'
        with zipfile.ZipFile(archive, 'w') as source:
            source.writestr('data0613/scene_qa/video/fixture.mp4', b'corrected')
        local._safe_extract(archive, destination, overlay=True)
        self.assertEqual(target.read_bytes(), b'corrected')
        self.assertFalse((destination / 'data0613').exists())


if __name__ == '__main__':
    unittest.main()
