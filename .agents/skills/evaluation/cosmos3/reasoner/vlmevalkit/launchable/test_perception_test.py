# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Synthetic trust-boundary and real-video tests; no network, judge or GPU."""
import copy
import io
import json
from pathlib import Path
import shutil
import stat
import tempfile
import unittest
from unittest.mock import patch
import warnings
import zipfile

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from vlmeval import perception_test_local as local
from vlmeval.dataset.perception_test import PerceptionTestLocal


class PerceptionTestTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data = self.root / 'data'
        self.data.mkdir()
        self.annotations = {}
        for video, size in (('video_9000', 2), ('video_8241', 5), ('video_2', 2)):
            questions = []
            for i in reversed(range(size)):
                questions.append({'id': i, 'question': f'Which shape is shown in question {i}?',
                                  'options': ['red square', 'blue circle', 'green triangle'],
                                  'answer_id': i % 3, 'area': 'memory' if i % 2 == 0 else 'physics',
                                  'reasoning': 'descriptive',
                                  'tag': ['shape', 'motion'] if i % 2 == 0 else ['shape']})
            self.annotations[video] = {'metadata': {'video_id': video, 'split': 'valid', 'num_frames': 12},
                                       'mc_question': questions}
        self.cuts = {'video_2': -1, 'video_8241': 6, 'video_9000': -1}
        self._json('mc_question_valid.json', self.annotations)
        self._json('cut_frame_mapping_valid.json', self.cuts)
        (self.data / 'README.md').write_text('Synthetic attribution fixture; not actual benchmark data.\n')
        hashes = {name: local._sha256(self.data / name) for name in local.METADATA_SHA256}
        for target, value in (('EXPECTED_ROWS', 9), ('EXPECTED_VIDEOS', 3), ('SMOKE_INDICES', list(range(2, 7)))):
            patcher = patch.object(local, target, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch.dict(local.METADATA_SHA256, hashes, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _json(self, name, value):
        (self.data / name).write_text(json.dumps(value), encoding='utf-8')

    def _repin(self, name, value):
        self._json(name, value)
        local.METADATA_SHA256[name] = local._sha256(self.data / name)

    def _table(self):
        result = local.prepare_dataset(self.data, smoke=True, require_media=False)
        table = pd.read_csv(result['data_file'], sep='\t', keep_default_na=False)
        table['prediction'] = ['(A)', 'B', 'Option C.', 'Answer: A', '(B) blue circle']
        return table

    def _predictions(self, table=None, name='predictions.tsv'):
        path = self.root / name
        (self._table() if table is None else table).to_csv(path, sep='\t', index=False)
        return path

    def _score(self, table=None, name='score'):
        return local.score_predictions(self._predictions(table), self.data, self.root / name, smoke=True)

    def _media(self, video='video_8241'):
        path = self.data / 'videos' / f'{video}.mp4'
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b'Synthetic identity bytes, not a decodable video.')
        return path

    def _real_video(self, red_after=6, size=(960, 480)):
        path = self.data / 'videos' / 'video_8241.mp4'
        path.parent.mkdir(exist_ok=True)
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 6, size)
        if not writer.isOpened():
            self.fail('OpenCV mp4v encoding is required for real frame-boundary tests')
        try:
            for i in range(12):
                # BGR: allowed frames green, forbidden tail red.
                color = (0, 180, 0) if i < red_after else (0, 0, 220)
                frame = np.empty((size[1], size[0], 3), dtype=np.uint8)
                frame[:] = color
                writer.write(frame)
        finally:
            writer.release()
        return path

    def _dataset(self, nframe=4):
        return PerceptionTestLocal(data_root=self.data, smoke=True, nframe=nframe)

    def test_small_real_frames_are_not_upscaled(self):
        self._real_video(size=(320, 160))
        dataset = self._dataset()
        for path in dataset.save_video_frames(dataset.data.iloc[0]):
            with Image.open(path) as image:
                self.assertEqual(image.size, (320, 160))

    def test_choice_adapter_rejects_malformed_or_conflicting_choices(self):
        options = ['red square', 'blue circle', 'green triangle']
        for text in ('(A', 'A)', 'A or B', '(A)(B)', 'Ared square', '???', 'nan'):
            with self.subTest(text=text):
                self.assertIsNone(local.parse_choice(text, options))
        self.assertEqual(local.parse_choice('blue circle', options), 1)
        self.assertEqual(local.parse_choice('Answer: C. It is the triangle.', options), 2)

    def test_numeric_video_and_question_order_preserves_stable_selected_ids(self):
        prepared = local.prepare_dataset(self.data, smoke=True, require_media=False)
        self.assertEqual(prepared['selected_indices'], [2, 3, 4, 5, 6])
        table = pd.read_csv(prepared['data_file'], sep='\t')
        self.assertEqual(table['question_id'].tolist(), [0, 1, 2, 3, 4])
        self.assertEqual(table['video_id'].unique().tolist(), ['video_8241'])
        self.assertEqual(prepared['coverage']['expected_rows'], 9)
        self.assertEqual(prepared['coverage']['selected_rows'], 5)

    def test_full_shape_preserves_fixed_absolute_smoke_identities(self):
        annotations, cuts = {}, {}
        template = self.annotations['video_8241']['mc_question'][0]
        video_ids = list(range(1, 4002)) + [8241] + list(range(8242, 9500))
        for position, video_number in enumerate(video_ids):
            video = f'video_{video_number}'
            if position < 4001:
                size = 4 if position < 1401 else 3
            elif video_number == 8241:
                size = 5
            else:
                size = 5 if position - 4002 < 699 else 4
            questions = [{**template, 'id': i} for i in reversed(range(size))]
            annotations[video] = {'metadata': {'video_id': video, 'split': 'valid', 'num_frames': 12},
                                  'mc_question': questions}
            cuts[video] = -1
        self._repin('mc_question_valid.json', annotations)
        self._repin('cut_frame_mapping_valid.json', cuts)
        with patch.object(local, 'EXPECTED_ROWS', 19140), patch.object(local, 'EXPECTED_VIDEOS', 5260), \
                patch.object(local, 'SMOKE_INDICES', [13404, 13405, 13406, 13407, 13408]):
            prepared = local.prepare_dataset(self.data, smoke=True, require_media=False)
            self.assertEqual(prepared['selected_indices'], [13404, 13405, 13406, 13407, 13408])
            self.assertEqual(prepared['coverage']['expected_rows'], 19140)
            self.assertEqual(prepared['coverage']['expected_videos'], 5260)
            full = local.prepare_dataset(self.data, require_media=False)
            self.assertEqual(full['selected_indices'], list(range(19140)))
            self.assertEqual(full['coverage']['selected_videos'], 5260)

    def test_native_export_preserves_missing_value_like_option_text(self):
        changed = copy.deepcopy(self.annotations)
        question = next(q for q in changed['video_8241']['mc_question'] if q['id'] == 0)
        question['options'] = ['nan', 'NA', 'null']
        self._repin('mc_question_valid.json', changed)
        self._media()
        dataset = self._dataset()
        self.assertEqual(dataset.data.iloc[0][['A', 'B', 'C']].tolist(), ['nan', 'NA', 'null'])

    def test_media_independent_manifest_and_offline_scoring(self):
        offline = local.prepare_dataset(self.data, smoke=True, require_media=False)
        self._media()
        online = local.prepare_dataset(self.data, smoke=True)
        self.assertEqual(offline['manifest_sha256'], online['manifest_sha256'])
        self.assertEqual(offline['annotation_manifest_sha256'], online['annotation_manifest_sha256'])
        self.assertIsNone(offline['media_manifest_sha256'])
        self.assertIn('videos/video_8241.mp4', online['media_sha256'])
        shutil.rmtree(self.data / 'videos')
        summary = self._score()
        self.assertEqual(summary['primary_metric'], {'name': 'top1', 'value': 1., 'unit': 'fraction',
                                                     'correct': 5, 'total': 5})
        self.assertEqual(summary['status'], 'smoke_only')
        self.assertEqual(summary['invalid_count'], 0)
        self.assertEqual(summary['provenance']['media_sha256'], {})

    def test_missing_full_media_never_drops_rows(self):
        self._media()
        with self.assertRaises(local.PerceptionTestCoverageError) as caught:
            local.prepare_dataset(self.data)
        self.assertEqual(caught.exception.report['coverage']['available_rows'], 5)
        self.assertEqual(caught.exception.report['coverage']['selected_rows'], 9)
        self.assertEqual(caught.exception.report['missing_media'], ['videos/video_2.mp4', 'videos/video_9000.mp4'])
        self._media('video_2')
        self._media('video_9000')
        prepared = local.prepare_dataset(self.data)
        self.assertEqual(prepared['selected_indices'], list(range(9)))
        self.assertEqual(prepared['coverage']['selected_rows'], 9)

    def test_prebuilt_tsv_cannot_replace_immutable_labels(self):
        table = self._table()
        prepared = local.prepare_dataset(self.data, smoke=True, require_media=False)
        Path(prepared['data_file']).write_text('index\tanswer_id\n2\t2\n')
        regenerated = local.prepare_dataset(self.data, smoke=True, require_media=False)
        fresh = pd.read_csv(regenerated['data_file'], sep='\t')
        self.assertEqual(fresh['answer_id'].tolist(), [0, 1, 2, 0, 1])
        self.assertEqual(self._score(table)['primary_metric']['correct'], 5)

    def test_annotation_labels_options_and_cuts_are_checksum_pinned(self):
        original = (self.data / 'mc_question_valid.json').read_bytes()
        for field, replacement in (('answer_id', 2), ('options', ['forged', 'b', 'c'])):
            changed = copy.deepcopy(self.annotations)
            changed['video_8241']['mc_question'][0][field] = replacement
            self._json('mc_question_valid.json', changed)
            with self.assertRaisesRegex(ValueError, 'checksum'):
                local.prepare_dataset(self.data, smoke=True, require_media=False)
            (self.data / 'mc_question_valid.json').write_bytes(original)
        self._json('cut_frame_mapping_valid.json', {**self.cuts, 'video_8241': -1})
        with self.assertRaisesRegex(ValueError, 'checksum'):
            local.prepare_dataset(self.data, smoke=True, require_media=False)

    def test_invalid_and_missing_trusted_cut_bounds_are_rejected(self):
        for cut in (0, -2, 13, True, 1.5, None):
            with self.subTest(cut=cut):
                self._repin('cut_frame_mapping_valid.json', {**self.cuts, 'video_8241': cut})
                with self.assertRaisesRegex(ValueError, 'cut'):
                    local.prepare_dataset(self.data, smoke=True, require_media=False)
        self._repin('cut_frame_mapping_valid.json', {'video_2': -1, 'video_9000': -1})
        with self.assertRaisesRegex(ValueError, 'cut'):
            local.prepare_dataset(self.data, smoke=True, require_media=False)

    def test_duplicate_missing_extra_and_modified_prediction_identity_rejected(self):
        table = self._table()
        cases = [table.iloc[:-1].copy(), pd.concat([table, table.iloc[:1]], ignore_index=True)]
        extra = table.copy()
        extra.loc[0, 'index'] = 999
        cases.append(extra)
        modified = table.copy()
        modified.loc[0, 'video_id'] = 'video_2'
        cases.append(modified)
        modified = table.copy()
        modified.loc[0, 'question_id'] = 99
        cases.append(modified)
        modified = table.copy()
        modified['index'] = modified['index'].astype(float)
        modified.loc[0, 'index'] = 2.5
        cases.append(modified)
        for i, case in enumerate(cases):
            with self.subTest(case=i), self.assertRaises(ValueError):
                self._score(case, name=f'bad-{i}')
            self.assertFalse((self.root / f'bad-{i}').exists())

    def test_prediction_ground_truth_options_question_and_cut_tampering_rejected(self):
        for column, replacement in (('answer', 'C'), ('answer_id', 2), ('A', 'forged'),
                                    ('options', '["forged", "blue circle", "green triangle"]'),
                                    ('question', 'forged question'), ('cut_frame', -1),
                                    ('num_frames', 99), ('tag', '["forged"]')):
            table = self._table()
            table.loc[0, column] = replacement
            with self.subTest(column=column), self.assertRaisesRegex(ValueError, 'Untrusted'):
                self._score(table, name=f'tampered-{column}')

    def test_minimal_shuffled_predictions_use_trusted_labels(self):
        table = self._table()[['index', 'prediction']].iloc[::-1]
        self.assertEqual(self._score(table)['primary_metric']['correct'], 5)

    def test_malformed_ambiguous_literal_nan_are_wrong_not_dropped(self):
        table = self._table()
        table['prediction'] = ['A or B', 'nan', 'unparseable model response', '(A)', '(C)']
        summary = self._score(table)
        self.assertEqual(summary['primary_metric']['correct'], 1)
        self.assertEqual(summary['primary_metric']['total'], 5)
        self.assertEqual(summary['invalid_count'], 3)
        scored = pd.read_csv(self.root / 'score/scored_predictions.tsv', sep='\t', keep_default_na=False)
        self.assertEqual(scored['prediction'][1], 'nan')
        self.assertEqual(scored['score'].tolist(), [0, 0, 0, 1, 0])

    def test_blank_missing_and_failed_outputs_reject_incomplete_runs(self):
        for i, failure in enumerate(('', '   ', None, 'Failed to obtain answer via API.',
                                     'VIDEO_LOAD_FAILED: truncated', 'transport_error: reset', 'API_ERROR')):
            table = self._table()
            table.loc[0, 'prediction'] = failure
            with self.subTest(failure=failure), self.assertRaisesRegex(ValueError, 'Missing or failed'):
                self._score(table, name=f'failure-{i}')

    def test_exact_counts_and_overlapping_tag_denominators(self):
        table = self._table()
        table['prediction'] = ['A', 'C', 'C', 'C', 'B']
        summary = self._score(table)
        self.assertEqual(summary['primary_metric']['correct'], 3)
        self.assertEqual(summary['primary_metric']['value'], .6)
        self.assertEqual(summary['per_area']['memory'], {'correct': 3, 'total': 3, 'accuracy': 1.})
        self.assertEqual(summary['per_area']['physics'], {'correct': 0, 'total': 2, 'accuracy': 0.})
        self.assertEqual(summary['per_reasoning']['descriptive']['total'], 5)
        self.assertEqual(summary['per_tag']['shape']['total'], 5)
        self.assertEqual(summary['per_tag']['motion']['total'], 3)
        self.assertEqual(sum(g['total'] for g in summary['per_tag'].values()), 8)

    def test_duplicate_option_text_preserves_official_option_id(self):
        changed = copy.deepcopy(self.annotations)
        question = next(q for q in changed['video_8241']['mc_question'] if q['id'] == 0)
        question['options'] = ['same text', 'same text', 'other text']
        question['answer_id'] = 1
        self._repin('mc_question_valid.json', changed)
        table = self._table()
        table.loc[0, 'prediction'] = 'B'
        self.assertEqual(self._score(table)['primary_metric']['correct'], 5)
        table.loc[0, 'prediction'] = 'same text'
        summary = self._score(table, name='ambiguous-text')
        self.assertEqual(summary['invalid_count'], 1)
        self.assertEqual(summary['primary_metric']['correct'], 4)

    def test_prediction_files_never_execute_pickle_or_expressions(self):
        payload = self.root / 'payload.pkl'
        payload.write_bytes(b'not a pickle')
        with self.assertRaisesRegex(ValueError, 'pickle'):
            local.score_predictions(payload, self.data, self.root / 'unsafe', smoke=True)
        table = self._table()
        table.loc[0, 'options'] = '__import__("os").system("false")'
        with self.assertRaisesRegex(ValueError, 'Untrusted options'):
            self._score(table)

    def test_fresh_output_required_and_stale_neighbor_scores_ignored(self):
        table = self._table()
        predictions = self._predictions(table)
        (self.root / 'predictions_score.tsv').write_text('forged stale score\n')
        summary = local.score_predictions(predictions, self.data, self.root / 'fresh', smoke=True)
        self.assertEqual(summary['primary_metric']['correct'], 5)
        before = (self.root / 'fresh/summary.json').read_bytes()
        with self.assertRaisesRegex(ValueError, 'fresh'):
            local.score_predictions(predictions, self.data, self.root / 'fresh', smoke=True)
        self.assertEqual((self.root / 'fresh/summary.json').read_bytes(), before)

    def test_real_sampled_frames_enforce_exclusive_cut_and_aspect_resize(self):
        self._real_video()
        dataset = self._dataset()
        prompt = dataset.build_prompt(0, video_llm=True)
        images = [m['value'] for m in prompt if m['type'] == 'image']
        self.assertEqual(len(images), 4)
        self.assertFalse(any(m['type'] == 'video' for m in prompt))
        for path in images:
            with Image.open(path) as image:
                self.assertEqual(image.size, (768, 384))
                red, green, blue = image.getpixel((300, 200))
                self.assertGreater(green, red + 100)
        line = dataset.data.iloc[0].copy()
        line['cut_frame'] = -1
        full_paths = dataset.save_video_frames(line)
        with Image.open(full_paths[-1]) as image:
            red, green, blue = image.getpixel((300, 200))
            self.assertGreater(red, green + 100)
        self.assertNotEqual(images, full_paths)
        self.assertEqual(PerceptionTestLocal.sample_indices(12, 6, 4), [0, 2, 3, 5])
        self.assertEqual(PerceptionTestLocal.sample_indices(1, -1, 4), [0, 0, 0, 0])

    def test_real_frame_cache_changes_for_content_cut_nframe_and_protocol(self):
        path = self._real_video()
        dataset = self._dataset()
        row = dataset.data.iloc[0].copy()
        original = dataset.save_video_frames(row)
        self.assertEqual(dataset.save_video_frames(row), original)
        key = dataset.clip_cache_key(row)
        changed = row.copy()
        changed['cut_frame'] = 5
        self.assertNotEqual(key, dataset.clip_cache_key(changed))
        dataset.nframe = 3
        self.assertNotEqual(key, dataset.clip_cache_key(row))
        dataset.nframe = 4
        with patch('vlmeval.dataset.perception_test.PROTOCOL', 'synthetic-next-protocol'):
            self.assertNotEqual(key, dataset.clip_cache_key(row))
        self._real_video(red_after=0)
        self.assertNotEqual(key, dataset.clip_cache_key(row))
        new_paths = dataset.save_video_frames(row)
        self.assertNotEqual(original, new_paths)
        with Image.open(new_paths[0]) as image:
            red, green, blue = image.getpixel((300, 200))
            self.assertGreater(red, green + 100)
        self.assertTrue(path.is_file())

    def test_invalid_decoded_bounds_fail_instead_of_clamping(self):
        for cut in (0, -2, 13, True, 1.5):
            with self.subTest(cut=cut), self.assertRaises(ValueError):
                PerceptionTestLocal.sample_indices(12, cut, 4)
        self._real_video()
        dataset = self._dataset()
        row = dataset.data.iloc[0].copy()
        row['cut_frame'] = 13
        with self.assertRaisesRegex(ValueError, 'cut'):
            dataset.save_video_frames(row)

    def test_prompt_never_uses_answer_metadata(self):
        self._real_video()
        dataset = self._dataset()
        row = dataset.data.iloc[0].copy()
        row['answer'] = 'SECRET_LABEL_SENTINEL'
        row['answer_id'] = 'SECRET_LABEL_SENTINEL'
        prompt = dataset.build_prompt(row)
        text = '\n'.join(m['value'] for m in prompt if m['type'] == 'text')
        self.assertNotIn('SECRET_LABEL_SENTINEL', text)
        self.assertIn('(A) red square', text)
        self.assertIn('(B) blue circle', text)
        self.assertIn('(C) green triangle', text)

    def test_safe_extraction_rejects_traversal_absolute_symlink_before_writing(self):
        for i, name in enumerate(('../escape', '/absolute', 'videos/../../escape', 'C:\\escape')):
            archive = self.root / f'bad-{i}.zip'
            with zipfile.ZipFile(archive, 'w') as stream:
                stream.writestr('valid.txt', 'must not be extracted')
                stream.writestr(name, 'bad')
            destination = self.root / f'extract-{i}'
            with self.assertRaises(ValueError):
                local._safe_extract(archive, destination)
            self.assertFalse((destination / 'valid.txt').exists())
        archive = self.root / 'symlink.zip'
        with zipfile.ZipFile(archive, 'w') as stream:
            info = zipfile.ZipInfo('videos/link')
            info.create_system = 3
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            stream.writestr(info, '../../outside')
        with self.assertRaisesRegex(ValueError, 'member type'):
            local._safe_extract(archive, self.root / 'symlink-out')

    def test_existing_symlink_parent_and_duplicate_members_rejected(self):
        outside = self.root / 'outside'
        outside.mkdir()
        destination = self.root / 'extract'
        destination.mkdir()
        (destination / 'videos').symlink_to(outside, target_is_directory=True)
        archive = self.root / 'parent-link.zip'
        with zipfile.ZipFile(archive, 'w') as stream:
            stream.writestr('videos/video_8241.mp4', 'bad')
        with self.assertRaisesRegex(ValueError, 'symlink'):
            local._safe_extract(archive, destination)
        self.assertEqual(list(outside.iterdir()), [])
        archive = self.root / 'duplicate.zip'
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            with zipfile.ZipFile(archive, 'w') as stream:
                stream.writestr('same.txt', 'first')
                stream.writestr('same.txt', 'second')
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            local._safe_extract(archive, self.root / 'duplicate-out')
        self.assertFalse((self.root / 'duplicate-out/same.txt').exists())

    def test_download_verifies_generation_bytes_and_sha_before_install(self):
        content = b'synthetic public pinned bytes'
        digest = __import__('hashlib').sha256(content).hexdigest()
        pin = {'object': 'synthetic', 'generation': '42', 'bytes': len(content), 'sha256': digest}
        class Response(io.BytesIO):
            headers = {'x-goog-generation': '42'}
        path = self.root / 'download'
        with patch.object(local.urllib.request, 'urlopen', return_value=Response(content)):
            local._download(path, pin)
        self.assertEqual(path.read_bytes(), content)
        with self.assertRaisesRegex(ValueError, 'checksum'):
            local._download(path, {**pin, 'sha256': '0' * 64})
        path.unlink()
        with patch.object(local.urllib.request, 'urlopen', return_value=Response(content + b'corrupt')):
            with self.assertRaisesRegex(ValueError, 'verification'):
                local._download(path, pin)
        self.assertFalse(path.exists())
        with patch.object(local.urllib.request, 'urlopen', return_value=Response(content)):
            with self.assertRaisesRegex(ValueError, 'generation'):
                local._download(path, {**pin, 'generation': '43'})
        self.assertFalse(path.exists())


if __name__ == '__main__':
    unittest.main()
