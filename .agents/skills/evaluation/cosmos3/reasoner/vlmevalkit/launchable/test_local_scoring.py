# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Synthetic, nonsecret local bundles exercising real native scoring behavior.

Run from the installed VLMEvalKit environment. No GPU, network, credentials,
model endpoint, or fabricated DVC semantic scorer is used.
"""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd
from PIL import Image

from vlmeval.local_scoring import (
    _validate_predictions, build_local_dataset, dataset_config, load_assets, score_predictions,
)
from vlmeval.scoring_assets import EVENT_FILES, SOURCE_DIRS, VQA_FILES, prepare_assets


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _image(path, size=(100, 100)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new('RGB', size, color='white').save(path)


def _video(path):
    # Only file identity is used by these scoring-only fixtures. No decoder or
    # inference is involved; duration is explicitly supplied by annotations.
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'synthetic scoring fixture: media is not decoded')


def _prepare(root, tasks):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return prepare_assets(root, 'synthetic-nonsecret-v1', tasks=tasks)


def _grounding(root, count=2):
    source = root / SOURCE_DIRS['2DGrounding']
    _image(source / 'images/example.png')
    rows = []
    for index in range(count):
        rows.append({
            'media': 'images/example.png',
            'conversations': [{'from': 'human', 'value': 'Locate the square.'},
                              {'from': 'gpt', 'value': '[[10, 10, 50, 50]]'}],
            '_meta': {'sentence': f'square {index}', 'image_width': 100, 'image_height': 100},
        })
    _json(source / 'refdrone_test_llava.json', rows)


def _sot(root):
    source = root / SOURCE_DIRS['SOT']
    for sequence in ('first', 'second'):
        directory = source / sequence
        _json(directory / 'gt.json', {
            'frame_ids': [0, 10, 20], 'label': f'camera/{sequence}',
            'init_bbox': [100, 100, 500, 500],
            'gt_bboxes': {'0': [100, 100, 500, 500], '10': [100, 100, 500, 500], '20': None},
        })
        for name in ('f00.png', 'f01.png', 'f02.png', 'crop.png'):
            _image(directory / 'frames' / name)


def _astro(root):
    from vlmeval.dataset.metropolis2d.astro_2d_dataset import Astro2DDetectionDataset
    for child, layout in Astro2DDetectionDataset._LAYOUT.items():
        source = root / SOURCE_DIRS['Astro2D'] / layout['data_root'].rsplit('/', 1)[1]
        # Unequal child sizes make micro-vs-sequence-macro mistakes observable.
        count = 2 if child.endswith('_C0065') else 1
        for index in range(count):
            _image(source / 'images' / f'{index}.png')
            labels = source / 'labels' / f'{index}.txt'
            labels.parent.mkdir(parents=True, exist_ok=True)
            labels.write_text('Pedestrian 0 0 0 10 10 50 50 0 0 0 0 0 0 0\n')


def _events(root):
    source = root / SOURCE_DIRS['EventVerification']
    for index, relative in enumerate(EVENT_FILES):
        annotation = source / relative
        _video(annotation.parent / 'clip.mp4')
        _json(annotation, {'bcq': [{'id': f'item-{index}', 'video': 'clip.mp4',
                                 'question': 'Is the square visible?', 'answer': 'yes',
                                 'system_prompt': 'Answer yes or no.'}]})


def _vqa(root):
    source = root / SOURCE_DIRS['VQA']
    files = [name for name in VQA_FILES if not name.startswith('VANTAGE_')]
    for index, name in enumerate(files):
        video = f'clip-{index}'
        _video(source / 'videos' / (video + '.mp4'))
        _json(source / 'data_jsons/annotations' / name, [{
            'q_uid': video, 'question_id': f'q-{index}', 'question': 'Choose the square.',
            'options': ['square', 'circle', 'triangle', 'star'], 'gt_option': 'A',
        }])


def _temporal(root):
    source = root / SOURCE_DIRS['Temporal']
    rows = []
    for video, question, answer in (
            ('first', 'q1', '<1><3>'), ('first', 'q2', '<1><3>'), ('second', 'q3', '<1><3>')):
        _video(source / 'videos' / (video + '.mp4'))
        rows.append({'vid': video, 'question_id': question, 'question': question,
                     'answer': answer, 'duration': 10})
    _json(source / 'data_jsons/annotations/examples.json', rows)


def _pointing(root):
    source = root / SOURCE_DIRS['2DPointing']
    _image(source / 'images/example.png')
    pd.DataFrame([
        {'index': index, 'question_id': f'point-{index}', 'question': 'Choose the marked point.',
         'answer': answer, 'target_point': '[20,30]', 'image_path': 'images/example.png',
         'A': 'upper left', 'B': 'upper right', 'C': 'lower left', 'D': 'lower right'}
        for index, answer in ((10, 'A'), (11, 'B'))
    ]).to_csv(source / 'Metropolis2DPointing.tsv', sep='\t', index=False)


class LocalScoringTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.assets = self.root / 'assets'
        self.assets.mkdir()

    def predictions(self, task, values, name='predictions.tsv'):
        table = build_local_dataset(self.assets, task).data.copy()
        table['prediction'] = values
        path = self.root / name
        table.to_csv(path, sep='\t', index=False)
        return path

    def score(self, task, path, name='result', **kwargs):
        return score_predictions(task, path, self.assets, self.root / name, **kwargs)

    def test_grounding_ignores_injected_gt_paths_and_reports_native_percent(self):
        _grounding(self.assets)
        _prepare(self.assets, ['2DGrounding'])
        path = self.predictions('2DGrounding', ['[[100,100,500,500]]'] * 2)
        table = pd.read_csv(path, sep='\t')
        table['gt_bboxes'] = '[[70,70,90,90]]'
        table['image_path'] = '/untrusted/does-not-exist.png'
        table['image_width'] = 98765
        table['image_height'] = 98765
        table.to_csv(path, sep='\t', index=False)
        diagnostic = io.StringIO()
        with contextlib.redirect_stdout(diagnostic), contextlib.redirect_stderr(diagnostic):
            summary = self.score('2DGrounding', path)
        self.assertEqual(diagnostic.getvalue(), '')
        self.assertEqual(summary['primary_metric']['value'], 100.0)
        self.assertEqual(summary['primary_metric']['unit'], 'percent')
        self.assertEqual(summary['native_metrics']['Mean_IoU'], 100.0)
        scored = pd.read_csv(self.root / 'result/predictions_VANTAGE_2DGrounding.tsv', sep='\t')
        self.assertTrue(scored['image_path'].map(lambda value: Path(value).is_file()).all())
        self.assertEqual(summary['provenance']['predictions_sha256'], hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(json.loads((self.root / 'result/summary.json').read_text()), summary)

    def test_partial_requires_explicit_subset_and_never_claims_full_coverage(self):
        _grounding(self.assets)
        _prepare(self.assets, ['2DGrounding'])
        path = self.predictions('2DGrounding', ['[[100,100,500,500]]'] * 2)
        table = pd.read_csv(path, sep='\t').iloc[:1]
        table.to_csv(path, sep='\t', index=False)
        with self.assertRaisesRegex(ValueError, 'Full local dataset coverage'):
            self.score('2DGrounding', path)
        self.assertFalse((self.root / 'result').exists())
        summary = self.score('2DGrounding', path, allow_subset=True)
        self.assertEqual(summary['coverage'], {'expected_rows': 2, 'scored_rows': 1, 'subset': True})

    def test_duplicate_unknown_and_mismatched_identity_fail_before_output(self):
        _grounding(self.assets)
        _prepare(self.assets, ['2DGrounding'])
        path = self.predictions('2DGrounding', ['[[100,100,500,500]]'] * 2)
        original = pd.read_csv(path, sep='\t')
        for kind in ('duplicate', 'unknown', 'mismatch', 'missing_identity'):
            with self.subTest(kind=kind):
                table = original.copy()
                if kind == 'duplicate':
                    table.loc[1, 'index'] = table.loc[0, 'index']
                elif kind == 'unknown':
                    table.loc[1, 'index'] = 999
                elif kind == 'mismatch':
                    table.loc[1, 'sentence'] = 'another example'
                else:
                    table = table.drop(columns=['sentence'])
                table.to_csv(path, sep='\t', index=False)
                with self.assertRaises(ValueError):
                    self.score('2DGrounding', path)
                self.assertFalse((self.root / 'result').exists())

    def test_native_sot_uses_integer_cache_and_absent_frame_semantics(self):
        _sot(self.assets)
        _prepare(self.assets, ['SOT'])
        predictions = [json.dumps({'frame_1': [100, 100, 500, 500], 'frame_2': None}),
                       json.dumps({'frame_1': None, 'frame_2': None})]
        path = self.predictions('SOT', predictions)
        summary = self.score('SOT', path)
        # Native threshold grid includes zero: a missed visible frame has AUC
        # 1/21, not exactly zero. The absent frame is excluded, not rewarded.
        self.assertAlmostEqual(summary['native_metrics']['camera/first'], 1.0)
        self.assertAlmostEqual(summary['native_metrics']['camera/second'], 1 / 21)
        self.assertAlmostEqual(summary['primary_metric']['value'], (1 + 1 / 21) / 2)
        self.assertEqual(summary['primary_metric']['unit'], 'fraction')

    def test_sot_missing_gt_frame_is_rejected_even_with_updated_fingerprint(self):
        _sot(self.assets)
        _prepare(self.assets, ['SOT'])
        path = self.assets / SOURCE_DIRS['SOT'] / 'first/gt.json'
        annotation = json.loads(path.read_text())
        del annotation['gt_bboxes']['20']
        path.write_text(json.dumps(annotation))
        manifest_path = self.assets / 'scoring-assets.json'
        manifest = json.loads(manifest_path.read_text())
        manifest['files'][path.relative_to(self.assets).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaises(ValueError):
            load_assets(self.assets, ['SOT'])

    def test_astro_distinct_roots_and_unweighted_native_sequence_macro(self):
        _astro(self.assets)
        _prepare(self.assets, ['Astro2D'])
        config = dataset_config(self.assets, 'Astro2D')
        self.assertEqual(len(set(config['data_roots'].values())), 3)
        dataset = build_local_dataset(self.assets, 'Astro2D')
        values = ['[{"bbox_2d":[100,100,500,500],"label":"person"}]'
                  if row['SUB_DATASET'].endswith('_C0065') else '[]'
                  for _, row in dataset.data.iterrows()]
        path = self.predictions('Astro2D', values)
        # Including VANTAGE_Astro2D in a parent directory must not corrupt the
        # native child file paths (ConcatDataset's historical string trap).
        summary = self.score('Astro2D', path, name='VANTAGE_Astro2D-output')
        self.assertAlmostEqual(summary['primary_metric']['value'], 100 / 3)
        self.assertEqual(summary['primary_metric']['aggregation'], 'unweighted_sequence_macro')
        self.assertEqual(summary['native_metrics']['VANTAGE_Astro2D_C0065:f1'], 100)
        self.assertEqual(summary['native_metrics']['VANTAGE_Astro2D_IVAKPI05:f1'], 0)

    def test_astro_inverted_source_boxes_remain_in_native_denominator(self):
        from vlmeval.dataset.metropolis2d.astro_2d_dataset import Astro2DDetectionDataset
        _astro(self.assets)
        layout = Astro2DDetectionDataset._LAYOUT['VANTAGE_Astro2D_C0065']
        label = self.assets / SOURCE_DIRS['Astro2D'] / layout['data_root'].rsplit('/', 1)[1] / 'labels/0.txt'
        label.write_text(label.read_text() + 'Person 0 0 0 50 10 10 50 0 0 0 0 0 0 0\n')
        _prepare(self.assets, ['Astro2D'])
        count = len(build_local_dataset(self.assets, 'Astro2D').data)
        path = self.predictions('Astro2D', ['[{"bbox_2d":[100,100,500,500],"label":"person"}]'] * count)
        summary = self.score('Astro2D', path)
        self.assertEqual(summary['native_metrics']['VANTAGE_Astro2D_C0065:total_gt_objects'], 3)
        self.assertEqual(summary['native_metrics']['VANTAGE_Astro2D_C0065:false_negatives'], 1)
        self.assertAlmostEqual(summary['native_metrics']['VANTAGE_Astro2D_C0065:f1'], 80)
        self.assertAlmostEqual(summary['primary_metric']['value'], (80 + 100 + 100) / 3)

    def test_event_invalid_predictions_are_not_silently_removed(self):
        _events(self.assets)
        _prepare(self.assets, ['EventVerification'])
        path = self.predictions('EventVerification', ['yes', 'yes', 'yes', 'unparseable'])
        with self.assertRaisesRegex(ValueError, 'silent fallback'):
            self.score('EventVerification', path)
        self.assertFalse((self.root / 'result').exists())
        path = self.predictions('EventVerification', ['yes'] * 4)
        summary = self.score('EventVerification', path)
        self.assertEqual(summary['native_metrics']['Valid Predictions'], 4)
        self.assertEqual(summary['primary_metric']['value'], 1)

    def test_vqa_uses_local_ground_truth_not_prediction_answer(self):
        _vqa(self.assets)
        _prepare(self.assets, ['VQA'])
        dataset = build_local_dataset(self.assets, 'VQA')
        path = self.predictions('VQA', ['B'] * len(dataset.data))
        table = pd.read_csv(path, sep='\t')
        table['answer'] = 'B'
        table.to_csv(path, sep='\t', index=False)
        summary = self.score('VQA', path)
        self.assertEqual(summary['primary_metric']['value'], 0)

    def test_vqa_binary_explicit_label_is_normalized_without_defaulting(self):
        _vqa(self.assets)
        source = self.assets / SOURCE_DIRS['VQA'] / 'data_jsons/annotations' / VQA_FILES[0]
        records = json.loads(source.read_text())
        records[0].update(options=['square', 'circle'], gt_option=' b ')
        _json(source, records)
        _prepare(self.assets, ['VQA'])
        count = len(build_local_dataset(self.assets, 'VQA').data)
        summary = self.score('VQA', self.predictions('VQA', ['A'] * count))
        self.assertEqual(summary['primary_metric']['value'], (count - 1) / count)

    def test_vqa_reused_fallback_qid_cannot_hide_wrong_question(self):
        _vqa(self.assets)
        source = self.assets / SOURCE_DIRS['VQA']
        _video(source / 'videos/shared.mp4')
        _json(source / 'data_jsons/annotations' / VQA_FILES[0], [
            {'q_uid': 'shared.mp4', 'question': question,
             'options': ['square', 'circle', 'triangle', 'star'], 'gt_option': answer}
            for question, answer in [('Choose the square.', 'A'), ('Choose the circle.', 'B')]
        ])
        _prepare(self.assets, ['VQA'])
        count = len(build_local_dataset(self.assets, 'VQA').data)
        path = self.predictions('VQA', ['A'] * count)
        table = pd.read_csv(path, sep='\t')
        shared = table.index[table['video'] == 'shared.mp4']
        self.assertEqual(table.loc[shared, 'qid'].nunique(), 1)
        table.loc[shared, 'question'] = table.loc[shared, 'question'].tolist()[::-1]
        table.to_csv(path, sep='\t', index=False)
        with self.assertRaisesRegex(ValueError, 'stable identity'):
            self.score('VQA', path)
        self.assertFalse((self.root / 'result').exists())

    def test_temporal_preserves_video_macro_not_question_micro(self):
        _temporal(self.assets)
        _prepare(self.assets, ['Temporal'])
        dataset = build_local_dataset(self.assets, 'Temporal')
        values = [json.dumps({'start': '1', 'end': '3'}) if row['video'] == 'first'
                  else json.dumps({'start': '5', 'end': '7'}) for _, row in dataset.data.iterrows()]
        path = self.predictions('Temporal', values)
        summary = self.score('Temporal', path)
        self.assertEqual(summary['primary_metric']['value'], 0.5)
        self.assertEqual(summary['native_metrics']['overall']['count'], 3)

    def test_temporal_malformed_prediction_cannot_receive_full_duration_default(self):
        _temporal(self.assets)
        _prepare(self.assets, ['Temporal'])
        path = self.predictions('Temporal', ['no parseable interval'] * 3)
        with self.assertRaisesRegex(ValueError, 'silent fallback'):
            self.score('Temporal', path)
        self.assertFalse((self.root / 'result').exists())

    def test_pointing_uses_native_accuracy_and_rejects_textual_id_alias(self):
        _pointing(self.assets)
        _prepare(self.assets, ['2DPointing'])
        path = self.predictions('2DPointing', ['A', 'D'])
        table = pd.read_csv(path, sep='\t')
        table['answer'] = ['A', 'D']
        table.to_csv(path, sep='\t', index=False)
        summary = self.score('2DPointing', path)
        self.assertEqual(summary['primary_metric']['value'], 0.5)
        self.assertEqual(summary['primary_metric']['unit'], 'fraction')
        table['index'] = ['010', '11']
        table.to_csv(path, sep='\t', index=False)
        with self.assertRaisesRegex(ValueError, 'identity does not match'):
            self.score('2DPointing', path, name='alias')

    def test_asset_hash_and_escape_and_existing_output_are_rejected(self):
        _grounding(self.assets)
        _prepare(self.assets, ['2DGrounding'])
        path = self.predictions('2DGrounding', ['[[100,100,500,500]]'] * 2)
        (self.root / 'existing').mkdir()
        with self.assertRaisesRegex(ValueError, 'fresh output'):
            self.score('2DGrounding', path, name='existing')
        manifest_path = self.assets / 'scoring-assets.json'
        original = manifest_path.read_text()
        manifest = json.loads(original)
        annotation = self.assets / SOURCE_DIRS['2DGrounding'] / 'refdrone_test_llava.json'
        annotation.write_text(annotation.read_text() + '\n')
        with self.assertRaisesRegex(ValueError, 'fingerprint mismatch'):
            load_assets(self.assets, ['2DGrounding'])
        manifest['tasks']['2DGrounding']['dataset_kwargs']['data_root'] = '../'
        manifest['files'][annotation.relative_to(self.assets).as_posix()] = hashlib.sha256(annotation.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'escapes'):
            load_assets(self.assets, ['2DGrounding'])

    def test_temporal_zero_interval_reduces_score_without_dropping_row(self):
        _temporal(self.assets)
        _prepare(self.assets, ['Temporal'])
        dataset = build_local_dataset(self.assets, 'Temporal')
        values = [
            json.dumps({'start': '0', 'end': '0'} if position == 0 else {'start': '1', 'end': '3'})
            for position, (_, row) in enumerate(dataset.data.iterrows())
        ]
        summary = self.score('Temporal', self.predictions('Temporal', values))
        self.assertEqual(summary['primary_metric']['value'], 0.75)
        self.assertEqual(summary['native_metrics']['overall']['count'], 3)
        self.assertEqual(summary['coverage'], {'expected_rows': 3, 'scored_rows': 3, 'subset': False})

    def test_dvc_degenerate_intervals_retain_native_zero_iou_without_fallback(self):
        from vlmeval.dataset.metropolis_dvc import MetropolisDVC, _iou

        rows = pd.DataFrame({'prediction': [
            json.dumps([{'start': start, 'end': end, 'caption': 'A square moves.'}])
            for start, end in [('0', '0'), ('3', '1'), ('1', '3')]
        ]})
        _validate_predictions('DVC', MetropolisDVC, rows)
        overlaps = []
        for text in rows['prediction']:
            event, = MetropolisDVC.parse_events_from_json(text)
            interval = [MetropolisDVC.parse_timestamp(event[key]) for key in ('start', 'end')]
            overlaps.append(_iou(interval, [1, 3]))
        self.assertEqual(overlaps, [0.0, 0.0, 1.0])
        self.assertEqual(sum(overlaps) / len(overlaps), 1 / 3)
        for event in (
            {'start': 'nan', 'end': '3', 'caption': 'A square moves.'},
            {'start': '1', 'caption': 'A square moves.'},
            {'start': '1', 'end': '3', 'caption': ''},
        ):
            with self.subTest(event=event), self.assertRaisesRegex(ValueError, 'silent fallback'):
                _validate_predictions('DVC', MetropolisDVC,
                                      pd.DataFrame({'prediction': [json.dumps([event])]}))

    def test_dvc_without_genuine_local_model_is_rejected(self):
        from vlmeval.scoring_assets import validate_bert_score_model
        empty_model = self.assets / 'model'
        empty_model.mkdir()
        with self.assertRaises((ValueError, FileNotFoundError)):
            validate_bert_score_model(empty_model)
        # Preparation must not create a manifest or silently use a remote or
        # dummy semantic scorer when a local checkpoint was not provisioned.
        with self.assertRaises(ValueError):
            _prepare(self.assets, ['DVC'])
        self.assertFalse((self.assets / 'scoring-assets.json').exists())


if __name__ == '__main__':
    unittest.main()
