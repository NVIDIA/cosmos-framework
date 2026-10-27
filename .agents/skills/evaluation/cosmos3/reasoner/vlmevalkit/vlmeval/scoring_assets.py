# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare authorized, private VANTAGE sources in place without network access.

Run ``python -m vlmeval.scoring_assets --assets-dir ROOT --asset-id REV``.
DVC additionally requires --bert-score-model RELDIR --bert-score-revision REV.
Only canonical tables and scoring-assets.json are written; source labels and media
are never changed or copied. This does not grant redistribution permission.
"""

import argparse
import ast
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path

import pandas as pd

TASKS = ('VQA', 'EventVerification', 'Temporal', 'DVC', '2DGrounding', '2DPointing', 'SOT', 'Astro2D')
VIDEO_TASKS = TASKS[:4]
SOURCE_DIRS = {
    'VQA': 'VQA',
    'EventVerification': 'event_verification_subset/filtered',
    'Temporal': 'Temporal',
    'DVC': 'Dense Video Caption',
    '2DGrounding': 'Spatial/2d_referring_expressions/refdrone',
    '2DPointing': 'datasets/Metropolis2DPointing',
    'SOT': 'tracking/MetropolisSOT_benchmark_8f',
    'Astro2D': '2dbbox',
}
VQA_FILES = (
    'metrics_spatial_wo_ss.json', 'Metropolis_VQA_Verification_Final_ITS_Data.json',
    'VANTAGE_VQA_Verification_Final_ITS_Data.json', 'metrics_spatial_ss.json',
    'metrics_temporal_filtered_ss.json', 'metrics_temporal_wo_ss.json',
)
EVENT_FILES = (
    'metropolis_event_verification/test_annotation.json',
    'tailgating/tailgating_building_r/test_annotation.json',
    'tailgating/tailgating_courtyard/test_annotation.json',
    'warehouse_near_miss/test_annotations.json',
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _vqa_option(value):
    option = value.strip().upper() if isinstance(value, str) else ''
    _require(option in tuple('ABCD'), 'VQA source is missing a valid ground-truth option')
    return option


def _local_path(root, path, directory=False):
    root = Path(root).resolve(strict=True)
    path = Path(path)
    resolved = (root / path).resolve(strict=True) if not path.is_absolute() else path.resolve(strict=True)
    _require(resolved.is_relative_to(root), 'Asset path escapes the local bundle')
    _require(resolved.is_dir() if directory else resolved.is_file(), 'Required local asset has the wrong file type')
    return resolved


def _json(path):
    with Path(path).open(encoding='utf-8') as stream:
        return json.load(stream)


def _records(path, key=None):
    raw = _json(path)
    if key and isinstance(raw, dict):
        raw = raw.get(key)
    _require(isinstance(raw, list) and len(raw) > 0, 'Required annotation must contain a nonempty record list')
    _require(all(isinstance(row, dict) for row in raw), 'Annotation contains a non-object record')
    return raw


def _number(value):
    _require(not isinstance(value, bool), 'Boolean is not a valid numeric label')
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError('Missing or malformed numeric ground-truth label') from exc
    _require(math.isfinite(result), 'Label contains a nonfinite number')
    return result


def _timestamp(value):
    _require(value is not None and str(value).strip() != '', 'Missing timestamp label')
    parts = str(value).strip().split(':')
    _require(1 <= len(parts) <= 3, 'Malformed timestamp label')
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60 + _number(part)
    _require(seconds >= 0, 'Negative timestamp label')
    return seconds


def _span(start, end):
    start, end = _timestamp(start), _timestamp(end)
    _require(end > start, 'Label interval must have positive duration')
    return start, end


def _bbox(box):
    _require(isinstance(box, (list, tuple)) and len(box) == 4, 'Missing or malformed bounding-box label')
    x1, y1, x2, y2 = map(_number, box)
    _require(x2 > x1 and y2 > y1, 'Bounding-box label must have positive area')


def _events(events):
    _require(isinstance(events, list) and len(events) > 0, 'DVC requires a nonempty ground-truth event list')
    for event in events:
        _require(isinstance(event, dict), 'Malformed DVC ground-truth event')
        start = event.get('start', event.get('start_time'))
        end = event.get('end', event.get('end_time'))
        _span(start, end)
        caption = event.get('caption') or event.get('event_caption') or event.get('description')
        _require(_text(caption), 'DVC event is missing its ground-truth caption')


def _validate_table(frame, task):
    required = {'index', 'video', 'question', 'answer'}
    _require(required.issubset(frame.columns) and len(frame) > 0, f'{task} table is empty or missing required columns')
    for field in required:
        _require(frame[field].notna().all() and frame[field].map(lambda value: str(value).strip() != '').all(),
                 f'{task} table contains a missing {field}')
    _require(not frame['index'].map(str).duplicated().any(), f'{task} table contains duplicate indexes')
    if task == 'VQA':
        _require(frame['answer'].isin(list('ABCD')).all(), 'VQA requires explicit A-D ground-truth answers')
    elif task == 'EventVerification':
        _require(frame['answer'].map(lambda value: str(value).strip().lower() in ('yes', 'no')).all(),
                 'EventVerification requires explicit yes/no ground-truth answers')
    elif task == 'Temporal':
        _require('duration' in frame, 'Temporal table requires real video duration metadata')
        for _, row in frame.iterrows():
            answer = json.loads(row['answer'])
            _span(answer.get('start'), answer.get('end'))
            _require(_number(row['duration']) > 0, 'Temporal video duration must be positive')
    elif task == 'DVC':
        _require(not frame['video'].duplicated().any(), 'DVC requires one table row per video')
        for answer in frame['answer']:
            _events(json.loads(answer))
    else:
        raise ValueError(f'Unknown local video task: {task}')


def validate_local_video_inputs(local_data_file, local_video_root, task):
    """Validate a canonical labeled table before a native constructor loads it.

    No cache or remote source is consulted. Explicit paths must both exist; local
    media references are relative to the supplied root and may not escape it.
    """
    _require(local_data_file is not None and local_video_root is not None,
             'local_data_file and local_video_root must be supplied together')
    data_file = Path(local_data_file).expanduser().resolve(strict=True)
    root = Path(local_video_root).expanduser().resolve(strict=True)
    _require(root.is_dir() and data_file.is_file(), 'Local video inputs must be a table and a directory')
    _require(data_file.suffix.lower() == '.tsv', 'Canonical local video data must be a TSV table')
    frame = pd.read_csv(data_file, sep='\t', keep_default_na=False)
    _validate_table(frame, task)
    for video in frame['video'].unique():
        media = str(video)
        if task != 'EventVerification' and not media.endswith('.mp4'):
            media += '.mp4'
        _require(not Path(media).is_absolute(), 'Local video table must contain relative media identities')
        _local_path(root, media)
    return {'data_file': str(data_file), 'root': str(root)}


def validate_bert_score_model(model_dir):
    """Require a complete local roberta-large checkpoint, never a Hub model ID."""
    _require(model_dir is not None, 'Local DVC scoring requires bert_score_model pointing to local roberta-large assets')
    path = Path(model_dir).expanduser().resolve(strict=True)
    _require(path.is_dir(), 'BERTScore model must be a local directory')
    config = _json(path / 'config.json')
    _require(config.get('model_type') == 'roberta' and config.get('num_hidden_layers') == 24,
             'Local DVC requires the roberta-large architecture')
    _local_path(path, 'vocab.json')
    _local_path(path, 'merges.txt')
    weights = next((path / name for name in ('model.safetensors', 'pytorch_model.bin') if (path / name).is_file()), None)
    if weights is None:
        index = next((path / name for name in ('model.safetensors.index.json', 'pytorch_model.bin.index.json')
                      if (path / name).is_file()), None)
        _require(index is not None, 'Local BERTScore model weights are missing')
        weight_map = _json(index).get('weight_map')
        _require(isinstance(weight_map, dict) and weight_map, 'Malformed local model weight index')
        for shard in set(weight_map.values()):
            _local_path(path, shard)
    else:
        _local_path(path, weights)
    return path


def _annotation_paths(root, task):
    source = _local_path(root, SOURCE_DIRS[task], directory=True)
    if task in ('VQA', 'Temporal'):
        annotations = _local_path(root, source / 'data_jsons/annotations', directory=True)
        if task == 'VQA':
            paths = [annotations / name for name in VQA_FILES if (annotations / name).is_file()]
            required = set(VQA_FILES) - {'Metropolis_VQA_Verification_Final_ITS_Data.json', 'VANTAGE_VQA_Verification_Final_ITS_Data.json'}
            _require(required.issubset({p.name for p in paths}) and len(paths) == 5,
                     'VQA requires four metrics files and exactly one Verification ITS annotation')
        else:
            paths = sorted(annotations.glob('*.json'))
    elif task == 'EventVerification':
        paths = [source / name for name in EVENT_FILES]
    elif task == 'DVC':
        paths = sorted((source / 'events').glob('*_events.json'))
    elif task == '2DGrounding':
        paths = [source / 'refdrone_test_llava.json']
    elif task == '2DPointing':
        paths = [source / 'Metropolis2DPointing.tsv']
    elif task == 'SOT':
        directories = [p for p in sorted(source.iterdir()) if p.is_dir()]
        _require(bool(directories), 'SOT has no sequence directories')
        paths = [p / 'gt.json' for p in directories]
    else:
        from .dataset.metropolis2d.astro_2d_dataset import Astro2DDetectionDataset
        paths = []
        for layout in Astro2DDetectionDataset._LAYOUT.values():
            sequence = source / layout['data_root'].rsplit('/', 1)[1]
            labels = _local_path(root, sequence / 'labels', directory=True)
            paths.extend(sorted(labels.glob('*.txt')))
    _require(bool(paths), f'{task} has no original ground-truth annotations')
    return [_local_path(root, path) for path in paths]


def validate_task_sources(assets_dir, task, dataset_kwargs=None):
    """Validate original labels and media; return required annotation fingerprints.

    The original filtered source layout is authoritative, not a prediction table.
    Supplied manifest constructor paths must resolve to these same original
    sources, so an alternate bundle-local label tree cannot evade validation.
    """
    _require(task in TASKS, 'Unknown scoring asset task')
    root = Path(assets_dir).resolve(strict=True)
    source = _local_path(root, SOURCE_DIRS[task], directory=True)
    paths = _annotation_paths(root, task)
    if dataset_kwargs is not None:
        expected = {}
        if task in VIDEO_TASKS:
            expected['local_video_root'] = source if task == 'EventVerification' else source / 'videos'
            _local_path(root, dataset_kwargs.get('local_data_file', ''))
        elif task == '2DGrounding':
            expected = {'data_root': source, 'annotation_file': paths[0]}
        elif task == '2DPointing':
            expected = {'data_root': source}
        elif task == 'SOT':
            expected = {'prepared_data_dir': source}
        else:
            from .dataset.metropolis2d.astro_2d_dataset import Astro2DDetectionDataset
            roots = dataset_kwargs.get('data_roots')
            _require(isinstance(roots, dict) and set(roots) == set(Astro2DDetectionDataset._LAYOUT),
                     'ASTRO manifest must identify exactly the registered original child sequences')
            for name, layout in Astro2DDetectionDataset._LAYOUT.items():
                actual = _local_path(root, roots[name], directory=True)
                _require(actual == source / layout['data_root'].rsplit('/', 1)[1],
                         'ASTRO manifest path does not identify its original sequence')
        for name, original in expected.items():
            _require(name in dataset_kwargs, f'{task} manifest is missing {name}')
            actual = _local_path(root, dataset_kwargs[name], directory=original.is_dir())
            _require(actual == original, f'{task} manifest path does not identify its original source')
    if task in ('VQA', 'Temporal'):
        for path in paths:
            for item in _records(path):
                video = item.get('q_uid', item.get('vid')) if task == 'VQA' else item.get('vid')
                _require(_text(video) and _text(item.get('question')), f'{task} annotation lacks video ID or question')
                video = video.removesuffix('.json') if task == 'VQA' else video
                filename = video if video.endswith('.mp4') else video + '.mp4'
                _local_path(root, source / 'videos' / filename)
                if task == 'VQA':
                    _vqa_option(item.get('gt_option'))
                    options = item.get('options')
                    _require(isinstance(options, list) and 2 <= len(options) <= 4 and all(_text(v) for v in options),
                             'VQA source requires two to four explicit choices')
                    _require(_vqa_option(item.get('gt_option')) in tuple('ABCD')[:len(options)],
                             'VQA source ground truth does not identify an available choice')
                else:
                    answer = item.get('answer')
                    _require(isinstance(answer, str), 'Temporal source is missing ground-truth timestamps')
                    match = re.match(r'<([\d.]+)>\s*<([\d.]+)>', answer)
                    _require(match is not None, 'Malformed Temporal source timestamps; no default interval is permitted')
                    _span(*match.groups())
    elif task == 'EventVerification':
        from .dataset.metropolis_event_verification import MetropolisEventVerification
        for path in paths:
            for item in _records(path, 'bcq'):
                _require(item.get('id') is not None and str(item['id']).strip() != '', 'Event source lacks original ID')
                _require(_text(item.get('question')), 'Event source lacks question')
                _require(str(item.get('answer', '')).strip().lower() in ('yes', 'no'), 'Event source lacks yes/no ground truth')
                video = item.get('video', item.get('video_id'))
                _require(_text(video), 'Event source lacks video ID')
                _local_path(root, path.parent / video)
        items = MetropolisEventVerification._load_items(source)
        identities = [(item.get('source_corpus', ''), str(item['id'])) for item in items]
        _require(len(set(identities)) == len(identities), 'Duplicate original EventVerification IDs within a corpus')
    elif task == 'DVC':
        from .dataset.metropolis_dvc import MetropolisDVC
        adapter = MetropolisDVC(test_mode=True)
        resolved = []
        for path in paths:
            raw = _json(path)
            _events(raw.get('events') if isinstance(raw, dict) else raw)
            video_id = path.name.removesuffix('_events.json')
            videos = source / 'videos'
            exact = videos / (video_id + '.mp4')
            with_ext = videos / video_id
            if not exact.is_file() and not (video_id.endswith('.mp4') and with_ext.is_file()):
                matches = list(videos.glob(video_id.replace('[', '[[]') + '*.mp4'))
                _require(len(matches) == 1, 'DVC source video ID has missing or ambiguous media')
            stem = adapter._resolve_video_stem(videos, video_id)
            _require(stem is not None, 'DVC source video cannot be resolved')
            _local_path(root, videos / (stem + '.mp4'))
            resolved.append(stem)
        _require(len(set(resolved)) == len(resolved), 'DVC has multiple annotations for the same video')
    elif task == '2DGrounding':
        for item in _records(paths[0]):
            _require(_text(item.get('media')), 'Grounding source lacks image identity')
            _local_path(root, source / item['media'])
            meta = item.get('_meta') or {}
            _require(_text(meta.get('sentence')), 'Grounding source lacks referring expression')
            _require(_number(meta.get('image_width')) > 0 and _number(meta.get('image_height')) > 0,
                     'Grounding source lacks real image dimensions')
            turns = [turn for turn in item.get('conversations', []) if turn.get('from') == 'gpt']
            _require(len(turns) == 1, 'Grounding source lacks an unambiguous ground-truth response')
            boxes = json.loads(turns[0].get('value', ''))
            _require(isinstance(boxes, list) and bool(boxes), 'Grounding source lacks bounding-box labels')
            for box in boxes:
                _bbox(box)
    elif task == '2DPointing':
        frame = pd.read_csv(paths[0], sep='\t', keep_default_na=False)
        _require({'index', 'question', 'answer', 'target_point'}.issubset(frame.columns) and len(frame) > 0,
                 'Pointing source lacks original IDs, answers, or target points')
        _require(not frame['index'].map(str).duplicated().any(), 'Pointing source contains duplicate IDs')
        _require(frame['answer'].isin(list('ABCD')).all(), 'Pointing source lacks A-D ground truth')
        _require(frame['target_point'].map(lambda value: str(value).strip() != '').all(), 'Pointing source lacks target-point labels')
        for value in frame['target_point']:
            try:
                point = ast.literal_eval(str(value))
            except (SyntaxError, ValueError) as exc:
                raise ValueError('Malformed Pointing target-point ground truth') from exc
            _require(isinstance(point, (list, tuple)) and len(point) == 2, 'Pointing target must have two coordinates')
            for coordinate in point:
                _number(coordinate)
        from .smp import toliststr
        _require('image_path' in frame, 'Pointing source lacks local image identities')
        for _, row in frame.iterrows():
            images = toliststr(row['image_path'])
            _require(len(images) == 1 and _text(images[0]), 'Pointing source requires one local image identity')
            _local_path(root, source / images[0])
    elif task == 'SOT':
        metadata = {}
        labels = set()
        metadata_path = (dataset_kwargs or {}).get('metadata_path')
        if metadata_path:
            metadata_path = _local_path(root, metadata_path)
            for line in metadata_path.read_text(encoding='utf-8').splitlines():
                if line.strip():
                    entry = json.loads(line)
                    _require(_text(entry.get('seq_id')) and entry['seq_id'] not in metadata, 'Malformed or duplicate SOT metadata identity')
                    metadata[entry['seq_id']] = entry
        for path in paths:
            meta = _json(path)
            if 'frame_ids' in meta:
                ids = meta['frame_ids']
                boxes = meta.get('gt_bboxes')
                _require(_text(meta.get('label')), 'SOT source lacks original sequence label')
                _require(isinstance(boxes, dict), 'SOT source lacks per-frame ground truth')
                _require(isinstance(ids, list) and len(ids) >= 2 and all(type(i) is int for i in ids)
                         and len(set(ids)) == len(ids), 'Malformed SOT frame identities')
                expected = {str(i) for i in ids[1:]}
                _require(expected.issubset(boxes) and set(boxes).issubset({str(i) for i in ids}),
                         'SOT ground truth must explicitly label every evaluation frame, including absent objects')
            else:
                count = meta.get('nframes')
                _require(type(count) is int and count >= 2, 'Malformed compact SOT frame count')
                ids = list(range(count))
                boxes = meta.get('bboxes')
                _require(isinstance(boxes, dict), 'Compact SOT lacks frame labels')
                expected = {f'frame_{i}' for i in ids[1:]}
                _require(set(boxes) == expected, 'Compact SOT must explicitly label every evaluation frame')
            entry = metadata.get(path.parent.name, {})
            init_bbox = meta.get('init_bbox') or entry.get('init_bbox')
            _bbox(init_bbox)
            if 'frame_ids' in meta and str(ids[0]) in boxes:
                _require(boxes[str(ids[0])] == init_bbox, 'SOT initialization box disagrees with its frame ground truth')
            label = meta.get('label') if 'frame_ids' in meta else entry.get('label') or path.parent.name.replace('__', '/')
            _require(_text(label) and label not in labels, 'SOT sequence labels must be present and unique')
            labels.add(label)
            for box in boxes.values():
                if box is not None:
                    _bbox(box)
            for index in range(len(ids)):
                _local_path(root, path.parent / 'frames' / f'f{index:02d}.png')
            _local_path(root, path.parent / 'frames/crop.png')
        if metadata_path:
            paths.append(metadata_path)
    else:
        from .dataset.metropolis2d.astro_2d_dataset import Astro2DDetectionDataset, parse_kitti_label
        for layout in Astro2DDetectionDataset._LAYOUT.values():
            sequence = source / layout['data_root'].rsplit('/', 1)[1]
            images_dir = _local_path(root, sequence / 'images', directory=True)
            images = [p for p in sorted(images_dir.iterdir()) if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')]
            _require(bool(images), 'ASTRO sequence has no images')
            stems = [p.stem for p in images]
            _require(len(set(stems)) == len(stems), 'ASTRO image identities have ambiguous label stems')
            labels = {p.stem: p for p in paths if p.parent == sequence / 'labels'}
            _require(set(stems) == set(labels), 'ASTRO requires an independent label file for every image, and no unmatched labels')
            for image in images:
                _local_path(root, image)
                label = labels[image.stem]
                lines = [line for line in label.read_text(encoding='utf-8').splitlines() if line.strip()]
                for line in lines:
                    fields = line.split()
                    _require(len(fields) >= 15, 'Malformed KITTI ground-truth label')
                    # Preserve the native KITTI policy, including finite inverted
                    # boxes. The scorer clamps their area to zero; deleting or
                    # repairing them would change its ground-truth denominator.
                    for value in fields[4:8]:
                        _number(value)
                _require(len(parse_kitti_label(str(label))) == len(lines), 'Native ASTRO parser skipped a malformed label')
    return paths


def _duration(video):
    result = subprocess.run(
        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', str(video)],
        check=True, capture_output=True, text=True,
    )
    duration = _number(result.stdout.strip())
    _require(duration > 0, 'Video has no usable duration metadata')
    return duration


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _snapshot(frame, root, target):
    frame = frame.copy()
    for column in ('image_path', 'video_path'):
        if column in frame:
            frame[column] = frame[column].map(
                lambda value: str(Path(value).relative_to(root))
                if _text(value) and Path(value).is_absolute() and Path(value).is_relative_to(root) else value
            )
    frame.to_csv(target, sep='\t', index=False)


def prepare_assets(assets_dir, asset_id, tasks=None, bert_score_model=None, bert_score_revision=None,
                   sot_metadata_path=None):
    """Build native canonical tables and a fingerprinted local scoring manifest."""
    from .dataset import (
        MetropolisVQA, MetropolisTemporal, MetropolisEventVerification, MetropolisDVC,
        Metropolis2DGroundingDataset, VANTAGE_2DPointing, VANTAGE_SOT, Astro2DDetectionDataset,
    )
    root = Path(assets_dir).expanduser().resolve(strict=True)
    _require(root.is_dir() and _text(asset_id), 'Preparation requires a local source directory and nonempty asset ID')
    selected = list(tasks or TASKS)
    _require(bool(selected) and len(set(selected)) == len(selected) and set(selected).issubset(TASKS), 'Invalid or duplicate preparation tasks')
    manifest = {'format_version': 1, 'asset_id': asset_id, 'tasks': {}, 'files': {}}
    if 'DVC' in selected:
        _require(_text(bert_score_revision), 'DVC model requires an explicit immutable revision identifier')
        _require(bert_score_model is not None and not Path(bert_score_model).is_absolute(), 'BERTScore model path must be relative to the bundle')
        model = _local_path(root, bert_score_model, directory=True)
        validate_bert_score_model(model)
        model_files = {}
        for path in sorted(model.rglob('*')):
            if path.is_file() and '.cache' not in path.relative_to(model).parts:
                _local_path(root, path)
                model_files[path.relative_to(model).as_posix()] = _sha256(path)
        manifest['metric_models'] = {'roberta_large': {
            'path': model.relative_to(root).as_posix(), 'revision': bert_score_revision,
            'files': model_files,
        }}
    # Validate every requested original source before writing generated tables.
    sot_kwargs = {}
    if sot_metadata_path:
        _require(not Path(sot_metadata_path).is_absolute(), 'SOT metadata path must be relative to the bundle')
        sot_kwargs['metadata_path'] = _local_path(root, sot_metadata_path).relative_to(root).as_posix()
    originals = {
        task: validate_task_sources(root, task, {'prepared_data_dir': SOURCE_DIRS[task], **sot_kwargs} if task == 'SOT' else None)
        for task in selected
    }
    for task in selected:
        source_diagnostics = {}
        source = root / SOURCE_DIRS[task]
        table = root / f'VANTAGE_{task}.tsv'
        _require(not table.is_symlink() and (not table.exists() or table.stat().st_nlink == 1),
                 'Refusing to overwrite a linked canonical output table')
        kwargs = {}
        if task in ('VQA', 'Temporal'):
            adapter = (MetropolisVQA if task == 'VQA' else MetropolisTemporal)(test_mode=True)
            for mapping_dir in (source / 'data_jsons/category_mapping', source / 'mappings'):
                if mapping_dir.is_dir():
                    for mapping in mapping_dir.glob('*.json'):
                        _local_path(root, mapping)
                        _require(isinstance(_json(mapping), dict), 'Malformed native category mapping')
                    adapter.category_mapping.update(adapter._load_category_mapping(str(mapping_dir)))
            rows = []
            durations = {}
            for path in originals[task]:
                for item in _records(path):
                    record = {**item, 'gt_option': _vqa_option(item.get('gt_option'))} if task == 'VQA' else item
                    row = adapter._process_annotation_item(record)
                    _require(row is not None, f'Native {task} converter rejected a validated source record')
                    if task == 'Temporal' and 'duration' not in item:
                        filename = row['video'] if row['video'].endswith('.mp4') else row['video'] + '.mp4'
                        video = source / 'videos' / filename
                        if video not in durations:
                            durations[video] = _duration(video)
                        row['duration'] = durations[video]
                    rows.append(row)
            rows.sort(key=lambda row: (row['video'], row.get('qid', '')))
            for index, row in enumerate(rows):
                row['index'] = index
            frame = pd.DataFrame(rows)
        elif task == 'EventVerification':
            frame = pd.DataFrame(MetropolisEventVerification._load_items(source))
            if 'index' not in frame:
                frame['index'] = range(len(frame))
            frame['index'] = frame['index'].astype(str)
        elif task == 'DVC':
            adapter = MetropolisDVC(test_mode=True)
            adapter._generate_tsv_from_events(source, table)
            frame = pd.read_csv(table, sep='\t', keep_default_na=False)
            _require(len(frame) == len(originals[task]), 'Native DVC conversion dropped original annotations')
        elif task == '2DGrounding':
            kwargs = {'data_root': SOURCE_DIRS[task], 'annotation_file': originals[task][0].relative_to(root).as_posix()}
            dataset = Metropolis2DGroundingDataset(dataset='VANTAGE_2DGrounding', model_family='cr',
                                                 data_root=str(source), annotation_file=str(originals[task][0]))
            frame = dataset.data
            _require(len(frame) == len(_records(originals[task][0])), 'Native Grounding loader dropped original annotations')
        elif task == '2DPointing':
            kwargs = {'data_root': SOURCE_DIRS[task]}
            dataset = VANTAGE_2DPointing(data_root=str(source), model_family='cr')
            frame = dataset.data
            _require(len(frame) == len(pd.read_csv(originals[task][0], sep='\t')), 'Native Pointing loader dropped original annotations')
        elif task == 'SOT':
            kwargs = {'prepared_data_dir': SOURCE_DIRS[task], **sot_kwargs}
            native_kwargs = {'prepared_data_dir': str(source), 'model_family': 'cr'}
            if sot_metadata_path:
                native_kwargs['metadata_path'] = str(_local_path(root, sot_metadata_path))
            frame = VANTAGE_SOT(**native_kwargs).data
            _require(len(frame) == sum(path.name == 'gt.json' for path in originals[task]),
                     'Native SOT loader dropped original sequences')
        else:
            from .dataset.metropolis2d.astro_2d_dataset import compute_bbox_area
            roots = {}
            frames = []
            offset = 0
            for name, layout in Astro2DDetectionDataset._LAYOUT.items():
                sequence = source / layout['data_root'].rsplit('/', 1)[1]
                roots[name] = sequence.relative_to(root).as_posix()
                child = Astro2DDetectionDataset(dataset=name, data_root=str(sequence), model_family='cr')
                non_positive = sum(
                    obj['label'] == 'person' and compute_bbox_area(obj['bbox']) == 0
                    for objects in child._gt_cache.values() for obj in objects
                )
                if non_positive:
                    source_diagnostics[name] = {
                        'non_positive_area_person_ground_truth': non_positive,
                        'retained_by_native_min_area': non_positive if child.min_bbox_area <= 0 else 0,
                        'native_min_bbox_area': child.min_bbox_area,
                        'policy': 'Original labels unchanged; native filtering and denominators preserved',
                    }
                child_frame = child.data.copy()
                child_frame['original_index'] = child_frame['index']
                child_frame['SUB_DATASET'] = name
                child_frame['index'] = range(offset, offset + len(child_frame))
                offset += len(child_frame)
                frames.append(child_frame)
            frame = pd.concat(frames, ignore_index=True)
            kwargs = {'data_roots': roots}
        if task in VIDEO_TASKS:
            _validate_table(frame, task)
            frame.to_csv(table, sep='\t', index=False)
            video_root = source if task == 'EventVerification' else source / 'videos'
            validate_local_video_inputs(table, video_root, task)
            kwargs = {'local_data_file': table.name, 'local_video_root': video_root.relative_to(root).as_posix()}
        else:
            _snapshot(frame, root, table)
        manifest['tasks'][task] = {'rows': len(frame), 'dataset_kwargs': kwargs}
        if source_diagnostics:
            manifest['tasks'][task]['annotation_diagnostics'] = source_diagnostics
        for path in originals[task] + [table]:
            manifest['files'][path.relative_to(root).as_posix()] = _sha256(path)
        # Category mappings influence native category-level metrics and must also
        # be fingerprinted when used by the video converters.
        if task in ('VQA', 'Temporal'):
            for mapping_dir in (source / 'data_jsons/category_mapping', source / 'mappings'):
                if mapping_dir.is_dir():
                    for path in sorted(mapping_dir.glob('*.json')):
                        path = _local_path(root, path)
                        manifest['files'][path.relative_to(root).as_posix()] = _sha256(path)
    output = root / 'scoring-assets.json'
    _require(not output.is_symlink() and (not output.exists() or output.stat().st_nlink == 1),
             'Refusing to overwrite a linked scoring manifest')
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets-dir', required=True)
    parser.add_argument('--asset-id', required=True)
    parser.add_argument('--tasks', nargs='+', choices=TASKS)
    parser.add_argument('--bert-score-model', help='Relative local roberta-large model directory inside the bundle')
    parser.add_argument('--bert-score-revision', help='Immutable source revision of the local model assets')
    parser.add_argument('--sot-metadata-path', help='Optional relative original SOT metadata JSONL inside the bundle')
    args = parser.parse_args(argv)
    manifest = prepare_assets(args.assets_dir, args.asset_id, args.tasks, args.bert_score_model,
                              args.bert_score_revision, args.sot_metadata_path)
    print(json.dumps({'asset_id': manifest['asset_id'], 'rows': {task: item['rows'] for task, item in manifest['tasks'].items()}}, sort_keys=True))


if __name__ == '__main__':
    main()
