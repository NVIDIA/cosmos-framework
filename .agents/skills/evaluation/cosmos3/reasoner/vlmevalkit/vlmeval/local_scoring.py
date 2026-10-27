# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Credential-free native scoring against a fingerprinted, private local asset bundle.

Prediction files are not annotation sources. Only their verified identities and
prediction values cross into a new table reconstructed from the local dataset.
The output directory contains private native artifacts and must not be published.
"""
import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import hashlib
import inspect
from io import BytesIO
import json
import logging
import math
import numbers
import os
from pathlib import Path
import re
import traceback

import pandas as pd

TASKS = ('VQA', 'EventVerification', 'Temporal', 'DVC', '2DGrounding',
         '2DPointing', 'SOT', 'Astro2D')
_CONFIG_DIR = Path(__file__).resolve().parents[1] / 'cosmos_eval/data/metropolis'
_CLASSES = {
    'VQA': 'MetropolisVQA', 'EventVerification': 'MetropolisEventVerification',
    'Temporal': 'MetropolisTemporal', 'DVC': 'MetropolisDVC',
    '2DGrounding': 'Metropolis2DGroundingDataset', '2DPointing': 'VANTAGE_2DPointing',
    'SOT': 'VANTAGE_SOT', 'Astro2D': 'Astro2DBenchDataset',
}
_KWARGS = {
    **{task: {'local_data_file', 'local_video_root'}
       for task in ('VQA', 'EventVerification', 'Temporal', 'DVC')},
    '2DGrounding': {'data_root', 'annotation_file'}, '2DPointing': {'data_root'},
    'SOT': {'prepared_data_dir', 'metadata_path'}, 'Astro2D': {'data_roots'},
}
_ASTRO_CHILDREN = ('VANTAGE_Astro2D_C0065', 'VANTAGE_Astro2D_IVAKPI05',
                   'VANTAGE_Astro2D_IVAKPI11')


def _task(task):
    if task.startswith('VANTAGE_'):
        task = task[len('VANTAGE_'):]
    if task not in TASKS:
        raise ValueError('Unknown local scoring task')
    return task


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _inside(root, value, directory=False):
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError('Asset paths must be nonempty bundle-relative paths')
    path = (root / value).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Asset path escapes the local bundle')
    if not (path.is_dir() if directory else path.is_file()):
        raise ValueError('Required local asset is absent')
    return path


def _require_fingerprint(bundle, path):
    path = Path(path).resolve()
    if not path.is_relative_to(bundle['root']):
        raise ValueError('Ground-truth path escapes the local bundle')
    key = path.relative_to(bundle['root']).as_posix()
    if key not in bundle['manifest']['files']:
        raise ValueError('Required ground truth is not fingerprinted')


@contextmanager
def _private_native_output():
    # Native scorers print categories, sequence identities and occasionally GT.
    # Do not collect these in a public log or interpolate their exceptions.
    previous = logging.root.manager.disable
    with open(os.devnull, 'w') as sink, redirect_stdout(sink), redirect_stderr(sink):
        logging.disable(logging.CRITICAL)
        try:
            yield
        finally:
            logging.disable(previous)


def _validate_source(bundle, task):
    from .scoring_assets import validate_local_video_inputs, validate_task_sources
    config = bundle['configs'][task]
    with _private_native_output():
        originals = validate_task_sources(
            bundle['root'], task,
            bundle['manifest']['tasks'][task]['dataset_kwargs'],
        )
    for path in originals:
        _require_fingerprint(bundle, path)
    if task in ('VQA', 'EventVerification', 'Temporal', 'DVC'):
        _require_fingerprint(bundle, config['local_data_file'])
        with _private_native_output():
            validate_local_video_inputs(config['local_data_file'], config['local_video_root'], task)
    elif task == '2DPointing':
        table = Path(config['data_root']) / 'Metropolis2DPointing.tsv'
        _require_fingerprint(bundle, table)
        if not table.is_file():
            raise ValueError('Local pointing table is absent; downloading is not permitted')
    elif task == '2DGrounding':
        _require_fingerprint(bundle, config['annotation_file'])
    elif task == 'SOT' and 'metadata_path' in config:
        _require_fingerprint(bundle, config['metadata_path'])


def load_assets(assets_dir, tasks):
    """Validate local paths, annotation hashes, and task label coverage.

    Returns bundle metadata with ``root`` (Path), ``manifest``, ``configs`` and
    ``manifest_sha256``. No credentials, cache lookup, endpoint, or model is used.
    """
    if isinstance(tasks, str):
        tasks = [tasks]
    selected = tuple(dict.fromkeys(_task(task) for task in tasks))
    if not selected:
        raise ValueError('At least one local task is required')
    root = Path(assets_dir).expanduser().resolve()
    manifest_file = root / 'scoring-assets.json'
    with open(manifest_file) as stream:
        manifest = json.load(stream)
    if (manifest.get('format_version') != 1 or not isinstance(manifest.get('asset_id'), str)
            or not manifest['asset_id'].strip() or not isinstance(manifest.get('tasks'), dict)
            or not isinstance(manifest.get('files'), dict) or not manifest['files']):
        raise ValueError('Invalid local scoring asset manifest')
    if set(manifest['tasks']) - set(TASKS):
        raise ValueError('Unknown task in local scoring manifest')
    for relative, expected in manifest['files'].items():
        path = _inside(root, relative)
        if not isinstance(expected, str) or not re.fullmatch('[a-f0-9]{64}', expected) or _sha256(path) != expected:
            raise ValueError('Local asset fingerprint mismatch')
    configs = {}
    for task in selected:
        entry = manifest['tasks'].get(task)
        if (not isinstance(entry, dict) or type(entry.get('rows')) is not int or entry['rows'] <= 0
                or not isinstance(entry.get('dataset_kwargs'), dict)):
            raise ValueError('Requested task is absent or incomplete in local assets')
        kwargs = entry['dataset_kwargs']
        required = _KWARGS[task] - ({'metadata_path'} if task == 'SOT' else set())
        if not required <= set(kwargs) or set(kwargs) - _KWARGS[task]:
            raise ValueError('Manifest may supply only the native local dataset paths')
        with open(_CONFIG_DIR / f'VANTAGE_{task}.json') as stream:
            config = json.load(stream)
        if config.get('class') != _CLASSES[task] or config.get('dataset') != f'VANTAGE_{task}':
            raise ValueError('Committed native dataset config is inconsistent')
        if task not in ('VQA', 'EventVerification', 'Temporal', 'DVC'):
            config['model_family'] = 'cr'
        for key, value in kwargs.items():
            if key == 'data_roots':
                if not isinstance(value, dict) or set(value) != set(_ASTRO_CHILDREN):
                    raise ValueError('ASTRO requires all three registered local child roots')
                config[key] = {name: str(_inside(root, path, directory=True)) for name, path in value.items()}
            else:
                config[key] = str(_inside(root, value, directory=key in {
                    'data_root', 'local_video_root', 'prepared_data_dir'}))
        configs[task] = config
    bundle = {'root': root, 'manifest': manifest, 'configs': configs,
              'manifest_sha256': _sha256(manifest_file)}
    for task in selected:
        _validate_source(bundle, task)
    if 'DVC' in selected:
        model = manifest.get('metric_models', {}).get('roberta_large', {})
        if not isinstance(model, dict) or not isinstance(model.get('revision'), str) or not model['revision'].strip():
            raise ValueError('DVC requires a revisioned, genuine local roberta-large model')
        bundle['bert_score_model'] = str(_inside(root, model.get('path'), directory=True))
        from .scoring_assets import validate_bert_score_model
        validate_bert_score_model(bundle['bert_score_model'])
        model_root = Path(bundle['bert_score_model'])
        files = {path.relative_to(model_root).as_posix(): path
                 for path in model_root.rglob('*')
                 if path.is_file() and '.cache' not in path.relative_to(model_root).parts}
        expected = model.get('files')
        if not isinstance(expected, dict) or not expected or set(files) != set(expected):
            raise ValueError('DVC metric-model files do not match the fingerprinted bundle')
        for relative, path in files.items():
            if not path.resolve().is_relative_to(root) or _sha256(path) != expected[relative]:
                raise ValueError('DVC metric-model fingerprint mismatch')
    return bundle


def dataset_config(assets_dir, task):
    """Return the committed native inference config with absolute local paths."""
    task = _task(task)
    return load_assets(assets_dir, [task])['configs'][task]


def _build_dataset(bundle, task):
    from . import dataset as native
    config = dict(bundle['configs'][task])
    cls = getattr(native, config.pop('class'))
    with _private_native_output():
        dataset = cls(**config)
    expected = bundle['manifest']['tasks'][task]['rows']
    if len(dataset.data) != expected or 'index' not in dataset.data:
        raise ValueError('Native dataset silently omitted required local examples')
    keys = [_identity(value) for value in dataset.data['index']]
    if len(set(keys)) != expected:
        raise ValueError('Native dataset identities are not unique')
    if task == 'SOT':
        if set(dataset.data['index']) != set(dataset._gt_cache) or any(
                type(key) is not int for key in dataset._gt_cache):
            raise ValueError('SOT native integer identity cache is incomplete')
    return dataset


def build_local_dataset(assets_dir, task):
    """Build and validate the native dataset without constructing a model."""
    task = _task(task)
    return _build_dataset(load_assets(assets_dir, [task]), task)


def _identity(value):
    # Integral numeric indexes and their canonical decimal spelling are equal;
    # distinct textual IDs (e.g. '00' and '0') are deliberately not collapsed.
    if isinstance(value, bool) or value is None:
        raise ValueError('Missing or invalid example identity')
    if isinstance(value, numbers.Integral):
        return str(int(value))
    if isinstance(value, numbers.Real):
        if math.isfinite(value) and value.is_integer():
            return str(int(value))
        raise ValueError('Invalid numeric example identity')
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError('Missing or invalid example identity')
    return value


def _stable_identity(value):
    if isinstance(value, str):
        if not value.strip():
            raise ValueError('Missing stable example identity')
        return value
    return _identity(value)


def _read_predictions(predictions):
    path = Path(predictions).expanduser().resolve()
    contents = path.read_bytes()
    stream = BytesIO(contents)
    if path.suffix.lower() in {'.tsv', '.csv'}:
        table = pd.read_csv(stream, sep='\t' if path.suffix.lower() == '.tsv' else ',',
                            dtype=str, keep_default_na=False)
    elif path.suffix.lower() == '.xlsx':
        table = pd.read_excel(stream, dtype=object, keep_default_na=False)
    elif path.suffix.lower() == '.json':
        rows = json.loads(contents)
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise ValueError('Prediction JSON must be an array of row objects')
        table = pd.DataFrame(rows)
    else:
        raise ValueError('Predictions must be TSV, CSV, XLSX, or JSON rows')
    if table.empty or not {'index', 'prediction'} <= set(table):
        raise ValueError('Predictions require nonempty index and prediction columns')
    return table, hashlib.sha256(contents).hexdigest()


def _identity_columns(task, trusted):
    if task == 'Astro2D':
        return ['SUB_DATASET', 'original_index', 'image_filename']
    if task == 'SOT':
        return ['label']
    if task == '2DGrounding':
        return ['sentence']
    if task == '2DPointing':
        return [column for column in ('question_id', 'question', 'A', 'B', 'C', 'D') if column in trusted]
    # VQA's native fallback qid is shared by every question in one video.
    # The actual question (including options) must match, not just video/qid.
    return [column for column in ('video', 'qid', 'id', 'question', 'options') if column in trusted]


def _trusted_predictions(task, dataset, predictions, allow_subset):
    trusted = dataset.data
    keys = [_identity(value) for value in trusted['index']]
    lookup = {key: position for position, key in enumerate(keys)}
    incoming = [_identity(value) for value in predictions['index']]
    if len(set(incoming)) != len(incoming):
        raise ValueError('Duplicate prediction identity')
    if set(incoming) - set(keys):
        raise ValueError('Prediction identity does not match the local dataset')
    if not allow_subset and set(incoming) != set(keys):
        raise ValueError('Full local dataset coverage is required; use allow_subset explicitly')
    identities = _identity_columns(task, trusted)
    if not set(identities) <= set(predictions):
        raise ValueError('Predictions must retain the native stable identity columns: ' + ', '.join(identities))
    rows = trusted.iloc[[lookup[key] for key in incoming]].copy().reset_index(drop=True)
    for column in identities:
        if any(_stable_identity(left) != _stable_identity(right)
               for left, right in zip(predictions[column], rows[column])):
            raise ValueError('Prediction stable identity does not match its native index')
    values = list(predictions['prediction'])
    if any(not isinstance(value, str) or not value.strip() or value.strip().lower() in {'null', 'nan', 'none'}
           or any(marker in value for marker in ('VIDEO_LOAD_FAILED', 'Failed to obtain answer via API.'))
           for value in values):
        raise ValueError('Missing or failed prediction; no score was produced')
    rows['prediction'] = values
    # Normalize only after verified native identities. Both sides of native
    # joins then survive TSV numeric inference, including textual '00' IDs.
    if task in ('VQA', 'EventVerification', 'Temporal', 'DVC', '2DPointing'):
        rows['index'] = [lookup[key] for key in incoming]
        dataset.data = trusted.copy()
        dataset.data['index'] = list(range(len(trusted)))
    if task == 'SOT':
        rows['index'] = rows['index'].astype(int)
    return rows


def _validate_predictions(task, dataset, rows):
    for position, row in rows.iterrows():
        text = row['prediction']
        try:
            if task == 'EventVerification':
                if dataset._extract_answer(text) not in ('yes', 'no'):
                    raise ValueError('Unparseable event prediction would be silently skipped')
            elif task == 'DVC':
                events = dataset.parse_events_from_json(text)
                if not events:
                    raise ValueError('DVC prediction has no scorable events')
                for event in events:
                    if not (('start' in event or 'start_time' in event)
                            and ('end' in event or 'end_time' in event)):
                        raise ValueError('DVC event is missing timestamps')
                    start = dataset.parse_timestamp(event.get('start') or event.get('start_time', '0'))
                    end = dataset.parse_timestamp(event.get('end') or event.get('end_time', '0'))
                    if (not math.isfinite(start) or not math.isfinite(end) or end <= start
                            or not (event.get('caption') or event.get('description') or '').strip()):
                        raise ValueError('DVC contains an invalid or skipped event')
            elif task == 'Temporal':
                # Check the same branch native evaluate uses, but disallow its
                # silent [0,duration] fallback for malformed responses.
                try:
                    stamps = dataset.parse_timestamps_json(text, row['duration'], strict=True)
                except Exception:
                    stamps = dataset.parse_timestamps(text, row['duration'], strict=True)
                    # Native JSON's permissive fallback returns a full interval
                    # rather than trying angle timestamps. Reject that mismatch.
                    try:
                        native_stamps = dataset.parse_timestamps_json(text, row['duration'])
                    except Exception:
                        native_stamps = stamps
                    if native_stamps != stamps:
                        raise ValueError('Temporal prediction would trigger a silent full-duration fallback')
                if not all(math.isfinite(value) for value in stamps) or stamps[1] <= stamps[0]:
                    raise ValueError('Invalid temporal prediction interval')
            elif task == 'SOT':
                from .dataset.vantage_sot import parse_sot_response
                parsed = parse_sot_response(text, int(row['n_frames']), model_family='cr')
                if not parsed and text.strip() != '{}':
                    raise ValueError('Unparseable SOT prediction')
            # MCQ, grounding and detection keep malformed nonempty outputs in
            # their native denominators (zero / empty detection), not skip them.
        except Exception:
            raise ValueError(f'{task} prediction at row {position} cannot be scored without silent fallback') from None


def _evaluate(task, dataset, rows, output, bundle):
    path = output / f'predictions_VANTAGE_{task}.tsv'
    rows.to_csv(path, sep='\t', index=False)
    with _private_native_output():
        if task == 'Astro2D':
            # Invoke children directly: ConcatDataset replaces dataset names in
            # the entire path, which can corrupt user-chosen output directories.
            metrics = {}
            present = []
            for child in dataset.datasets:
                subset = rows[rows['SUB_DATASET'] == child].copy()
                if subset.empty:
                    continue
                subset['index'] = subset.pop('original_index')
                subset = subset.drop(columns=['SUB_DATASET'])
                child_path = output / f'{child}.tsv'
                subset.to_csv(child_path, sep='\t', index=False)
                result = dataset.dataset_map[child].evaluate(str(child_path), verbose=False)
                if result.get('error') or result.get('total_predictions') != len(subset):
                    raise ValueError('ASTRO native scorer omitted examples')
                metrics.update({f'{child}:{key}': value for key, value in result.items()})
                present.append(child)
            for key in ('f1', 'precision', 'recall', 'f1_0.95', 'f1_mIOU'):
                metrics[key] = sum(metrics[f'{child}:{key}'] for child in present) / len(present)
        else:
            kwargs = {'verbose': False}
            if task == 'DVC':
                kwargs.update(local_scoring=True, bert_score_model=bundle['bert_score_model'])
            metrics = dataset.evaluate(str(path), **kwargs)
    if isinstance(metrics, pd.DataFrame):
        if len(metrics) != 1:
            raise ValueError('Unexpected native metric table')
        metrics = metrics.iloc[0].to_dict()
    if not isinstance(metrics, dict) or metrics.get('error'):
        raise ValueError('Native scoring failed')
    if task == 'EventVerification' and metrics.get('Valid Predictions') != len(rows):
        raise ValueError('Native event scorer silently omitted predictions')
    if task == 'Temporal' and metrics.get('overall', {}).get('count') != len(rows):
        raise ValueError('Native temporal scorer silently omitted predictions')
    if task == 'DVC' and metrics.get('overall', {}).get('count') != len(rows):
        raise ValueError('Native DVC scorer silently omitted predictions')
    if task == '2DGrounding' and metrics.get('total_samples') != len(rows):
        raise ValueError('Native grounding scorer silently omitted predictions')
    if task == 'SOT' and len(metrics) != len(rows) + 1:
        raise ValueError('Native tracking scorer silently omitted predictions')
    return metrics, path


def _primary(task, metrics):
    name, unit, aggregation = {
        'VQA': ('accuracy', 'fraction', 'question_micro'),
        'EventVerification': ('macro_f1', 'fraction', 'class_macro'),
        'Temporal': ('iou', 'fraction', 'video_macro_of_question_means'),
        'DVC': ('SODA_c', 'fraction', 'video_macro'),
        '2DGrounding': ('Acc@0.5', 'percent', 'expression_micro'),
        '2DPointing': ('accuracy', 'fraction', 'question_micro'),
        'SOT': ('success_auc', 'fraction', 'sequence_macro'),
        'Astro2D': ('f1', 'percent', 'unweighted_sequence_macro'),
    }[task]
    if task in ('VQA', '2DPointing', 'SOT'):
        value = metrics['Overall']
    elif task in ('Temporal', 'DVC'):
        value = metrics['overall'][name]
    else:
        value = metrics[name]
    if not isinstance(value, numbers.Real) or not math.isfinite(value):
        raise ValueError('Native primary metric is not finite')
    return {'name': name, 'value': float(value), 'unit': unit, 'aggregation': aggregation}


def _json_value(value):
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if hasattr(value, 'item'):
        return _json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError('Native metrics contain a nonfinite value')
    return value


def score_predictions(task, predictions, assets_dir, output_dir, allow_subset=False):
    """Score native predictions locally; write private artifacts and summary.json.

    Stable identity columns from the native inference table must be retained.
    A subset is never labeled full coverage, even when explicitly permitted.
    Existing output directories are rejected rather than reused or overwritten.
    """
    task = _task(task)
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise ValueError('Local scoring requires a fresh output directory')
    bundle = load_assets(assets_dir, [task])
    predictions, prediction_sha256 = _read_predictions(predictions)
    dataset = _build_dataset(bundle, task)
    rows = _trusted_predictions(task, dataset, predictions, allow_subset)
    _validate_predictions(task, dataset, rows)
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    try:
        metrics, scoring_table = _evaluate(task, dataset, rows, output, bundle)
        source_paths = {Path(__file__).resolve(), Path(inspect.getfile(type(dataset))).resolve(),
                        Path(__file__).with_name('scoring_assets.py').resolve(),
                        _CONFIG_DIR / f'VANTAGE_{task}.json'}
        if task == 'Astro2D':
            source_paths.update(Path(inspect.getfile(type(child))).resolve()
                                for child in dataset.dataset_map.values())
        if task in ('Astro2D', '2DGrounding'):
            from .dataset.metropolis2d import utils as geometry
            source_paths.add(Path(inspect.getfile(geometry)).resolve())
        provenance = {
            'asset_id': bundle['manifest']['asset_id'], 'manifest_sha256': bundle['manifest_sha256'],
            'asset_files': bundle['manifest']['files'], 'predictions_sha256': prediction_sha256,
            'scoring_table_sha256': _sha256(scoring_table),
            'source_files': {str(path.relative_to(Path(__file__).resolve().parents[1])): _sha256(path)
                             for path in sorted(source_paths)},
        }
        if task == 'DVC':
            provenance['metric_model'] = bundle['manifest']['metric_models']['roberta_large']
        summary = _json_value({
            'format_version': 1, 'task': task, 'status': 'complete',
            'coverage': {'expected_rows': len(dataset.data), 'scored_rows': len(rows),
                         'subset': len(rows) != len(dataset.data)},
            'primary_metric': _primary(task, metrics), 'native_metrics': metrics,
            'annotation_diagnostics': bundle['manifest']['tasks'][task].get('annotation_diagnostics', {}),
            'provenance': provenance,
        })
        temporary = output / 'summary.json.tmp'
        temporary.write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
        temporary.replace(output / 'summary.json')
        return summary
    except Exception:
        (output / 'summary.json.tmp').unlink(missing_ok=True)
        (output / 'scoring-error.log').write_text(traceback.format_exc())
        raise ValueError('Local scoring failed; inspect private output/scoring-error.log') from None


def main(argv=None):
    # CLI subprocesses can inherit Jupyter's inline backend. Leave importing
    # callers' kernels untouched, but always use a headless backend here.
    os.environ['MPLBACKEND'] = 'Agg'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True)
    parser.add_argument('--predictions', required=True, type=Path)
    parser.add_argument('--assets-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--allow-subset', action='store_true')
    args = parser.parse_args(argv)
    try:
        summary = score_predictions(args.task, args.predictions, args.assets_dir,
                                    args.output_dir, args.allow_subset)
    except Exception:
        parser.exit(2, 'Local scoring failed: check local assets, native identities, coverage, and private output/scoring-error.log if created.\n')
    print(json.dumps({'task': summary['task'], 'coverage': summary['coverage'],
                      'primary_metric': summary['primary_metric'],
                      'annotation_diagnostics': summary['annotation_diagnostics']}))


if __name__ == '__main__':
    main()
