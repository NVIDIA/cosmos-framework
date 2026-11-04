#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pinned public MMSI-Video-Bench Uniform-50 download and evaluation.

This is not the distinct multi-image MMSI-Bench. Assets stay outside source;
only the maintained prompt/sampling and CosmosReason2 implementation are reused.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import sys
import threading
import time
from urllib.parse import urlsplit
import urllib.request
import zipfile

DATASET_REVISION = '8eab53d8eb8b5c0863cfbfe1428c936f3e9e243e'
EASI_REVISION = '52bb1dd2bc5cd58a291b3a780dab08d19c0ccf0d'
OFFICIAL_REVISION = '412f5a4a9eb6d89613c60b541b17df74f11a6736'
DATASET_URL = f'https://huggingface.co/datasets/rbler/MMSI-Video-Bench/resolve/{DATASET_REVISION}'
OFFICIAL_URL = f'https://raw.githubusercontent.com/InternRobotics/MMSI-Video-Bench/{OFFICIAL_REVISION}'
FILES = {
    'frames.zip': (f'{DATASET_URL}/frames.zip', '278bafcf4f8dc4abbaac7e0c10370aa4fe223c20e5fda47037478f04a44b1f08'),
    'ref_images.zip': (f'{DATASET_URL}/ref_images.zip', '50347c966703a749bacc87769c1f011deb012e82a9fd0bf0503e6551490ea7f7'),
    'mmsivideo.json': (f'{DATASET_URL}/mmsivideo.json', '58dd55fbbcf52eeafa368c07ac9ace0968a06d4a07da79856afff706db3b2bc9'),
    'MMSIVideoBench.tsv': (
        f'https://huggingface.co/datasets/lmms-lab-si/EASI-Leaderboard-Data/resolve/{EASI_REVISION}/MMSIVideoBench.tsv',
        '0fb24d10e25d2387496a641b3807ad570303d442759f3e31fd7f93de565bf3bd'),
    'official_README.md': (f'{OFFICIAL_URL}/README.md', 'c4fb14c27445faeb16c90bc56fa4040ebe628229b50a76093bf75f13ed7a2c1c'),
    'official_dataset.py': (f'{OFFICIAL_URL}/dataset.py', 'cb719ea7eaf03fe98b9b6db31ec2c5fa07f25d76982cdd6ad64a950b48355801'),
    'official_evaluation.py': (f'{OFFICIAL_URL}/evaluation.py', '6424797dc6fe1517839e1fd8c4027d605fd344dc8822418cd892e1062603011d'),
}
WARNINGS = [
    'Publisher license: CC BY 4.0. Attribution: MMSI-Video-Bench, Jingli Lin et al. (2025), arXiv:2512.10863; InternRobotics/MMSI-Video-Bench.',
    'Underlying third-party media redistribution rights are not independently verified (including ScanNet, Ego4D and Waymo). Keep media local; no raw-media redistribution. Further redistribution/commercial use requires underlying rights review.',
    'Independent public Uniform-50 measurement, not reference parity. Paper reference 27.2%; exact benchmark-specific paper recipe is undisclosed.',
    'Pinned EASI TSV differs from the maintained adapter stale MD5; SHA256 is pinned and selected IDs/labels/options/prompts/media are checked against official JSON.',
    'Maintained no-judge extraction is diagnostic only: it can undercount valid verbose outputs compared with the official extractor.',
]
PROTOCOL = {
    'setting': 'MMSIVideoBench_50frame (Uniform-50)',
    'sampling': 'Maintained global numpy.linspace(0,total-1,min(50,total),dtype=int) across canonical annotation frames, segment order preserved; all references',
    'prompt': 'Maintained MMSIVideoBench.build_prompt; official system + newline + task + user + format text interleaved with canonical frames/references; no hints',
    'temperature': 0, 'top_p': 1.0, 'top_k': -1, 'seed': 1,
    'max_tokens': 2048, 'repetition_penalty': 1.0, 'presence_penalty': 0.0,
    'img_size': -1, 'img_detail': 'low', 'max_pixels': 151200,
    'thinking_mode': 'No override; CosmosReason2 (not Think), server chat-template default; no suffix',
    'chat_template_kwargs': 'omitted', 'response_format': 'omitted',
    'primary_scorer': 'Pinned official evaluation.py extract_answer, exact match; invalid/blank/truncated remain in denominator',
    'diagnostic_scorer': 'Maintained MMSIVideoBench.evaluate(model=exact_matching), no external judge',
    'raw_videos_downloaded': False, 'reference_score_percent': 27.2,
}


def _scope(scope):
    if scope not in ('smoke', 'full'):
        raise ValueError('scope must be smoke or full')


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def _external_root(path):
    root = Path(path).expanduser().resolve()
    source = next((parent for parent in Path(__file__).resolve().parents
                   if (parent / '.git').exists()), Path(__file__).resolve().parents[1])
    if root == source or root.is_relative_to(source):
        raise ValueError('Dataset assets must be outside the source checkout')
    return root


def _member_path(root, name):
    relative = PurePosixPath(name)
    if relative.is_absolute() or '..' in relative.parts or '\\' in name:
        raise ValueError(f'Unsafe media path: {name!r}')
    path = root.joinpath(*relative.parts)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'Media path escapes owned root: {name!r}')
    return path


def _fetch(root, name, url, expected):
    path = root / name
    if path.is_file() and not path.is_symlink() and _sha(path) == expected:
        print(f'MMSI verified cache: {name}', flush=True)
    else:
        partial = root / (name + '.part')
        if partial.is_symlink() or path.is_symlink():
            raise ValueError(f'Refusing symlink download target: {name}')
        received = 0
        last = time.monotonic()
        print(f'MMSI downloading: {name}', flush=True)
        with urllib.request.urlopen(url, timeout=300) as response, partial.open('wb') as output:
            total = response.headers.get('Content-Length', 'unknown')
            while block := response.read(8 << 20):
                output.write(block)
                received += len(block)
                if time.monotonic() - last >= 5:
                    print(f'MMSI {name}: {received} / {total} bytes', flush=True)
                    last = time.monotonic()
        if _sha(partial) != expected:
            raise ValueError(f'SHA256 mismatch: {name}; partial retained for inspection')
        partial.replace(path)
    return {'file': name, 'url': url, 'sha256': expected, 'bytes': path.stat().st_size}


def _extract(root, name):
    # CRC-check cached members too: a marker alone cannot establish media integrity.
    count = 0
    with zipfile.ZipFile(root / name) as archive:
        for info in archive.infolist():
            destination = _member_path(root, info.filename)
            mode = info.external_attr >> 16
            if stat.S_ISLNK(mode) or (stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR)):
                raise ValueError(f'Unsafe archive member: {info.filename!r}')
            if info.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            if destination.is_symlink():
                raise ValueError(f'Refusing symlink extraction target: {info.filename!r}')
            destination.parent.mkdir(parents=True, exist_ok=True)
            # Reading through zipfile validates each pinned archive member's CRC.
            with archive.open(info) as source, destination.open('wb') as output:
                shutil.copyfileobj(source, output, 1 << 20)
            count += 1
            if count % 2000 == 0:
                print(f'MMSI extracting {name}: {count} files', flush=True)
    return {'archive': name, 'files': count}


def download(data_root: Path, scope: str):
    """Download and safely unpack public pinned canonical frames, never videos.zip."""
    _scope(scope)
    started = time.monotonic()
    root = _external_root(data_root)
    root.mkdir(parents=True, exist_ok=True)
    downloads = [_fetch(root, name, *spec) for name, spec in FILES.items()]
    extracted = [_extract(root, name) for name in ('frames.zip', 'ref_images.zip')]
    result = {'benchmark': 'mmsivideo-bench', 'scope': scope, 'status': 'downloaded',
              'data_root': str(root), 'downloads': downloads, 'extracted': extracted,
              'dataset_revision': DATASET_REVISION, 'easi_revision': EASI_REVISION,
              'official_revision': OFFICIAL_REVISION, 'warnings': WARNINGS,
              'raw_videos_downloaded': False, 'elapsed_seconds': time.monotonic() - started}
    _save(root / 'download.json', result)
    return result


def _verify_sources(root):
    for name, (_, expected) in FILES.items():
        # The archive pins establish provenance even when only extracted members are consumed.
        if not (root / name).is_file() or (root / name).is_symlink() or _sha(root / name) != expected:
            raise ValueError(f'Missing or changed pinned asset {name}; run download first')


def _dataset(root):
    import pandas as pd
    from vlmeval.dataset.mmsibench import MMSIVideoBench

    class StagedMMSIVideo(MMSIVideoBench):
        def __init__(self):
            # Do not call VideoBaseDataset.__init__: it creates shared LMU caches
            # and invokes network staging. Only owned paths/data are needed here.
            self.dataset_name = 'MMSIVideoBench'
            self.dataset_path = str(root)
            self.data_root = str(root)
            self.pack = False
            self.nframe = 50
            self.fps = -1
            self.data = pd.read_csv(root / 'MMSIVideoBench.tsv', sep='\t', keep_default_na=False)
            def absolute(value):
                if isinstance(value, list):
                    return [absolute(item) for item in value]
                return str(_member_path(root, value.lstrip('/\\')))
            for column in ('image_path', 'ref_images'):
                self.data[column] = self.data[column].map(lambda value: absolute(ast.literal_eval(value)))

    return StagedMMSIVideo()


def _qualify(dataset, root, scope, work):
    annotations = json.loads((root / 'mmsivideo.json').read_text())
    by_id = {row['id']: row for row in annotations}
    data = dataset.data
    if len(by_id) != 1106 or len(annotations) != 1106 or len(data) != 1106:
        raise ValueError('Expected exactly 1106 unique official questions and metadata rows')
    if data.raw_id.duplicated().any() or data['index'].duplicated().any() or set(data.raw_id) != set(by_id):
        raise ValueError('Metadata identities differ from official annotations')
    if len(set(data.task_type)) != 5:
        raise ValueError('Expected five task categories')
    selected = data if scope == 'full' else data.groupby('task_type', sort=True).head(1)
    manifest = []
    prompts = {}
    for _, row in selected.iterrows():
        original = by_id[row.raw_id]
        if Path(row.raw_id).name != row.raw_id or row.raw_id in ('.', '..'):
            raise ValueError('Unsafe official question ID')
        expected_type = (row.task_type if row.task_type in ('Planning', 'Prediction')
                         else f'({row.task_type}) {row.sub_task_type}')
        if original['type'] != expected_type:
            raise ValueError(f'Official task category mismatch: {row.raw_id}')
        options = ast.literal_eval(row.options)
        if row.answer != original['ground_truth'] or options != original['options']:
            raise ValueError(f'Label/options mismatch: {row.raw_id}')
        if row.answer not in list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')[:len(options)]:
            raise ValueError(f'Invalid ground truth: {row.raw_id}')
        for field in ('system_prompt', 'task_prompt', 'user_prompt', 'format_prompt'):
            if row[field] != original[field]:
                raise ValueError(f'Official prompt mismatch: {row.raw_id} {field}')
        frames = dataset.parse_image_path(row.image_path)
        expected_frames = [[str(_member_path(root, 'frames/' + name)) for name in segment]
                           for segment in original['frames_list']]
        refs = row.ref_images
        expected_refs = [str(_member_path(root, 'ref_images/' + name)) for name in original['ref_images']]
        if frames != expected_frames or refs != expected_refs:
            raise ValueError(f'Canonical media mismatch: {row.raw_id}')
        for value in [item for segment in frames for item in segment] + refs:
            path = Path(value)
            if not path.resolve().is_relative_to(root) or not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f'Missing/unsafe media for {row.raw_id}: {path.name}')
        prompt = dataset.build_prompt(row)
        sampled = dataset.save_video_frames(frames, row.video_fps)
        prompts[row.raw_id] = prompt
        manifest.append({'id': row.raw_id, 'index': int(row['index']), 'task_type': row.task_type,
                         'sub_task_type': row.sub_task_type, 'answer': row.answer,
                         'selected_paths': [[str(Path(p).relative_to(root)) for p in segment] for segment in sampled],
                         'references': [str(Path(p).relative_to(root)) for p in refs],
                         'annotation_frames': sum(map(len, frames)), 'selected_frames': sum(map(len, sampled)),
                         'prompt': prompt,
                         'prompt_text_sha256': hashlib.sha256(''.join(p['value'] for p in prompt if p['type'] == 'text').encode()).hexdigest()})
    _save(work / 'sample_manifest.json', manifest)
    print(f'MMSI qualified {len(selected)} questions; all selected labels/options/prompts/media match official annotations', flush=True)
    return selected, by_id, prompts


def _runtime_provenance():
    root = Path(__file__).resolve().parents[1]
    paths = ['launchable/mmsivideo.py', 'vlmeval/dataset/mmsibench.py',
             'vlmeval/api/cosmos_reason.py', 'vlmeval/dataset/utils/multiple_choice.py',
             'vlmeval/dataset/utils/spatial_bench/cal_scores.py', 'vlmeval/execution_evidence.py']
    return {name: _sha(root / name) for name in paths}


def _score(dataset, selected, records, by_id, root, work, scope, api_nproc):
    from vlmeval.dataset.utils.multiple_choice import build_choices, extract_answer_from_item
    spec = importlib.util.spec_from_file_location('mmsivideo_official_evaluation', root / 'official_evaluation.py')
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    inputs = work / 'official_inputs'
    inputs.mkdir()
    scored = []
    for record in records:
        row = selected.loc[selected.raw_id == record['id']].iloc[0].to_dict()
        row.update(dict(zip('ABCDEFGHIJKLMNOPQRSTUVWXYZ', ast.literal_eval(row['options']))))
        row['prediction'] = record['prediction']
        diagnostic = extract_answer_from_item(None, row, dataset_name='MMSIVideoBench')['opt']
        primary = official.extract_answer(record['prediction'])
        annotation = by_id[record['id']]
        _save(inputs / f"{record['id']}.json", {'id': record['id'], 'response': record['prediction'],
              'ground_truth': annotation['ground_truth'], 'type': annotation['type']})
        scored.append({'id': record['id'], 'answer': row['answer'], 'task_type': row['task_type'],
                       'sub_task_type': row['sub_task_type'], 'official_prediction': primary,
                       'diagnostic_prediction': diagnostic, 'official_hit': int(primary == row['answer']),
                       'diagnostic_hit': int(diagnostic == row['answer']),
                       'official_invalid': primary not in build_choices(row),
                       'diagnostic_invalid': diagnostic not in build_choices(row),
                       'truncated': record['truncated'], 'finish_reason': record['finish_reason']})
    _save(work / 'scored.json', scored)
    frame = selected.copy()
    predictions = {r['id']: r['prediction'] for r in records}
    frame['prediction'] = frame.raw_id.map(predictions)
    frame.to_csv(work / 'predictions.tsv', sep='\t', index=False)
    dataset.data = selected.copy()
    native = dataset.evaluate(str(work / 'predictions.tsv'), model='exact_matching', nproc=api_nproc)
    def reduce(items):
        count = len(items)
        return {'count': count, 'official_correct': sum(r['official_hit'] for r in items),
                'official_accuracy_percent': 100 * sum(r['official_hit'] for r in items) / count,
                'diagnostic_accuracy_percent': 100 * sum(r['diagnostic_hit'] for r in items) / count,
                'official_invalid': sum(r['official_invalid'] for r in items),
                'diagnostic_invalid': sum(r['diagnostic_invalid'] for r in items),
                'truncated': sum(r['truncated'] for r in items)}
    result = reduce(scored)
    if abs(float(native['overall']) - result['diagnostic_accuracy_percent']) > 1e-9:
        raise RuntimeError('Maintained diagnostic scorer disagrees with independent reduction')
    _save(work / 'diagnostic_summary.json', native)
    if scope == 'full':
        completed = subprocess.run([sys.executable, str(root / 'official_evaluation.py'),
                                    '--eval_dir', str(inputs), '--bench', 'main'],
                                   capture_output=True, text=True, cwd=root)
        (work / 'official_scorer.log').write_text(completed.stdout + completed.stderr)
        completed.check_returncode()
        line = next(line for line in completed.stdout.splitlines() if line.startswith('Overall :'))
        accuracy = float(line.split(':', 1)[1].split()[0]) * 100
        if abs(accuracy - result['official_accuracy_percent']) > 1e-9:
            raise RuntimeError('Official CLI disagrees with independent reduction')
        result['official_cli_agrees'] = True
    else:
        (work / 'official_scorer.log').write_text('Smoke: pinned official extract_answer used locally for every row. CLI requires all 13 subcategories; full scope runs unmodified CLI.\n')
    result['per_category'] = {name: reduce([r for r in scored if r['task_type'] == name])
                              for name in sorted({r['task_type'] for r in scored})}
    result['per_subcategory'] = {name: reduce([r for r in scored if r['sub_task_type'] == name])
                                 for name in sorted({r['sub_task_type'] for r in scored})}
    result['scorer_discrepancies'] = sum(r['official_prediction'] != r['diagnostic_prediction'] for r in scored)
    result['diagnostic_scorer_agrees'] = True
    return result


def _successful_transport(code, http_status, wrapper_error):
    """A truncation marker cannot turn an HTTP success into a transport failure."""
    return (200 <= http_status < 300
            and (code == 0 or wrapper_error == 'truncated'))


def run(data_root: Path, scope: str, work: Path, api_nproc: int = 4):
    """Evaluate a fresh work directory, retaining every response and generation issue."""
    _scope(scope)
    if not isinstance(api_nproc, int) or isinstance(api_nproc, bool) or not 1 <= api_nproc <= 4:
        raise ValueError('api_nproc must be an integer from 1 through 4')
    root = _external_root(data_root)
    work = _external_root(work)
    # A clean run has no prediction reuse or selective generation retries.
    work.mkdir(parents=True, exist_ok=False, mode=0o700)
    started = time.monotonic()
    records = []
    from vlmeval.execution_evidence import aggregate_execution_records, nano_processor_profile, PINNED_SERVING_COMPATIBILITY
    expected = 5 if scope == 'smoke' else 1106
    def progress(message):
        print(message, flush=True)
        with (work / 'run.log').open('a') as log:
            log.write(message + '\n')
    try:
        _verify_sources(root)
        dataset = _dataset(root)
        selected, by_id, prompts = _qualify(dataset, root, scope, work)
        base = os.environ.get('COSMOS_API_BASE', 'http://cosmos-endpoint:8080/v1/chat/completions')
        parsed = urlsplit(base)
        if parsed.scheme not in ('http', 'https') or not parsed.netloc or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError('Use a plain HTTP(S) endpoint without URL credentials, query or fragment; supply COSMOS_API_KEY separately')
        if not parsed.path.endswith('/chat/completions'):
            raise ValueError('COSMOS_API_BASE must be the complete /v1/chat/completions URL')
        model_name = os.environ.get('COSMOS_MODEL', 'nvidia/Cosmos3-Nano')
        provenance = {'benchmark': 'mmsivideo-bench', 'scope': scope,
                      'dataset_revision': DATASET_REVISION, 'easi_revision': EASI_REVISION,
                      'official_revision': OFFICIAL_REVISION,
                      'assets': {name: {'url': url, 'sha256': digest} for name, (url, digest) in FILES.items()},
                      'model': model_name, 'model_revision': os.environ.get('COSMOS_MODEL_REVISION', 'unknown'),
                      'endpoint': base, 'protocol': dict(PROTOCOL, api_nproc=api_nproc),
                      'runtime_source_sha256': _runtime_provenance(), 'warnings': WARNINGS,
                      'execution_evidence': {'summary': 'execution-summary.json', 'per_response': 'execution-responses'},
                      'pinned_runtime_compatibility': PINNED_SERVING_COMPATIBILITY,
                      'reference_parity': False, 'started_unix': time.time()}
        _save(work / 'provenance.json', provenance)
        # Import/instantiate only after all selected samples have been qualified.
        from vlmeval.api.cosmos_reason import CosmosReason2
        local = threading.local()
        def infer(row):
            sample_started = time.monotonic()
            record = {'id': row.raw_id, 'index': int(row['index'])}
            try:
                if not hasattr(local, 'model'):
                    local.model = CosmosReason2(model=model_name, api_base=base,
                        key=os.environ.get('COSMOS_API_KEY', 'EMPTY'), temperature=0, top_p=1.0,
                        top_k=-1, repetition_penalty=1.0, presence_penalty=0.0,
                        max_tokens=2048, seed=1, timeout=600, retry=1, wait=0,
                        img_size=-1, img_detail='low',
                        execution_evidence_dir=str(work / 'execution-responses'),
                        execution_processor_profile=nano_processor_profile(
                            model_name, os.environ.get('COSMOS_MODEL_REVISION', 'unknown')),
                        extra_body={'mm_processor_kwargs': {'max_pixels': 151200}})
                if local.model.model != model_name:
                    raise RuntimeError('Endpoint model differs from COSMOS_MODEL; refusing implicit remapping')
                code, prediction, response = local.model.generate_inner(prompts[row.raw_id])
                raw = response.json() if response is not None else None
                record.update(status_code=code, prediction=prediction, response=raw,
                              wrapper_error=local.model._last_error_type, served_model=local.model.model)
                record['execution_evidence'] = local.model._last_execution_record
                # Preserve the actual wire response as well as wrapper-extracted text.
                if response is not None:
                    record['raw_response_text'] = response.text
                transport_ok = (response is not None and _successful_transport(
                    code, response.status_code, local.model._last_error_type))
                if not transport_ok or not isinstance(raw, dict) or not raw.get('choices'):
                    raise RuntimeError('Transport failure or malformed completion response')
                choice = raw['choices'][0]
                if not isinstance(prediction, str) or not isinstance(choice.get('message', {}).get('content'), str):
                    raise RuntimeError('Completion missing string content')
                finish = choice.get('finish_reason')
                if not isinstance(finish, str):
                    raise RuntimeError('Completion missing finish reason')
                record.update(finish_reason=finish, truncated=finish != 'stop')
                if local.model._last_error_type not in (None, 'truncated'):
                    raise RuntimeError('Endpoint generation failed')
            except Exception as error:
                # Exception strings can contain authentication URLs/headers. Never persist them.
                record['failure'] = type(error).__name__
            record['elapsed_seconds'] = time.monotonic() - sample_started
            _save(work / f"{row.raw_id}.json", record)
            return record
        with ThreadPoolExecutor(max_workers=api_nproc) as pool:
            futures = [pool.submit(infer, row) for _, row in selected.iterrows()]
            for future in as_completed(futures):
                record = future.result()
                records.append(record)
                progress(f'MMSI {len(records)}/{expected}: {record["id"]} ' +
                         ('FAILED' if record.get('failure') else str(record['finish_reason'])))
        records.sort(key=lambda record: record['index'])
        _save(work / 'responses.json', records)
        evidence = aggregate_execution_records(
            [record.get('execution_evidence') or {} for record in records], expected_requests=expected)
        evidence.update(summary='execution-summary.json', per_response='execution-responses')
        _save(work / 'execution-summary.json', evidence)
        provenance['execution_evidence'] = evidence
        _save(work / 'provenance.json', provenance)
        if len(records) != expected or any(r.get('failure') for r in records):
            raise RuntimeError('Incomplete coverage/transport failure; inspect durable per-sample records; no score published')
        metrics = _score(dataset, selected, records, by_id, root, work, scope, api_nproc)
        truncated = sum(r['truncated'] for r in records)
        warnings = list(WARNINGS)
        if truncated:
            warnings.append(f'{truncated} non-stop/truncated completions retained and scored without regeneration.')
        if metrics['official_invalid']:
            warnings.append(f'{metrics["official_invalid"]} invalid official extractions retained as incorrect.')
        summary = {'benchmark': 'mmsivideo-bench', 'scope': scope,
                   'status': ('completed_with_generation_issues'
                              if truncated or metrics['official_invalid'] else 'completed'),
                   'primary_metric': {'name': 'official_accuracy', 'value': metrics['official_accuracy_percent'], 'unit': 'percent'},
                   'coverage': {'expected': expected, 'scored': len(records)},
                   'protocol': provenance['protocol'], 'warnings': warnings, 'reference_parity': False,
                   'elapsed_seconds': time.monotonic() - started, 'metrics': metrics,
                   'execution_evidence': evidence,
                   'finish_reasons': dict(Counter(r['finish_reason'] for r in records)),
                   'model': model_name, 'model_revision': provenance['model_revision']}
        _save(work / 'summary.json', summary)
        provenance['elapsed_seconds'] = summary['elapsed_seconds']
        provenance['finished_unix'] = time.time()
        _save(work / 'provenance.json', provenance)
        progress(f'MMSI {summary["status"]}: official accuracy {metrics["official_accuracy_percent"]:.4f}%')
        return summary
    except Exception as error:
        _save(work / 'failure.json', {'benchmark': 'mmsivideo-bench', 'scope': scope, 'status': 'failed',
              'error_type': type(error).__name__, 'expected': expected, 'responses_retained': len(records),
              'elapsed_seconds': time.monotonic() - started, 'score_published': False})
        progress('MMSI failed; inspect failure.json and retained evidence. No completed score published.')
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('download', 'run'))
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--scope', choices=('smoke', 'full'), default='smoke')
    parser.add_argument('--work', type=Path)
    parser.add_argument('--api-nproc', type=int, default=4)
    args = parser.parse_args()
    if args.action == 'download':
        result = download(args.data_root, args.scope)
    else:
        if args.work is None:
            parser.error('--work is required for run')
        result = run(args.data_root, args.scope, args.work, args.api_nproc)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
