# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public Perception Test validation inputs and offline MCQ top-1 scoring.

The metric is official option-ID accuracy. The conservative text-to-option
adapter below is a launchable protocol, not an official letter-parser claim.
Images use a prototype 16-frame/aspect-preserving 768-maxside protocol, not a
published Cosmos3-Nano score recipe. Labels never enter inference prompts.
"""
import csv
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
import urllib.request
import warnings
import zipfile

DATASET_REPO = 'https://github.com/google-deepmind/perception_test'
DATASET_REVISION = '3938d2f1ba3a6b502025741cea4cd73c7b3bdfaf'
EXPECTED_ROWS = 19140
EXPECTED_VIDEOS = 5260
SMOKE_VIDEO = 'video_8241'
SMOKE_INDICES = [13404, 13405, 13406, 13407, 13408]
MAX_FRAME_SIDE = 768
PROTOCOL = 'perception-test-local-images-v1'
FRAME_PREPROCESSING = 'aspect-preserving downscale only; no crop'
SAMPLING = 'uniform segment midpoints in [0, exclusive_stop); duplicate samples allowed for short clips'
RIGHTS_NOTICE = (
    'Perception Test by DeepMind Technologies Limited (2022), Patraucean et al. (NeurIPS 2023). '
    'Software: Apache-2.0; other materials: CC BY 4.0 '
    '(https://creativecommons.org/licenses/by/4.0/). Retain attribution and consult the pinned README.md. '
    'https://github.com/google-deepmind/perception_test'
)
METADATA_SHA256 = {
    'mc_question_valid.json': '174d3fa7ffa0958ae1d61311f85747dd5d0a9ae3d1e21c504dd4a47e723f78c3',
    'cut_frame_mapping_valid.json': '3cd9ae2d56f28ba70fa80ccca04aca6481ebab1bf1099f3aabe92afad89a67f7',
    'README.md': '9f739a07616e3f60af0723800d813ca87953496ba9e288fc62b4bfd7578afa85',
}
PUBLIC_ASSETS = {
    'downloads/mc_question_valid_annotations.zip': {
        'object': 'zip_data/mc_question_valid_annotations.zip', 'generation': '1686575054970958',
        'bytes': 204076, 'sha256': '3f3ebb388b042e35d5c07fa6e2404d78feb63e4e477740948fc92dfc83c480ad'},
    'cut_frame_mapping_valid.json': {
        'object': 'misc/cut_frame_mapping_valid.json', 'generation': '1692694363800279',
        'bytes': 106874, 'sha256': METADATA_SHA256['cut_frame_mapping_valid.json']},
    'README.md': {
        'url': f'https://raw.githubusercontent.com/google-deepmind/perception_test/{DATASET_REVISION}/README.md',
        'bytes': 30386, 'sha256': METADATA_SHA256['README.md']},
    'downloads/sample_videos.zip': {
        'object': 'zip_data/sample_videos.zip', 'generation': '1685097626197809',
        'bytes': 225370639, 'sha256': '68e5cca8c3064859f273ec7a52c15d5eb4a337032d12e6d2f379079825a72537'},
    'downloads/valid_videos.zip': {
        'object': 'zip_data/valid_videos.zip', 'generation': '1685479088600941',
        'bytes': 75375143328, 'sha256': '553c06c6f525af4d8ebb0c31b2868e21ed517ffa239f43aef3b82ed3793cd377'},
}
_MEDIA_HASH_CACHE = {}


class PerceptionTestCoverageError(ValueError):
    """Selected media are incomplete; report contains every missing video."""

    def __init__(self, message, report):
        super().__init__(message)
        self.report = report


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    os.replace(temporary, path)


def _inside(root, relative):
    name = str(relative)
    parts = PurePosixPath(name)
    if (not name or '\\' in name or '\x00' in name or ':' in name
            or parts.is_absolute() or '..' in parts.parts):
        raise ValueError(f'Unsafe relative path: {name!r}')
    root = Path(root).resolve()
    target = root.joinpath(*parts.parts)
    current = target
    while current != root:
        if current.is_symlink():
            raise ValueError(f'Refusing symlink dataset path: {current}')
        current = current.parent
    if not target.resolve().is_relative_to(root):
        raise ValueError(f'Path escapes dataset root: {name!r}')
    return target


def _safe_extract(archive, destination, allowed=None):
    """Validate every ZIP member before extracting regular files atomically."""
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        members, seen = [], set()
        for info in source.infolist():
            target = _inside(destination, info.filename)
            mode = info.external_attr >> 16
            if stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR):
                raise ValueError(f'Unsupported ZIP member type: {info.filename!r}')
            name = PurePosixPath(info.filename).as_posix().rstrip('/')
            if name in seen:
                raise ValueError(f'Duplicate ZIP member: {name!r}')
            seen.add(name)
            if allowed is not None and not allowed(name, info.is_dir()):
                raise ValueError(f'Unexpected ZIP member: {name!r}')
            members.append((info, target))
        for info, target in members:
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open(info) as reader, tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as writer:
                temporary = Path(writer.name)
                try:
                    shutil.copyfileobj(reader, writer, length=8 * 1024 * 1024)
                except BaseException:
                    temporary.unlink(missing_ok=True)
                    raise
            os.replace(temporary, target)


def _download(path, pin):
    if path.exists():
        if path.stat().st_size != pin['bytes'] or _sha256(path) != pin['sha256']:
            raise ValueError(f'Pinned file checksum/size mismatch: {path}; remove damaged file and fetch again')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    url = pin.get('url') or ('https://storage.googleapis.com/dm-perception-test/'
                             + pin['object'] + '?generation=' + pin['generation'])
    temporary = None
    try:
        with urllib.request.urlopen(url, timeout=180) as reader, tempfile.NamedTemporaryFile(
                dir=path.parent, delete=False) as writer:
            temporary = Path(writer.name)
            generation = reader.headers.get('x-goog-generation')
            if pin.get('generation') and generation != pin['generation']:
                raise ValueError(f'Public object generation mismatch: {url}')
            digest, count = hashlib.sha256(), 0
            while block := reader.read(8 * 1024 * 1024):
                writer.write(block)
                digest.update(block)
                count += len(block)
            if count != pin['bytes'] or digest.hexdigest() != pin['sha256']:
                raise ValueError(f'Public download failed pinned SHA-256/size verification: {url}')
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def fetch_dataset(root, scope='smoke'):
    """Fetch generation/SHA-pinned public metadata, smoke sample, or full videos.

    No authentication, mirrors, internal hosts, or private dataset prerequisites.
    Full requires sufficient storage for both the 75.4GB ZIP and extracted media.
    """
    if scope not in ('metadata', 'smoke', 'full'):
        raise ValueError('scope must be metadata, smoke, or full')
    root = Path(root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    warnings.warn(RIGHTS_NOTICE, UserWarning, stacklevel=2)
    requested = ['downloads/mc_question_valid_annotations.zip', 'cut_frame_mapping_valid.json', 'README.md']
    if scope != 'metadata':
        requested.append('downloads/sample_videos.zip' if scope == 'smoke' else 'downloads/valid_videos.zip')
    for name in requested:
        pin = PUBLIC_ASSETS[name]
        path = _inside(root, name)
        _download(path, pin)
        if name.endswith('_annotations.zip'):
            _safe_extract(path, root, lambda n, directory: n == 'mc_question_valid.json' and not directory)
        elif name.endswith('_videos.zip'):
            _safe_extract(path, root, lambda n, directory: (directory and n == 'videos') or
                          (not directory and re.fullmatch(r'videos/video_\d+\.mp4', n) is not None))
    _read_annotations(root)
    notice = root / '.perception-test' / 'RIGHTS_NOTICE.txt'
    notice.parent.mkdir(parents=True, exist_ok=True)
    notice.write_text(RIGHTS_NOTICE + '\n', encoding='utf-8')
    return {'root': str(root), 'dataset_revision': DATASET_REVISION, 'scope': scope,
            'downloaded_files': requested, 'rights_notice': RIGHTS_NOTICE,
            'public_assets': {name: PUBLIC_ASSETS[name] for name in requested}}


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}: {value!r}')
    return value


def _read_annotations(root):
    """Authenticate all label, option, identity, cutoff and rights source bytes."""
    for name, expected in METADATA_SHA256.items():
        path = _inside(root, name)
        if not path.is_file():
            raise ValueError(f'Missing pinned metadata {name}; run fetch_dataset(root, scope="metadata")')
        if _sha256(path) != expected:
            raise ValueError(f'Pinned metadata checksum mismatch: {name}; annotations are not trusted')
    annotations = json.loads((root / 'mc_question_valid.json').read_text(encoding='utf-8'))
    cuts = json.loads((root / 'cut_frame_mapping_valid.json').read_text(encoding='utf-8'))
    if len(annotations) != EXPECTED_VIDEOS or any(re.fullmatch(r'video_\d+', v) is None for v in annotations):
        raise ValueError(f'Expected exactly {EXPECTED_VIDEOS} canonical video identities')
    rows = []
    for video in sorted(annotations, key=lambda v: int(v.split('_')[1])):
        source = annotations[video]
        meta = source['metadata']
        num_frames = _integer(meta['num_frames'], f'{video} num_frames', 1)
        if meta['video_id'] != video or meta['split'] != 'valid':
            raise ValueError(f'Metadata identity/split mismatch: {video}')
        cut = cuts.get(video)
        if isinstance(cut, bool) or not isinstance(cut, int) or (cut != -1 and not 0 < cut <= num_frames):
            raise ValueError(f'Missing or invalid exclusive frame cut: {video}: {cut!r}')
        questions = sorted(source['mc_question'], key=lambda q: q['id'])
        if [q['id'] for q in questions] != list(range(len(questions))) or not questions:
            raise ValueError(f'Invalid or duplicate question identities: {video}')
        for question in questions:
            _integer(question['id'], f'{video} question ID')
            options = question['options']
            answer = _integer(question['answer_id'], 'answer_id')
            if (not isinstance(options, list) or len(options) != 3
                    or not all(isinstance(o, str) and o.strip() for o in options) or answer >= 3):
                raise ValueError(f'Invalid official MCQ options/answer ID: {video}/{question["id"]}')
            tags = question['tag']
            if (not isinstance(tags, list) or not all(isinstance(t, str) and t for t in tags)
                    or len(tags) != len(set(tags))):
                raise ValueError(f'Invalid diagnostic tags: {video}/{question["id"]}')
            for field in ('question', 'area', 'reasoning'):
                if not isinstance(question[field], str) or not question[field].strip():
                    raise ValueError(f'Invalid {field}: {video}/{question["id"]}')
            rows.append({'index': len(rows), 'video_id': video, 'question_id': question['id'],
                         'video': f'videos/{video}.mp4', 'question': question['question'],
                         'options': json.dumps(options, ensure_ascii=False), 'answer_id': answer,
                         'answer': chr(65 + answer), 'A': options[0], 'B': options[1], 'C': options[2],
                         'area': question['area'], 'reasoning': question['reasoning'],
                         'tag': json.dumps(tags, ensure_ascii=False), 'cut_frame': cut, 'num_frames': num_frames})
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f'Expected exactly {EXPECTED_ROWS} official questions, found {len(rows)}')
    return rows


def _selected(rows, smoke):
    selected = [r for r in rows if r['video_id'] == SMOKE_VIDEO and r['question_id'] in range(5)] if smoke else rows
    if smoke and [r['index'] for r in selected] != SMOKE_INDICES:
        raise ValueError('Pinned smoke identities do not match canonical stable indices')
    return selected


def _media_content_sha256(root, row):
    path = _inside(root, row['video'])
    def signature():
        info = path.stat()
        return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns
    before = signature()
    cached = _MEDIA_HASH_CACHE.get(str(path))
    if cached is not None and cached[0] == before:
        return cached[1]
    value = _sha256(path)
    if signature() != before:
        raise ValueError(f'Media changed while hashing: {path}')
    _MEDIA_HASH_CACHE[str(path)] = before, value
    return value


def prepare_dataset(root, smoke=False, require_media=True):
    """Rebuild trusted TSV; metadata/scope identity never depends on media.

    Media-free scoring skips content hashing; no unobserved media identity is
    inferred. Coverage counts present nonempty files, not decoder validation.
    """
    root = Path(root).expanduser().resolve()
    rows = _read_annotations(root)
    selected = _selected(rows, smoke)
    available = {}
    for row in rows:
        if row['video'] not in available:
            path = _inside(root, row['video'])
            available[row['video']] = path.is_file() and path.stat().st_size > 0
    missing = sorted(video for video, present in available.items() if not present)
    coverage = {'expected_rows': EXPECTED_ROWS, 'expected_videos': EXPECTED_VIDEOS,
                'available_rows': sum(available[r['video']] for r in rows),
                'available_videos': sum(available.values()), 'selected_rows': len(selected),
                'selected_videos': len({r['video_id'] for r in selected}),
                'selected_available_rows': sum(available[r['video']] for r in selected), 'subset': bool(smoke)}
    scope = 'smoke' if smoke else 'full'
    annotation_manifest = _fingerprint({'repo': DATASET_REPO, 'revision': DATASET_REVISION,
                                        'metadata_sha256': METADATA_SHA256, 'canonical_rows': rows})
    manifest = _fingerprint({'annotation_manifest_sha256': annotation_manifest,
                             'scope': scope, 'selected_indices': [r['index'] for r in selected]})
    report = {'dataset_revision': DATASET_REVISION, 'coverage': coverage, 'missing_media': missing,
              'manifest_sha256': manifest, 'rights_notice': RIGHTS_NOTICE}
    state = root / '.perception-test'
    _json_write(state / f'coverage-{scope}.json', report)
    if require_media and any(not available[r['video']] for r in selected):
        raise PerceptionTestCoverageError(
            f'Perception Test media incomplete: {coverage["selected_available_rows"]}/{len(selected)} selected '
            f'questions have media. Full requires all {EXPECTED_ROWS} questions/{EXPECTED_VIDEOS} videos; '
            f'run fetch_dataset(root, scope="{scope}"). See {state / ("coverage-" + scope + ".json")}', report)
    media = {}
    if require_media:
        for row in selected:
            if row['video'] not in media:
                media[row['video']] = _media_content_sha256(root, row)
    media_manifest = _fingerprint(media) if require_media else None
    data_file = state / f'PerceptionTest-{scope}-{manifest[:16]}.tsv'
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=state, delete=False, newline='') as stream:
        temporary = Path(stream.name)
        writer = csv.DictWriter(stream, fieldnames=list(selected[0]), delimiter='\t')
        writer.writeheader()
        writer.writerows(selected)
    os.replace(temporary, data_file)
    if require_media:
        _json_write(state / f'media-manifest-{scope}.json', {'sha256': media_manifest, 'media': media})
    return {'root': str(root), 'data_file': str(data_file), 'dataset_revision': DATASET_REVISION,
            'manifest_sha256': manifest, 'annotation_manifest_sha256': annotation_manifest,
            'coverage': coverage, 'selected_indices': [r['index'] for r in selected],
            'media_sha256': media, 'media_manifest_sha256': media_manifest}


def build_dataset(root, nframe=16, smoke=False):
    from vlmeval.dataset.perception_test import PerceptionTestLocal
    return PerceptionTestLocal(data_root=str(root), nframe=nframe, smoke=smoke)


def _load_predictions(path):
    import pandas as pd
    path = Path(path)
    if path.suffix.lower() == '.tsv':
        return pd.read_csv(path, sep='\t', keep_default_na=False, dtype={'prediction': str})
    if path.suffix.lower() == '.xlsx':
        return pd.read_excel(path, keep_default_na=False, dtype={'prediction': str})
    if path.suffix.lower() == '.json':
        return pd.DataFrame(json.loads(path.read_text(encoding='utf-8')))
    raise ValueError('Predictions must be TSV, XLSX or JSON; eval and pickle inputs are not accepted')


def _verified_predictions(incoming, trusted):
    if not {'index', 'prediction'}.issubset(incoming.columns) or incoming.columns.duplicated().any():
        raise ValueError('Predictions require unique index and prediction columns')
    indices = []
    for value in incoming['index']:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            raise ValueError(f'Invalid prediction identity: {value!r}') from None
        if isinstance(value, bool) or not math.isfinite(numeric) or numeric != int(numeric):
            raise ValueError(f'Invalid prediction identity: {value!r}')
        indices.append(int(numeric))
    expected = [r['index'] for r in trusted]
    if len(indices) != len(set(indices)) or set(indices) != set(expected):
        raise ValueError('Prediction coverage must match selected stable indices exactly: no missing, duplicate, or extra rows')
    incoming = incoming.copy()
    incoming['index'] = indices
    incoming = incoming.set_index('index').loc[expected]
    predictions = []
    for row in trusted:
        supplied = incoming.loc[row['index']]
        for column, wanted in row.items():
            if column == 'index' or column not in incoming.columns:
                continue
            actual = supplied[column]
            if column in ('options', 'tag'):
                try:
                    actual = json.loads(actual) if isinstance(actual, str) else actual
                except (TypeError, ValueError):
                    raise ValueError(f'Untrusted {column} for index {row["index"]}') from None
                wanted = json.loads(wanted)
            if actual != wanted:
                raise ValueError(f'Untrusted {column} for index {row["index"]}')
        prediction = supplied['prediction']
        if (not isinstance(prediction, str) or not prediction.strip()
                or any(marker in prediction.lower() for marker in (
                    'failed to obtain answer', 'video_load_failed', 'image_load_failed',
                    'transport_error', 'api_error'))):
            raise ValueError(f'Missing or failed prediction for index {row["index"]}; inference must complete before scoring')
        predictions.append(prediction)
    return predictions


def parse_choice(prediction, options):
    """Conservative adapter: exact option text or one A/B/C choice expression.

    A/B/C, (A), A., option A, answer: A, and a leading A followed by its exact
    option text are accepted. Explanations may follow an explicit answer marker,
    but any conflicting standalone choice letter makes the output ambiguous.
    Duplicate option text is never resolved to an arbitrary option ID.
    """
    text = prediction.strip()
    if re.fullmatch(r'(?:[ABC]|\([ABC]\))[.!]?', text):
        return ord(re.search(r'[ABC]', text).group()) - 65
    exact = [i for i, option in enumerate(options) if text.casefold() == option.strip().casefold()]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        return None
    standalone = set(re.findall(r'(?<!\w)([ABC])(?!\w)', text))
    if len(standalone) != 1:
        return None
    choice = next(iter(standalone))
    if re.match(r'(?i)^(?:(?:the\s+)?(?:correct\s+)?(?:answer|option|choice)(?:\s+is)?\s*[:=]?\s*)(?:[ABC]|\([ABC]\))(?:[\s.,:]|$)', text):
        return ord(choice) - 65
    index = ord(choice) - 65
    if re.fullmatch(r'(?:\(' + choice + r'\)|' + choice + r'[.):]?)\s+' + re.escape(options[index]), text):
        return index
    return None


def _counts(correct, total):
    return {'correct': correct, 'total': total, 'accuracy': correct / total}


def score_predictions(predictions, root, output_dir, smoke=False, nframe=16):
    """Score authenticated selected identities without video assets, GPU or judge.

    Missing/blank/transport failures reject a run. Nonempty malformed/ambiguous
    choices are wrong and counted invalid. All selected rows remain denominators.
    A fresh output directory prevents stale score reuse or accidental overwrite.
    """
    _integer(nframe, 'nframe', 1)
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists():
        raise ValueError('Scoring requires a fresh output directory')
    prepared = prepare_dataset(root, smoke=smoke, require_media=False)
    trusted = _selected(_read_annotations(Path(prepared['root'])), smoke)
    texts = _verified_predictions(_load_predictions(predictions), trusted)
    groups = {'per_area': {}, 'per_reasoning': {}, 'per_tag': {}}
    scored, correct, invalid = [], 0, 0
    for row, text in zip(trusted, texts):
        selected = parse_choice(text, json.loads(row['options']))
        hit = int(selected == row['answer_id'])
        correct += hit
        invalid += int(selected is None)
        scored.append({**row, 'prediction': text, 'selected_option_id': selected,
                       'valid_choice': selected is not None, 'score': hit})
        for group, labels in (('per_area', [row['area']]), ('per_reasoning', [row['reasoning']]),
                              ('per_tag', json.loads(row['tag']))):
            for label in labels:
                counts = groups[group].setdefault(label, [0, 0])
                counts[0] += hit
                counts[1] += 1
    total = len(trusted)
    summary = {
        'status': 'smoke_only' if smoke else 'complete', 'dataset': 'PerceptionTest',
        'coverage': prepared['coverage'],
        'primary_metric': {'name': 'top1', 'value': correct / total, 'unit': 'fraction',
                           'correct': correct, 'total': total},
        **{group: {label: _counts(*counts) for label, counts in sorted(values.items())}
           for group, values in groups.items()},
        'invalid_count': invalid, 'diagnostic_note': 'Tag groups overlap; their denominators must not be summed.',
        'provenance': {
            'dataset_repo': DATASET_REPO, 'dataset_revision': DATASET_REVISION,
            'manifest_sha256': prepared['manifest_sha256'],
            'annotation_manifest_sha256': prepared['annotation_manifest_sha256'],
            'selected_indices': prepared['selected_indices'],
            'metadata_sha256': dict(METADATA_SHA256), 'prediction_sha256': _sha256(predictions),
            'media_sha256': {}, 'media_manifest_sha256': None,
            'media_note': 'Offline scorer does not authenticate inference media; retain inference preparation provenance.',
            'scorer': 'local official option-ID top1; conservative launchable text adapter',
            'scorer_sha256': _sha256(__file__), 'network_judge': False,
            'protocol': PROTOCOL, 'nframe': nframe, 'max_frame_side': MAX_FRAME_SIDE,
            'frame_preprocessing': FRAME_PREPROCESSING, 'sampling': SAMPLING,
            'cut_policy': '-1 full clip; positive cut is exclusive source-frame stop; never clamp',
            'prompt_mode': 'sampled images only', 'paper_parity': False, 'rights_notice': RIGHTS_NOTICE,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / 'scored_predictions.tsv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(scored[0]), delimiter='\t')
        writer.writeheader()
        writer.writerows(scored)
    _json_write(output_dir / 'summary.json', summary)
    return summary
