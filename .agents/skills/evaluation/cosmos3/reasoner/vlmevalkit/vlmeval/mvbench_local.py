# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pinned, local MVBench inputs and native, network-free exact-match scoring.

Video rights are not conferred by the dataset card's MIT metadata license.
The source card restricts videos to academic research and requires respecting
upstream agreements. NTU RGB+D acquisition is manual, never an old-revision or
mirror fallback. This module deliberately does not claim paper-score parity.
"""
import ast
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import warnings
import zipfile

DATASET_REPO = 'OpenGVLab/MVBench'
DATASET_REVISION = '230a2d4fac8900333c61754641c7a13e069ac9c6'
EXPECTED_ROWS = 4000
ROWS_PER_TASK = 200
FRAME_RESOLUTION = 224
FRAME_PREPROCESSING = 'GroupScale(224,bicubic)+GroupCenterCrop(224)+Stack(roll=False)+ToTorchFormatTensor(div=True)+JPEG'
SMOKE_TASK = 'Egocentric Navigation'
SMOKE_ROWS = 5
RIGHTS_NOTICE = (
    'MVBench video copyrights belong to their original creators/platforms; '
    'the dataset card permits academic research use only and requires compliance '
    'with all source agreements. Review the downloaded README.md before use. '
    'The card describes 320 NTU RGB+D videos; this pinned revision lists 200 '
    'required RGB AVI files for Fine-grained Pose. Obtain authorized access at '
    'https://rose1.ntu.edu.sg/dataset/actionRecognition/ and use the exact pinned '
    'video/MVBench_videos_ntu.txt list. Place authorized media in '
    '<dataset-root>/video/nturgbd. No removed media are downloaded.'
)
# SHA-256 of the exact public bytes at DATASET_REVISION (not a user manifest).
METADATA_SHA256 = {
    'json/action_antonym.json': '0ebbec1f540f598a82bf27f569e7377fdc5c4b6e62d53d3847a88abb785f5513',
    'json/action_count.json': '5e3389d3f8a81744096825d735d45d04b1842fb49da7f5b9a70d4c2224505117',
    'json/action_localization.json': '164290bb5f447b6cdff26eb51a2470bcf4ce31f84d092ec4337f7bd4ce558eec',
    'json/action_prediction.json': '15601d39ca8604dbb642149c097f416518f4816b0c0bec57a6202d95f8bdac79',
    'json/action_sequence.json': 'd5f8c063a5ae5a433c3fb8ff78a03acbf0513a01928b384ed90a1e044adf230e',
    'json/character_order.json': '72c20ad5749da31759c92f4ed83ed117a52f70963ce1d85d5dfd1582accdead4',
    'json/counterfactual_inference.json': '6fabbbaba68b10b48bfb714c51b87569266bc65303530e5ab9c39cab2924bdfa',
    'json/egocentric_navigation.json': '4a5f80cca3471a15efe6d7d4ded192218871992cab32fa3a3eede93ea7fb3972',
    'json/episodic_reasoning.json': '39ca717f721c274a20835835e4ca417f58d5fb2d17ffebb7042d373561c871c3',
    'json/fine_grained_action.json': '21fb61328ab02e399170a22a2a4080a05dff18e8423de709e49b9fa2180fde58',
    'json/fine_grained_pose.json': 'f0cebc0a498474c03618aac20f67810cc41197ac97534a51f930c217dc13f091',
    'json/moving_attribute.json': 'b760214e4320b2ceed4580cc0febfc8c8b084487c7efcaef712b704adb850548',
    'json/moving_count.json': 'a561538a69b2722c96d2b608af488c04db6a0d803b03b49661b8834e9425d245',
    'json/moving_direction.json': '81379646d91fb078a8b38de897f40977eeeebd946ca1c6ef0f58e4bde9d21209',
    'json/object_existence.json': '2bb290258243c7dafe57042768d491e6a239d30d0ff74d0f163a989c60f69620',
    'json/object_interaction.json': '9624cf2fef5ad4ec882edf955ce1d10c806e71e4e8c6a880fd867b224914302b',
    'json/object_shuffle.json': 'f9e1bd17a9bbab1d9f5be71411ca619cf9743ca47bb808985b202138b7928118',
    'json/scene_transition.json': 'edab8a7992851061cac170f7e50782961ec21826f99e70ee72ae73a9dafb79f9',
    'json/state_change.json': '21fa00a1e4fdbd3a0fda4d04f1b4dfd8ee62a1e3f3663fdb0f6e40799c36027f',
    'json/unexpected_action.json': '76b6f440ccb1ed007c67236c0ec7a39ee7cfbfb04197e9cc4a303b9230ca958b',
    'README.md': '0e9520eb913e0825e8cff14eb7a9821a23587312ed2ee97cb86a4b2c2335ec99',
    'video/MVBench_videos_ntu.txt': '878766d10973e1c9e030fca19abb4417f2a8182803199724ef6a9b0b64ec906e',
}
ARCHIVE_SHA256 = {
    'video/FunQA_test.zip': 'd6ef954652c2a1b48fb772d52c97da35dc7f8b54e5bdf4b43ba4f773ed726723',
    'video/Moments_in_Time_Raw.zip': '16add0d03cbb7c6b76974551879f7dc8a7ce985b89699a2231136b81d3ba9967',
    'video/clevrer.zip': '9040ad663440611bd8580fc0fc3da31bd1a4bfef877631db7b570a7d7d41de1c',
    'video/data0613.zip': 'f045863b2ccf7f32ada437d72529fea1c026fc015a7df548f703c22b71a24962',
    'video/perception.zip': '98fc197a90c875fb24f34fe89cb91e2788e1e7ac2459bf860ed62d9f07104b34',
    'video/scene_qa.zip': '26b3dc385e5a8f7b8fd9aab32a3522488fdfe7bbe06f0d98206d43647c77cce9',
    'video/ssv2_video.zip': '8493fe3762bdc205b587c4d02e016f9ccbb314b523ed0bf2c5bdbe367456d1e9',
    'video/sta.zip': '4c27cbf62bbc1b0217b5183871f298e7a76b2a498dd4482365c617f25338de9a',
    'video/star.zip': 'bdb5f192a98f78e79e51da6ae801f8ba79d1d09c0d8c19b6e63105251cb17585',
    'video/tvqa.zip': '6c5bccf237164286123a7afcc1bd7e8323487d7f2fd0dc5f43609a34064f5d73',
    'video/vlnqa.zip': '578f9edf2973560ac0d79990d9a2961ccae8edcdb1cb0be2c3bf15045f04428a',
}
NATIVE_SCORER = 'VLMEvalKit MVBench.evaluate(model=exact_matching), vendored utils/mvbench.py'


class MVBenchCoverageError(ValueError):
    """Incomplete media coverage; ``report`` gives exact missing paths by task."""

    def __init__(self, message, report):
        super().__init__(message)
        self.report = report


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic replacement avoids half-written canonical inputs when interrupted.
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
    os.replace(temporary, path)


def _inside(root, relative):
    name = str(relative)
    parts = PurePosixPath(name)
    if not name or '\\' in name or '\x00' in name or ':' in name or parts.is_absolute() or '..' in parts.parts:
        raise ValueError(f'Unsafe relative path: {name!r}')
    target = Path(root).joinpath(*parts.parts)
    if not target.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError(f'Path escapes dataset root: {name!r}')
    return target


def _safe_extract(archive, destination, overlay=False):
    """Extract regular ZIP members only; validate *all* names before writing.

    data0613 is the official June-2024 correction overlay. Strip that single
    archive's data0613/ wrapper and overwrite base media, after base extraction.
    """
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        members = []
        for info in source.infolist():
            _inside(destination, info.filename)
            mode = info.external_attr >> 16
            if stat.S_ISLNK(mode) or (stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR)):
                raise ValueError(f'Unsupported ZIP member type: {info.filename!r}')
            parts = PurePosixPath(info.filename).parts
            if overlay:
                if not parts or parts[0] != 'data0613':
                    raise ValueError(f'Unexpected correction-overlay path: {info.filename!r}')
                parts = parts[1:]
                if not parts:
                    continue
            target = _inside(destination, PurePosixPath(*parts))
            if target.is_symlink():
                raise ValueError(f'Refusing ZIP symlink target: {target}')
            members.append((info, target))
        for info, target in members:
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open(info) as reader, tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as writer:
                temporary = Path(writer.name)
                try:
                    shutil.copyfileobj(reader, writer)
                except BaseException:
                    temporary.unlink(missing_ok=True)
                    raise
            os.replace(temporary, target)


def fetch_dataset(root: Path, scope='smoke'):
    """Download only approved pinned public inputs, without login or tokens.

    ``metadata`` gets all official annotations and rights notices. ``smoke``
    adds vlnqa.zip. ``full`` adds the current public archives (never NTU).
    Fetch is not a coverage assertion: call prepare_dataset before inference.
    """
    from huggingface_hub import hf_hub_download

    if scope not in ('metadata', 'smoke', 'full'):
        raise ValueError('scope must be metadata, smoke, or full')
    root = Path(root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    warnings.warn(RIGHTS_NOTICE, UserWarning, stacklevel=2)
    archives = [] if scope == 'metadata' else ['video/vlnqa.zip']
    if scope == 'full':
        archives = [name for name in ARCHIVE_SHA256 if name != 'video/data0613.zip'] + ['video/data0613.zip']
    requested = list(METADATA_SHA256) + archives
    for name in requested:
        _inside(root, name)
        path = Path(hf_hub_download(
            repo_id=DATASET_REPO, repo_type='dataset', revision=DATASET_REVISION,
            filename=name, local_dir=str(root), token=False,
        ))
        expected = METADATA_SHA256.get(name, ARCHIVE_SHA256.get(name))
        if _sha256(path) != expected:
            raise ValueError(f'Pinned file checksum mismatch: {name}. Remove the damaged local file and fetch again.')
        if name in archives:
            _safe_extract(path, root / 'video', overlay=name == 'video/data0613.zip')
    notice = root / '.mvbench' / 'RIGHTS_NOTICE.txt'
    notice.parent.mkdir(parents=True, exist_ok=True)
    notice.write_text(RIGHTS_NOTICE + '\n', encoding='utf-8')
    return {
        'root': str(root), 'dataset_revision': DATASET_REVISION, 'scope': scope,
        'downloaded_files': requested, 'rights_notice': RIGHTS_NOTICE,
        'manual_ntu_required_for_full': True,
    }


def _task_definitions():
    from vlmeval.dataset.mvbench import MVBench
    return MVBench.TASKS


def _read_annotations(root):
    """Authenticate every source byte before deriving identities or labels."""
    for name, expected in METADATA_SHA256.items():
        path = _inside(root, name)
        if not path.is_file():
            raise ValueError(f'Missing pinned metadata {name}; run fetch_dataset(root, scope="metadata").')
        if _sha256(path) != expected:
            raise ValueError(f'Pinned metadata checksum mismatch: {name}; local annotations are not trusted.')
    rows = []
    for task, (name, prefix, data_type, bound) in _task_definitions().items():
        source = json.loads((root / 'json' / name).read_text(encoding='utf-8'))
        if not isinstance(source, list) or len(source) != ROWS_PER_TASK:
            raise ValueError(f'{name} must contain exactly {ROWS_PER_TASK} questions')
        for position, item in enumerate(source):
            candidates = item.get('candidates')
            if (not isinstance(candidates, list) or not 1 <= len(candidates) <= 26
                    or not all(isinstance(option, str) and option.strip() for option in candidates)
                    or len(set(candidates)) != len(candidates)):
                raise ValueError(f'Invalid candidates in {name}:{position}')
            answer = item.get('answer')
            if not isinstance(answer, str) or candidates.count(answer) != 1:
                raise ValueError(f'Answer must identify exactly one candidate in {name}:{position}')
            if not isinstance(item.get('question'), str) or not item['question'].strip():
                raise ValueError(f'Missing question in {name}:{position}')
            if not isinstance(item.get('video'), str) or not item['video'].strip():
                raise ValueError(f'Missing video in {name}:{position}')
            media_prefix = prefix.replace('your_data_path', 'video')
            _inside(root, str(PurePosixPath(media_prefix) / item['video']))
            start, end = item.get('start'), item.get('end')
            if bound and (not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                                  for v in (start, end)) or start < 0 or end <= start):
                raise ValueError(f'Invalid temporal bounds in {name}:{position}')
            rows.append({
                'index': len(rows), 'task_type': task, 'prefix': media_prefix,
                'data_type': data_type, 'bound': bound, 'start': start, 'end': end,
                'video': item['video'], 'question': item['question'], 'answer': answer,
                # repr is exclusively of authenticated, validated string lists;
                # upstream evaluate's eval never receives incoming candidates.
                'candidates': repr(candidates),
            })
    if len(rows) != EXPECTED_ROWS or len(_task_definitions()) != 20:
        raise ValueError('MVBench must contain 4000 questions across all 20 native tasks')
    return rows


def _media_available(root, row):
    path = _inside(root, str(PurePosixPath(row['prefix']) / row['video']))
    if row['data_type'] != 'frame':
        return path.is_file() and path.stat().st_size > 0
    if not path.is_dir():
        return False
    frames = sorted(path.glob('*.jpg'))
    return bool(frames) and all(
        frame.name == f'{position:05d}.jpg' and frame.stat().st_size > 0
        for position, frame in enumerate(frames, 1)
    )


_MEDIA_HASH_CACHE = {}


def _media_content_sha256(root, row):
    """Hash media bytes, memoizing only while file identities/stats are stable."""
    path = _inside(root, str(PurePosixPath(row['prefix']) / row['video']))
    files = sorted(path.glob('*.jpg')) if row['data_type'] == 'frame' else [path]
    if not files:
        raise ValueError(f'Media has no frames: {path}')

    def signature():
        result = []
        for file in files:
            _inside(root, file.relative_to(root).as_posix())
            info = file.stat()
            result.append((file.name, info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns))
        return tuple(result)

    before = signature()
    key = str(path)
    cached = _MEDIA_HASH_CACHE.get(key)
    if cached is not None and cached[0] == before:
        return cached[1]
    if row['data_type'] == 'frame':
        digest = hashlib.sha256()
        for file in files:
            digest.update(file.name.encode())
            digest.update(b'\0')
            digest.update(bytes.fromhex(_sha256(file)))
        value = digest.hexdigest()
    else:
        value = _sha256(path)
    if signature() != before:
        raise ValueError(f'Media changed while hashing: {path}')
    _MEDIA_HASH_CACHE[key] = (before, value)
    return value


def _manifest_sha256():
    payload = {'repo': DATASET_REPO, 'revision': DATASET_REVISION, 'metadata': METADATA_SHA256,
               'archives': ARCHIVE_SHA256, 'tasks': _task_definitions(),
               'task_order': list(_task_definitions())}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def prepare_dataset(root: Path, smoke=False):
    """Rebuild canonical TSV from pinned annotations; never silently drop rows.

    Full requires all 4000 rows/media; smoke selects ONLY the first five official
    Egocentric Navigation identities. Coverage always reports the full universe.
    """
    import pandas as pd

    root = Path(root).expanduser().resolve()
    rows = _read_annotations(root)
    selected = [row for row in rows if row['task_type'] == SMOKE_TASK][:SMOKE_ROWS] if smoke else rows
    available = {}
    missing = {}
    for row in rows:
        key = (row['prefix'], row['video'], row['data_type'])
        if key not in available:
            available[key] = _media_available(root, row)
        if not available[key]:
            missing.setdefault(row['task_type'], set()).add(str(PurePosixPath(row['prefix']) / row['video']))
    coverage = {
        'expected_rows': EXPECTED_ROWS,
        'available_rows': sum(available[(r['prefix'], r['video'], r['data_type'])] for r in rows),
        'selected_rows': len(selected), 'expected_tasks': 20,
        'selected_tasks': list(dict.fromkeys(r['task_type'] for r in selected)),
        'subset': bool(smoke),
    }
    report = {
        'dataset_revision': DATASET_REVISION, 'coverage': coverage,
        'missing_media_by_task': {task: sorted(paths) for task, paths in missing.items()},
        'rights_notice': RIGHTS_NOTICE,
    }
    scope = 'smoke' if smoke else 'full'
    report_file = root / '.mvbench' / f'coverage-{scope}.json'
    _json_write(report_file, report)
    missing_selected = [r['index'] for r in selected if not available[(r['prefix'], r['video'], r['data_type'])]]
    if missing_selected:
        detail = ' Full evaluation requires all 4000 questions and all 20 tasks.' if not smoke else ' Smoke requires its fixed five navigation videos.'
        raise MVBenchCoverageError(
            f'MVBench media coverage incomplete: {coverage["available_rows"]}/4000 available; '
            f'{len(missing_selected)} selected rows lack usable media.{detail} '
            f'See {report_file}. {RIGHTS_NOTICE}', report,
        )
    media_hashes = {}
    for row in selected:
        relative = str(PurePosixPath(row['prefix']) / row['video'])
        if relative not in media_hashes:
            media_hashes[relative] = _media_content_sha256(root, row)
    annotation_manifest = _manifest_sha256()
    media_manifest = hashlib.sha256(json.dumps(media_hashes, sort_keys=True).encode()).hexdigest()
    manifest = hashlib.sha256(f'{annotation_manifest}:{media_manifest}'.encode()).hexdigest()
    _json_write(root / '.mvbench' / f'media-manifest-{scope}.json', {
        'dataset_revision': DATASET_REVISION, 'sha256': media_manifest, 'media': media_hashes,
    })
    data_file = root / '.mvbench' / f'MVBench-{scope}-{manifest[:16]}.tsv'
    with tempfile.NamedTemporaryFile(mode='w', dir=data_file.parent, delete=False) as stream:
        temporary = Path(stream.name)
        pd.DataFrame(selected).to_csv(stream, sep='\t', index=False)
    os.replace(temporary, data_file)
    return {
        'root': str(root), 'data_file': str(data_file), 'dataset_revision': DATASET_REVISION,
        'manifest_sha256': manifest, 'coverage': coverage,
        'annotation_manifest_sha256': annotation_manifest,
        'media_manifest_sha256': media_manifest, 'media_sha256': media_hashes,
        'selected_indices': [row['index'] for row in selected],
    }


def build_dataset(root: Path, nframe=16, smoke=False):
    from vlmeval.dataset.mvbench import MVBenchLocal
    return MVBenchLocal(data_root=str(root), nframe=nframe, smoke=smoke)


def _load_predictions(path):
    import pandas as pd

    path = Path(path)
    if path.suffix == '.tsv':
        return pd.read_csv(path, sep='\t', keep_default_na=False, dtype={'prediction': str})
    if path.suffix == '.xlsx':
        return pd.read_excel(path, keep_default_na=False, dtype={'prediction': str})
    if path.suffix == '.json':
        return pd.DataFrame(json.loads(path.read_text(encoding='utf-8')))
    raise ValueError('Predictions must be TSV, XLSX, or JSON; pickle inputs are not accepted')


def _verified_predictions(incoming, trusted):
    """Only strings cross the trust boundary; identities and labels are fixed."""
    if not {'index', 'prediction'}.issubset(incoming.columns):
        raise ValueError('Predictions require index and prediction columns')
    indices = []
    for value in incoming['index']:
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError(f'Invalid prediction identity: {value!r}') from None
        if isinstance(value, bool) or not math.isfinite(number) or number != int(number):
            raise ValueError(f'Invalid prediction identity: {value!r}')
        indices.append(int(number))
    expected = list(trusted['index'])
    if len(indices) != len(set(indices)) or set(indices) != set(expected):
        raise ValueError('Prediction coverage must match selected stable indices exactly: no missing, duplicate, or extra rows')
    incoming = incoming.copy()
    incoming['index'] = indices
    incoming = incoming.set_index('index').loc[expected]
    for position, row in trusted.reset_index(drop=True).iterrows():
        supplied = incoming.loc[row['index']]
        for column in trusted.columns:
            if column == 'index' or column not in incoming.columns:
                continue
            actual, wanted = supplied[column], row[column]
            if column == 'candidates':
                try:
                    actual = ast.literal_eval(actual) if isinstance(actual, str) else actual
                    wanted = ast.literal_eval(wanted)
                except (ValueError, SyntaxError, TypeError):
                    raise ValueError(f'Untrusted candidates for index {row["index"]}') from None
            elif column in ('start', 'end'):
                if wanted is None or (isinstance(wanted, float) and math.isnan(wanted)):
                    if actual == '' or actual is None or (isinstance(actual, float) and math.isnan(actual)):
                        continue
                try:
                    actual = float(actual)
                except (ValueError, TypeError):
                    pass
            elif column == 'bound' and isinstance(actual, str):
                actual = {'True': True, 'False': False}.get(actual, actual)
            if actual != wanted:
                raise ValueError(f'Untrusted {column} for index {row["index"]}')
    predictions = incoming['prediction'].tolist()
    for index, prediction in zip(expected, predictions):
        if (not isinstance(prediction, str) or not prediction.strip()
                or 'failed to obtain answer' in prediction.lower()
                or 'video_load_failed' in prediction.lower()):
            raise ValueError(f'Missing or failed prediction for index {index}; inference must complete before scoring')
    return predictions


def _summarize_rating(rating):
    per_task = {}
    for task, counts in rating.items():
        if task == 'overall':
            continue
        correct, total = counts[:2]
        if not (0 <= correct <= total and total > 0):
            raise ValueError(f'Invalid native score counts for {task}')
        per_task[task] = {'correct': int(correct), 'total': int(total), 'accuracy': correct / total}
    correct = sum(v['correct'] for v in per_task.values())
    total = sum(v['total'] for v in per_task.values())
    if not total or list(rating['overall'][:2]) != [correct, total]:
        raise ValueError('Native overall counts disagree with task counts')
    return {
        'primary_metric': {'name': 'accuracy', 'value': correct / total, 'unit': 'fraction',
                           'correct': correct, 'total': total},
        'task_macro_accuracy': sum(v['accuracy'] for v in per_task.values()) / len(per_task),
        'per_task': per_task,
    }


def score_predictions(predictions: Path, root: Path, output_dir: Path, smoke=False, nframe=16):
    """Native MVBench exact matching on authenticated labels, with no judge.

    Missing/failure predictions are rejected before native rating sees them:
    upstream get_dimension_rating treats -1 as truthy. Malformed nonempty
    strings still receive upstream wrong-answer semantics, not a fallback score.
    Native pandas row positions are made contiguous for its .loc assignment;
    original stable identities are restored in the saved scored_predictions.tsv.
    """
    import pandas as pd
    from vlmeval.dataset.mvbench import MVBench
    from vlmeval.smp import get_intermediate_file_path, load

    if isinstance(nframe, bool) or not isinstance(nframe, int) or nframe <= 0:
        raise ValueError('nframe must be a positive integer')
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists():
        raise ValueError('Scoring requires a fresh output directory')
    prepared = prepare_dataset(root, smoke=smoke)
    trusted = pd.read_csv(prepared['data_file'], sep='\t')
    trusted['prediction'] = _verified_predictions(_load_predictions(predictions), trusted)
    official_indices = trusted['index'].tolist()
    trusted['index'] = range(len(trusted))
    output_dir.mkdir(parents=True, exist_ok=False)
    # Never consume a stale native _score cache supplied alongside predictions.
    with tempfile.TemporaryDirectory(prefix='mvbench-native-', dir=output_dir) as temporary:
        native_file = Path(temporary) / 'trusted.tsv'
        trusted.to_csv(native_file, sep='\t', index=False)
        rating = MVBench.evaluate(str(native_file), trusted_data=trusted, model='exact_matching')
        scored = load(get_intermediate_file_path(str(native_file), '_score'))
        if len(scored) != len(trusted) or not scored['score'].isin([0, 1]).all():
            raise ValueError('Native scorer returned missing, failed, or nonbinary scores; no result published')
        scored['index'] = official_indices
        scored.to_csv(output_dir / 'scored_predictions.tsv', sep='\t', index=False)
    summary = {
        'status': 'smoke_only' if smoke else 'complete', 'dataset': 'MVBench',
        'coverage': prepared['coverage'], **_summarize_rating(rating),
        'native_rating': rating,
        'provenance': {
            'dataset_repo': DATASET_REPO, 'dataset_revision': DATASET_REVISION,
            'manifest_sha256': prepared['manifest_sha256'], 'selected_indices': official_indices,
            'annotation_manifest_sha256': prepared['annotation_manifest_sha256'],
            'media_manifest_sha256': prepared['media_manifest_sha256'],
            'media_sha256': prepared['media_sha256'],
            'prediction_sha256': _sha256(predictions), 'scorer': NATIVE_SCORER,
            'scorer_sha256': _sha256(Path(__file__).parent / 'dataset' / 'mvbench.py'),
            'scorer_utils_sha256': _sha256(Path(__file__).parent / 'dataset' / 'utils' / 'mvbench.py'),
            'judge': 'exact_matching', 'network_judge': False, 'nframe': nframe,
            'local_scoring_guards': [
                'reject missing/failure predictions before native rating',
                'punctuation-only first option token scores zero, never substring credit',
            ],
            'prompt_mode': 'native sampled images', 'rights_notice': RIGHTS_NOTICE,
            'frame_resolution': FRAME_RESOLUTION, 'frame_preprocessing': FRAME_PREPROCESSING,
            'reference_recipe': {'nframe': 16, 'resolution': 224},
            'model_normalization': 'Cosmos serving model, not VideoChat2 tensor normalization',
            'paper_parity': False,
        },
    }
    _json_write(output_dir / 'summary.json', summary)
    return summary
