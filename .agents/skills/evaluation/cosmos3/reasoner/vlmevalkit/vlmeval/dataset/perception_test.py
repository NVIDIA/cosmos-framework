# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native video harness integration for public Perception Test MCQ validation."""
import json
from numbers import Integral
import os
from pathlib import Path
import tempfile

from PIL import Image
import portalocker

from vlmeval.perception_test_local import (
    DATASET_REVISION, FRAME_PREPROCESSING, MAX_FRAME_SIDE, PROTOCOL, SAMPLING,
    _fingerprint, _inside, _integer, _media_content_sha256, prepare_dataset, score_predictions,
)
from .video_base import VideoBaseDataset


class PerceptionTestLocal(VideoBaseDataset):
    TYPE = 'Video-MCQ'
    DEFAULT_JUDGE = 'exact_matching'

    def __init__(self, dataset='PerceptionTest', data_root=None, nframe=16, smoke=False,
                 fps=-1, pack=False, total_pixels=None, **kwargs):
        import pandas as pd

        if dataset != 'PerceptionTest':
            raise ValueError('PerceptionTestLocal supports dataset="PerceptionTest" only')
        _integer(nframe, 'nframe', 1)
        if fps > 0 or pack or total_pixels is not None:
            raise ValueError('PerceptionTestLocal requires fixed-frame unpacked images with maxside768 preprocessing')
        root = data_root or os.environ.get('PERCEPTION_TEST_DATA_DIR')
        if not root:
            raise ValueError('Set PERCEPTION_TEST_DATA_DIR or provide data_root; fetch the public dataset first')
        self._local_root = Path(root).expanduser().resolve()
        self._local_smoke = bool(smoke)
        super().__init__(dataset=dataset, nframe=nframe, fps=fps, **kwargs)
        # The generic loader recognizes strings such as \"NA\" as missing values;
        # official option text must survive native inference/export unchanged.
        self.data = pd.read_csv(self.data_file, sep='\t', keep_default_na=False)
        self.frame_root = str(self._local_root / '.perception-test' / 'frames')
        Path(self.frame_root).mkdir(parents=True, exist_ok=True)

    @classmethod
    def supported_datasets(cls):
        return ['PerceptionTest']

    def prepare_dataset(self, dataset='PerceptionTest'):
        self.local_provenance = prepare_dataset(self._local_root, smoke=self._local_smoke, require_media=True)
        return self.local_provenance

    @staticmethod
    def _bounds(line):
        cut, declared = line['cut_frame'], line['num_frames']
        if (isinstance(cut, bool) or not isinstance(cut, Integral)
                or isinstance(declared, bool) or not isinstance(declared, Integral)
                or declared <= 0 or (cut != -1 and not 0 < cut <= declared)):
            raise ValueError(f'Invalid exclusive cut {cut!r} for declared {declared!r} frames; never clamp')
        return int(cut), int(declared)

    def clip_cache_key(self, line):
        cut, declared = self._bounds(line)
        return _fingerprint({
            'dataset_revision': DATASET_REVISION, 'media': line['video'],
            'media_sha256': _media_content_sha256(self._local_root, line),
            'cut_frame': cut, 'metadata_num_frames': declared,
            'nframe': self.nframe, 'protocol': PROTOCOL, 'sampling': SAMPLING,
            'max_frame_side': MAX_FRAME_SIDE, 'preprocessing': FRAME_PREPROCESSING,
        })

    @staticmethod
    def sample_indices(total_frames, cut_frame, nframe):
        _integer(total_frames, 'decoded frame count', 1)
        _integer(nframe, 'nframe', 1)
        if (isinstance(cut_frame, bool) or not isinstance(cut_frame, int)
                or (cut_frame != -1 and not 0 < cut_frame <= total_frames)):
            raise ValueError(f'Invalid exclusive frame cut {cut_frame!r} for {total_frames} decoded frames; never clamp')
        stop = total_frames if cut_frame == -1 else cut_frame
        return [(2 * i + 1) * stop // (2 * nframe) for i in range(nframe)]

    def save_video_frames(self, line):
        """Decode only allowed source indices; cache content+cut+protocol identity."""
        import cv2

        if isinstance(line, str):
            matches = self.data[self.data['video'] == line]
            if matches.empty:
                raise ValueError(f'Unknown canonical video: {line}')
            line = matches.iloc[0]
        key = self.clip_cache_key(line)
        directory = Path(self.frame_root) / key
        paths = [directory / f'frame-{i + 1}-of-{self.nframe}.jpg' for i in range(self.nframe)]
        directory.mkdir(parents=True, exist_ok=True)
        with portalocker.Lock(str(directory) + '.lock', 'w', timeout=300):
            if all(path.is_file() and path.stat().st_size > 0 for path in paths):
                return [str(path) for path in paths]
            source = _inside(self._local_root, line['video'])
            reader = cv2.VideoCapture(str(source))
            try:
                if not reader.isOpened():
                    raise ValueError(f'Cannot decode Perception Test media: {source}')
                count = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
                cut, declared = self._bounds(line)
                indices = self.sample_indices(count, cut, self.nframe)
                for path, index in zip(paths, indices):
                    if not reader.set(cv2.CAP_PROP_POS_FRAMES, index):
                        raise ValueError(f'Cannot seek source frame {index}: {source}')
                    ok, frame = reader.read()
                    if not ok or frame is None:
                        raise ValueError(f'Cannot decode required source frame {index}: {source}')
                    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    image.thumbnail((MAX_FRAME_SIDE, MAX_FRAME_SIDE), Image.Resampling.LANCZOS)
                    with tempfile.NamedTemporaryFile(dir=directory, suffix='.jpg', delete=False) as stream:
                        temporary = Path(stream.name)
                    try:
                        image.save(temporary, format='JPEG')
                        os.replace(temporary, path)
                    finally:
                        temporary.unlink(missing_ok=True)
            finally:
                reader.release()
            if self.clip_cache_key(line) != key:
                raise ValueError(f'Media changed during frame extraction: {source}; retry preparation')
        return [str(path) for path in paths]

    def build_prompt(self, line, video_llm=False):
        # VIDEO_LLM=True must not switch to raw video or bypass the official cut.
        if isinstance(line, int):
            line = self.data.iloc[line]
        options = line['options']
        options = json.loads(options) if isinstance(options, str) else options
        if not isinstance(options, list) or len(options) != 3 or not all(isinstance(o, str) for o in options):
            raise ValueError('Perception Test prompt requires three ordered options')
        text = ('Watch the sampled video frames and choose the best answer.\n'
                f'Question: {line["question"]}\nOptions:\n'
                + '\n'.join(f'({chr(65 + i)}) {option}' for i, option in enumerate(options))
                + '\nRespond with exactly one option letter: A, B, or C.')
        return ([{'type': 'image', 'value': path} for path in self.save_video_frames(line)]
                + [{'type': 'text', 'value': text}])

    def evaluate(self, eval_file, **judge_kwargs):
        if judge_kwargs.get('model', 'exact_matching') != 'exact_matching':
            raise ValueError('PerceptionTestLocal uses local option-ID top1 only; remote judges are not allowed')
        path = Path(eval_file)
        return score_predictions(path, self._local_root, path.parent / f'{path.stem}_perception_test',
                                 smoke=self._local_smoke, nframe=self.nframe)
