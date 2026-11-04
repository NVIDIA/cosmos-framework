# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Redacted, observational evidence; requested settings are never server attestations."""
from __future__ import annotations

from collections import Counter
import json
import math
import re

NANO_REVISION = 'e59a53c25979a090fa8706c9acc0c254a6e89b92'
PINNED_SERVING_COMPATIBILITY = {
    'image': 'vllm/vllm-openai:cosmos3@sha256:db0bb920b0b54e82ea96a98659bbd21921f87d0dcfc86feffdafa2db3f08be55',
    'inspected_version': '0.23.1rc1.dev1306+gbfee6a802',
    'inspection_scope': 'existing review container source and serve help; not attestation of this response endpoint',
    'sampler_environment': {'VLLM_USE_FLASHINFER_SAMPLER': '0'},
    'trust_request_mm_kwargs_flag_supported': False,
    'supported_equivalent': 'unconditional request mm_processor_kwargs via renderers/online_renderer.py RenderParams and renderers/hf.py',
}


def _identity(value):
    return value if isinstance(value, str) and re.fullmatch(r'[A-Za-z0-9_./+:-]{1,240}', value) and '://' not in value else None


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _knobs(value, names):
    if not isinstance(value, dict):
        return {}
    return {key: value[key] for key in names if key in value and
            (value[key] is None or isinstance(value[key], bool) or _number(value[key]))}


def nano_processor_profile(model, revision='unknown'):
    """Explicit pinned architecture assumption, NOT proof of the endpoint revision."""
    if model != 'nvidia/Cosmos3-Nano':
        return None
    return {'patch_size': 16, 'spatial_merge_size': 2, 'temporal_patch_size': 2,
            'image_token_id': 151655, 'video_token_id': 151656,
            'vision_start_token_id': 151652, 'vision_end_token_id': 151653,
            'min_pixels': 65536, 'max_pixels': 16777216,
            'profile_model': model, 'profile_revision': NANO_REVISION,
            'requested_model_revision': revision,
            'provenance': 'pinned-public-nano-config'}


def _profile(value):
    if not isinstance(value, dict):
        return None
    names = ('patch_size', 'spatial_merge_size', 'temporal_patch_size', 'image_token_id',
             'video_token_id', 'vision_start_token_id', 'vision_end_token_id')
    result = {key: value[key] for key in names if isinstance(value.get(key), int)
              and not isinstance(value[key], bool) and value[key] > 0}
    for key in ('min_pixels', 'max_pixels'):
        if isinstance(value.get(key), int) and not isinstance(value[key], bool) and value[key] > 0:
            result[key] = value[key]
    if not all(key in result for key in names[:5]):
        return None
    result['basis'] = 'explicit caller-supplied processor profile; not server-attested'
    if value.get('provenance') == 'pinned-public-nano-config':
        result.update(profile_model='nvidia/Cosmos3-Nano', profile_revision=NANO_REVISION,
                      sources=[f'https://huggingface.co/nvidia/Cosmos3-Nano/resolve/{NANO_REVISION}/{name}'
                               for name in ('config.json', 'preprocessor_config.json')])
    return result


def _groups(ids, token, profile):
    groups = []
    index = 0
    while index < len(ids):
        if ids[index] != token:
            index += 1
            continue
        start = index
        while index < len(ids) and ids[index] == token:
            index += 1
        groups.append({'tokens': index - start,
                       'bounded_by_vision_markers': bool(start > 0 and index < len(ids) and
                           ids[start - 1] == profile.get('vision_start_token_id') and
                           ids[index] == profile.get('vision_end_token_id'))})
    return groups


def build_execution_record(payload: dict, response_json: dict, *, media=None, processor_profile=None,
                           requested_model_revision=None) -> dict:
    """Keep only allowlisted settings and visual-token geometry, never prompts/URLs/labels.

    A video group describes one temporal patch (frame slots), not decoded unique
    frames. Repeated/padded frames, pruning, width/height and decoder FPS cannot be
    established from token counts. Pixel area is conditional on the explicit
    processor profile and unpruned standard Qwen3-VL placeholder expansion.
    """
    raw = response_json if isinstance(response_json, dict) else {}
    media = media if isinstance(media, dict) else {}
    profile = _profile(processor_profile)
    generation = _knobs(payload, ('temperature', 'top_p', 'top_k', 'seed', 'max_tokens',
                                  'max_completion_tokens', 'repetition_penalty', 'presence_penalty',
                                  'frequency_penalty', 'n'))
    processor = _knobs(payload.get('mm_processor_kwargs'),
                       ('fps', 'do_sample_frames', 'min_pixels', 'max_pixels', 'total_pixels',
                        'nframes', 'min_frames', 'max_frames', 'do_resize'))
    io = payload.get('media_io_kwargs', {})
    io = {kind: _knobs(io.get(kind), ('num_frames', 'fps', 'min_frames', 'max_frames'))
          for kind in ('image', 'video') if isinstance(io, dict) and kind in io}
    parts = [part for msg in payload.get('messages', []) if isinstance(msg, dict)
             for part in (msg.get('content') if isinstance(msg.get('content'), list) else [])
             if isinstance(part, dict)]
    images = sum(part.get('type') == 'image_url' for part in parts)
    videos = sum(part.get('type') == 'video_url' for part in parts)
    requested_media = _knobs(media, ('sampled_frames', 'width', 'height', 'expected_frame_slots',
                                     'expected_group_tokens', 'nframes', 'min_pixels', 'max_pixels',
                                     'fps', 'total_pixels', 'max_frames'))
    dimensions = media.get('image_dimensions', [])
    if isinstance(dimensions, list) and all(isinstance(item, dict) and
            all(_number(item.get(key)) and item[key] > 0 for key in ('width', 'height')) for item in dimensions):
        requested_media['image_dimensions'] = [_knobs(item, ('width', 'height')) for item in dimensions]
    choices = raw.get('choices')
    valid = ('error' not in raw and isinstance(choices, list) and bool(choices) and
             all(isinstance(choice, dict) and choice.get('finish_reason') in ('stop', 'length') and
                 isinstance(choice.get('message'), dict) and 'content' in choice['message'] and
                 (choice['message']['content'] is None or isinstance(choice['message']['content'], str))
                 for choice in choices))
    observed = {'response_valid': bool(valid), 'model': _identity(raw.get('model')),
                'usage': _knobs(raw.get('usage'), ('prompt_tokens', 'completion_tokens', 'total_tokens')),
                'finish_reasons': [choice.get('finish_reason') if isinstance(choice, dict) and
                                   choice.get('finish_reason') in ('stop', 'length', 'error', 'content_filter') else None
                                   for choice in choices] if isinstance(choices, list) else [],
                'image_token_groups': None, 'video_token_groups': None,
                'processor_grid_pixel_area_per_image': None, 'processor_grid_pixel_area_per_video_temporal_group': None,
                'video_frame_slots': None, 'decoded_video_frames': None, 'processor_width_height': None,
                'effective_generation_settings': None, 'effective_model_revision': None,
                'effective_decoder_fps': None}
    checks = {'image_count': 'unknown', 'video_frame_slots': 'unknown', 'pixel_geometry': 'unknown'}
    ids = raw.get('prompt_token_ids')
    if valid and profile and isinstance(ids, list) and all(isinstance(token, int) for token in ids):
        ig = _groups(ids, profile['image_token_id'], profile)
        vg = _groups(ids, profile['video_token_id'], profile)
        area = (profile['patch_size'] * profile['spatial_merge_size']) ** 2
        observed.update(image_token_groups=ig, video_token_groups=vg,
                        processor_grid_pixel_area_per_image=[group['tokens'] * area for group in ig],
                        processor_grid_pixel_area_per_video_temporal_group=[group['tokens'] * area for group in vg])
        bounded = all(group['bounded_by_vision_markers'] for group in ig + vg)
        if images and not videos:
            checks['image_count'] = 'match' if bounded and len(ig) == images and not vg else 'mismatch'
        if videos and vg and bounded and not ig:
            observed['video_frame_slots'] = len(vg) * profile['temporal_patch_size']
            expected = media.get('expected_frame_slots', media.get('sampled_frames', media.get('nframes')))
            if _number(expected):
                checks['video_frame_slots'] = 'match' if observed['video_frame_slots'] == expected else 'mismatch'
        expected_tokens = media.get('expected_group_tokens')
        if _number(expected_tokens):
            groups = vg if videos else ig
            checks['pixel_geometry'] = 'match' if groups and bounded and all(
                group['tokens'] == expected_tokens for group in groups) else 'mismatch'
        elif images and len(requested_media.get('image_dimensions', [])) == images and bounded and len(ig) == images:
            # Alignment alone does not prevent smart_resize from enlarging or shrinking.
            dims = requested_media['image_dimensions']
            factor = profile['patch_size'] * profile['spatial_merge_size']
            minimum = processor.get('min_pixels', profile.get('min_pixels'))
            maximum = processor.get('max_pixels', profile.get('max_pixels'))
            if (_number(minimum) and _number(maximum) and all(
                    item['width'] % factor == 0 and item['height'] % factor == 0 and
                    minimum <= item['width'] * item['height'] <= maximum for item in dims)):
                checks['pixel_geometry'] = 'match' if all(group['tokens'] * area == item['width'] * item['height']
                    for group, item in zip(ig, dims)) else 'mismatch'
        if (ig or vg) and bounded:
            cap = processor.get('max_pixels')
            if _number(cap) and any(group['tokens'] * area > cap for group in ig + vg):
                checks['pixel_geometry'] = 'mismatch'
    required = ['image_count', 'pixel_geometry'] if images and not videos else ['video_frame_slots', 'pixel_geometry']
    status = ('failed_response' if not valid else 'mismatch' if 'mismatch' in checks.values() else
              'verified_under_explicit_profile' if all(checks[key] == 'match' for key in required) else 'unknown')
    return {'format_version': 1,
            'requested': {'model': _identity(payload.get('model')), 'generation': generation,
                          'model_revision': _identity(requested_model_revision),
                          'mm_processor_kwargs': processor, 'media_io_kwargs': io,
                          'chat_template_kwargs': _knobs(payload.get('chat_template_kwargs'), ('enable_thinking',)),
                          'image_inputs': images, 'video_inputs': videos, 'media': requested_media},
            'observed': observed, 'processor_profile': profile,
            'verification': {'status': status, 'checks': checks},
            'derivation': {'pixel_area_units': 'pixels per image / per temporal group spatial grid (not summed over time)',
                           'pixel_area_formula': 'consecutive visual tokens * (patch_size * spatial_merge_size)^2',
                           'video_frame_slots_formula': 'bounded video token groups * temporal_patch_size',
                           'limits': ['conditional on explicit processor profile and no visual-token pruning',
                                      'token groups do not prove decoded/unique frames or source sampling indices',
                                      'area does not establish width, height, crop, decoder FPS or source resolution',
                                      'generation settings and model revision are requested/configured, not server-attested']}}


def aggregate_execution_records(records, expected_requests=None):
    records = [record for record in records if isinstance(record, dict) and
               isinstance(record.get('requested'), dict) and isinstance(record.get('observed'), dict)]
    counts = Counter(record.get('verification', {}).get('status', 'unknown') for record in records)
    return {'requests_recorded': len(records), 'expected_requests': expected_requests,
            'coverage_complete': expected_requests is not None and len(records) == expected_requests,
            'verification_counts': dict(counts),
            'all_requested_geometry_checks_matched_under_explicit_profile': bool(records) and
                expected_requests == len(records) and counts.get('verified_under_explicit_profile', 0) == len(records),
            'observed_models': sorted({record['observed']['model'] for record in records if record.get('observed', {}).get('model')}),
            'finish_reasons': dict(Counter(reason for record in records for reason in
                                          record.get('observed', {}).get('finish_reasons', []) if reason)),
            'observed_visual_shapes': [
                {'image_group_tokens': list(shape[0]), 'video_group_tokens': list(shape[1]),
                 'image_grid_pixel_areas': list(shape[2]), 'video_temporal_grid_pixel_areas': list(shape[3]),
                 'video_frame_slots': shape[4], 'requests': count}
                for shape, count in Counter(
                    (tuple(group['tokens'] for group in (record.get('observed', {}).get('image_token_groups') or [])),
                     tuple(group['tokens'] for group in (record.get('observed', {}).get('video_token_groups') or [])),
                     tuple(record.get('observed', {}).get('processor_grid_pixel_area_per_image') or []),
                     tuple(record.get('observed', {}).get('processor_grid_pixel_area_per_video_temporal_group') or []),
                     record.get('observed', {}).get('video_frame_slots')) for record in records).items()],
            'requested_settings': [
                {'settings': json.loads(value), 'requests': count}
                for value, count in Counter(json.dumps(record.get('requested', {}), sort_keys=True)
                                            for record in records).items()],
            'usage_totals': {key: sum(record.get('observed', {}).get('usage', {}).get(key, 0) or 0 for record in records)
                             for key in ('prompt_tokens', 'completion_tokens', 'total_tokens')},
            'unknown_fields': ['decoded_video_frames', 'processor_width_height', 'effective_decoder_fps',
                               'effective_generation_settings', 'effective_model_revision'],
            'effective_generation_settings': None, 'effective_model_revision': None,
            'decoded_video_frames': None}
