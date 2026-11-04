# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Consumer-boundary evidence tests: no endpoint, model imports or media downloads."""
import importlib.util
import json
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    'execution_evidence', Path(__file__).resolve().parents[1] / 'vlmeval/execution_evidence.py')
evidence = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evidence)


def video_response(sizes=None):
    sizes = [49] * 16 if sizes is None else sizes
    return {'model': 'nvidia/Cosmos3-Nano', 'usage': {'prompt_tokens': 850, 'completion_tokens': 1, 'total_tokens': 851},
            'prompt_token_ids': [token for size in sizes for token in [151652, *([151656] * size), 151653]],
            'choices': [{'finish_reason': 'stop', 'message': {'content': '4'}}]}


def request():
    return {'model': 'nvidia/Cosmos3-Nano', 'temperature': 0, 'max_tokens': 256,
            'messages': [{'role': 'user', 'content': [{'type': 'video_url', 'video_url': {'url': 'data:video/mp4;base64,PRIVATE'}},
                                                    {'type': 'text', 'text': 'PRIVATE QUESTION; answer label B'}]}],
            'mm_processor_kwargs': {'do_sample_frames': False},
            'return_token_ids': True}


def record(raw=None, profile=True, media=None):
    return evidence.build_execution_record(request(), video_response() if raw is None else raw,
        processor_profile=evidence.nano_processor_profile('nvidia/Cosmos3-Nano') if profile else None,
        media={'sampled_frames': 32, 'expected_group_tokens': 49} if media is None else media)


class ExecutionEvidenceTests(unittest.TestCase):
    def test_per_group_area_and_frame_slots_are_not_decoded_frame_claims(self):
        result = record()
        self.assertEqual(result['verification']['status'], 'verified_under_explicit_profile')
        observed = result['observed']
        self.assertEqual(observed['video_frame_slots'], 32)
        self.assertEqual(observed['processor_grid_pixel_area_per_video_temporal_group'], [224 * 224] * 16)
        self.assertIsNone(observed['decoded_video_frames'])
        self.assertIsNone(observed['processor_width_height'])
        self.assertIsNone(observed['effective_decoder_fps'])
        self.assertIsNone(observed['effective_generation_settings'])
        self.assertIsNone(observed['effective_model_revision'])
        self.assertEqual(observed['model'], 'nvidia/Cosmos3-Nano')
        self.assertEqual(observed['usage']['total_tokens'], 851)
        self.assertEqual(observed['finish_reasons'], ['stop'])
        self.assertEqual(result['requested']['generation']['temperature'], 0)

    def test_dropped_frame_slots_and_equal_total_pixel_mismatch_are_detected(self):
        dropped = record(video_response([49] * 2))
        self.assertEqual(dropped['observed']['video_frame_slots'], 4)
        self.assertEqual(dropped['verification']['checks']['video_frame_slots'], 'mismatch')
        wrong_pixels = record(video_response([48, 50] + [49] * 14))
        self.assertEqual(wrong_pixels['verification']['checks']['video_frame_slots'], 'match')
        self.assertEqual(wrong_pixels['verification']['checks']['pixel_geometry'], 'mismatch')
        self.assertEqual(wrong_pixels['verification']['status'], 'mismatch')

    def test_unobservable_profile_tokens_or_requested_geometry_never_verify(self):
        for result in (record(profile=False), record({'choices': video_response()['choices']}),
                       record(media={})):
            with self.subTest(result=result['verification']):
                self.assertEqual(result['verification']['status'], 'unknown')
        self.assertIsNone(record(profile=False)['observed']['video_frame_slots'])
        self.assertIsNone(evidence.nano_processor_profile('unrecognized-model'))

    def test_raw_unbounded_token_total_does_not_prove_temporal_geometry(self):
        raw = video_response()
        raw['prompt_token_ids'] = [151656] * 784
        result = record(raw)
        self.assertIsNone(result['observed']['video_frame_slots'])
        self.assertEqual(result['verification']['checks']['pixel_geometry'], 'mismatch')

    def test_failed_or_malformed_response_never_looks_verified(self):
        for raw in ({}, {'error': {'message': 'PRIVATE API KEY'}},
                    dict(video_response(), error={'message': 'failure'}),
                    dict(video_response(), choices=[]),
                    dict(video_response(), choices=[{'message': {'content': '4'}, 'finish_reason': 'error'}])):
            result = record(raw)
            self.assertEqual(result['verification']['status'], 'failed_response')
            self.assertIsNone(result['observed']['video_frame_slots'])

    def test_redaction_is_allowlisted_not_recursive_copying(self):
        payload = request()
        payload.update(api_key='SECRET', authorization='Bearer SECRET', endpoint='https://u:p@host/?token=SECRET',
                       labels={'answer': 'PRIVATE LABEL'}, mm_processor_kwargs={'max_pixels': 50176, 'api_key': 'SECRET'})
        raw = video_response()
        raw['choices'][0]['message']['content'] = 'PRIVATE OUTPUT'
        raw.update(debug='SECRET', prompt='PRIVATE QUESTION')
        result = evidence.build_execution_record(payload, raw,
            media={'sampled_frames': 32, 'source': 'PRIVATE PATH', 'labels': 'PRIVATE LABEL'},
            processor_profile=dict(evidence.nano_processor_profile('nvidia/Cosmos3-Nano'), api_key='SECRET'))
        encoded = json.dumps(result, allow_nan=False)
        for secret in ('SECRET', 'PRIVATE', 'Bearer', 'data:video', 'https://u:p'):
            self.assertNotIn(secret, encoded)
        self.assertNotIn('prompt_token_ids', result['observed'])
        self.assertEqual(result['requested']['mm_processor_kwargs']['max_pixels'], 50176)

    def test_image_pixel_mismatch_and_resize_ambiguity_remain_distinct(self):
        payload = {'model': 'nvidia/Cosmos3-Nano', 'messages': [{'content': [{'type': 'image_url'}]}]}
        raw = video_response([])
        raw['prompt_token_ids'] = [151652] + [151655] * 64 + [151653]
        profile = evidence.nano_processor_profile('nvidia/Cosmos3-Nano')
        aligned = evidence.build_execution_record(payload, raw, processor_profile=profile,
            media={'image_dimensions': [{'width': 256, 'height': 256}]})
        self.assertEqual(aligned['verification']['status'], 'verified_under_explicit_profile')
        mismatch = evidence.build_execution_record(payload, raw, processor_profile=profile,
            media={'image_dimensions': [{'width': 256, 'height': 288}]})
        self.assertEqual(mismatch['verification']['checks']['pixel_geometry'], 'mismatch')
        resized = evidence.build_execution_record(payload, raw, processor_profile=profile,
            media={'image_dimensions': [{'width': 224, 'height': 224}]})
        self.assertEqual(resized['verification']['checks']['pixel_geometry'], 'unknown')
        self.assertEqual(resized['observed']['processor_grid_pixel_area_per_image'], [65536])

    def test_aggregate_retains_actual_shapes_and_never_counts_missing_evidence_as_coverage(self):
        summary = evidence.aggregate_execution_records([record(), record(video_response([49] * 2)), {}], expected_requests=3)
        self.assertEqual(summary['requests_recorded'], 2)
        self.assertFalse(summary['coverage_complete'])
        self.assertFalse(summary['all_requested_geometry_checks_matched_under_explicit_profile'])
        self.assertEqual(summary['verification_counts']['mismatch'], 1)
        self.assertEqual({shape['video_frame_slots'] for shape in summary['observed_visual_shapes']}, {4, 32})
        self.assertEqual(summary['usage_totals']['total_tokens'], 1702)
        self.assertIsNone(summary['effective_model_revision'])
        json.dumps(summary, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
