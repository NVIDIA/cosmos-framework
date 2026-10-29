"""CPU checks for the SGLang Bimanual cookbook: input preparation, payload encoding, and the HTTP client."""

import ast
import contextlib
import io
import itertools
import json
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx
import numpy as np
from PIL import Image

BIMANUAL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BIMANUAL_ROOT))

import bimanual_helpers
from bimanual_helpers import (
    CANVASES,
    UNIFIED_ACTION_DIM,
    UNIFIED_ACTION_SPACE,
    BimanualInput,
    domain_runs,
    encode_extra_params,
    first_media_image,
    load_action_sidecar,
    prepare_bimanual_input,
    submit_bimanual,
)

try:
    import orjson
except ImportError:  # The helper falls back to the standard library; both paths are covered when orjson is present.
    orjson = None

NUM_FRAMES = 9  # A short 4k+1 rollout with eight action rows.
DOMAINS = ["agibotworld"] * 4 + ["camera_pose"] * 2 + ["agibotworld"] * 2
MODEL = "nvidia/Cosmos3-Nano-Sim-Bimanual"


def make_rows(count: int = NUM_FRAMES - 1, seed: int = 0) -> list[list[float]]:
    """Pre-normalized double-precision rows, as the published sidecars store them."""
    rows = np.random.default_rng(seed).uniform(-1, 1, (count, UNIFIED_ACTION_DIM))
    rows[0, :4] = (0.0, 0.0, 1.1982105263541598e-07, 1.0)  # The first values of the published sample.
    return rows.tolist()


def parse_multipart(request: httpx.Request) -> dict[str, tuple[str | None, str | None, bytes]]:
    """Return each multipart part as name -> (filename, content type, body)."""
    boundary = request.headers["content-type"].split("boundary=", 1)[1].encode()
    parts = {}
    for chunk in request.read().split(b"--" + boundary)[1:-1]:
        header, body = chunk.removeprefix(b"\r\n").split(b"\r\n\r\n", 1)
        fields = {key.decode(): value.decode() for key, value in re.findall(rb'(\w+)="([^"]*)"', header)}
        content_type = re.search(rb"Content-Type: ([^\r\n]+)", header)
        parts[fields["name"]] = (
            fields.get("filename"), content_type.group(1).decode() if content_type else None, body.removesuffix(b"\r\n"),
        )
    return parts


class BimanualFixture(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.image_path = self.root / "initial.png"
        Image.new("RGB", (640, 360), (40, 80, 120)).save(self.image_path)  # The published example's image size.
        self.rows = make_rows()
        self.sidecar_path = self.write_sidecar("actions.json", self.rows)
        # The notebook builds its record from the published manifest row plus the downloaded paths.
        self.record = {
            "num_frames": NUM_FRAMES, "fps": 30, "seed": 2026, "resolution": "480", "raw_action_dim": 59,
            "format_prompt_as_json": False, "prompt": "  Place the cup in the sink.  ",
            "vision_path": "initial.png", "action_path": "actions.json",
        }

    def write_sidecar(self, name: str, rows, domain_names=DOMAINS) -> Path:
        """Write a unified sidecar; ``domain_names=None`` omits the field."""
        payload = {"action_space": UNIFIED_ACTION_SPACE, "action": rows, "contract": "slow_loop_middle_third_v1"}
        if domain_names is not None:
            payload["domain_names"] = domain_names
        path = self.root / name
        path.write_text(json.dumps(payload))
        return path

    def prepare(self, **overrides) -> BimanualInput:
        return prepare_bimanual_input({**self.record, **overrides}, base_dir=self.root)


class PreparationTests(BimanualFixture):
    def test_published_shape_record_prepares_canvas_output_size_and_metadata(self):
        record = dict(self.record)
        prepared = prepare_bimanual_input(record, base_dir=self.root)
        self.assertEqual(record, self.record)  # The record is read, not rewritten.
        self.assertEqual(prepared.prompt, "Place the cup in the sink.")
        self.assertEqual((prepared.image.size, prepared.image.mode), ((640, 360), "RGB"))
        self.assertEqual((prepared.action.dtype, prepared.action.shape), (np.float32, (NUM_FRAMES - 1, UNIFIED_ACTION_DIM)))
        np.testing.assert_array_equal(prepared.action, np.asarray(self.rows, dtype=np.float32))
        self.assertEqual(prepared.domain_names, DOMAINS)
        self.assertEqual((prepared.num_frames, prepared.fps, prepared.seed), (NUM_FRAMES, 30, 2026))
        self.assertEqual((prepared.width, prepared.height), (832, 480))  # The 480p canvas closest to 16:9.
        self.assertEqual((prepared.output_width, prepared.output_height), (640, 352))  # Content region in 16-pixel steps.
        self.assertEqual(prepared.extra_args, {
            "action_mode": "forward_dynamics", "domain_name": "agibotworld", "domain_names": DOMAINS,
            "action_normalization": "none", "action_view_point": "ego_view", "format_prompt_as_json": False,
        })
        self.assertEqual(prepared.metadata, {
            "vision_path": str(self.image_path), "action_source": str(self.sidecar_path),
            "action_space": UNIFIED_ACTION_SPACE,
            "domain_runs": [
                {"domain_name": "agibotworld", "rows": [0, 4]},
                {"domain_name": "camera_pose", "rows": [4, 6]},
                {"domain_name": "agibotworld", "rows": [6, 8]},
            ],
            "num_frames": NUM_FRAMES, "fps": 30, "seed": 2026,
            "canvas": {"height": 480, "width": 832}, "output": {"height": 352, "width": 640},
            "action_mode": "forward_dynamics", "domain_name": "agibotworld", "action_normalization": "none",
            "action_view_point": "ego_view", "format_prompt_as_json": False,
        })
        # Without format_prompt_as_json the server's export default applies, so the field is not sent.
        record.pop("format_prompt_as_json")
        self.assertNotIn("format_prompt_as_json", prepare_bimanual_input(record, base_dir=self.root).extra_args)

    def test_integral_floats_are_accepted_for_integer_fields(self):
        # The published manifest rows carry fps: 30; a record written with 30.0 names the same clock.
        prepared = self.prepare(num_frames=float(NUM_FRAMES), fps=30.0, seed=2026.0, height=480.0, width=832.0)
        for value, expected in (
            (prepared.num_frames, NUM_FRAMES), (prepared.fps, 30), (prepared.seed, 2026),
            (prepared.height, 480), (prepared.width, 832),
        ):
            self.assertIs(type(value), int)
            self.assertEqual(value, expected)
        self.assertEqual(json.dumps(prepared.metadata["fps"]), "30")
        self.assertEqual(self.prepare(seed=0).seed, 0)
        for field, value, message in (
            ("fps", 29.97, "fps must be an integer >= 1"),
            ("fps", True, "fps must be an integer >= 1"),
            ("fps", "30", "fps must be an integer >= 1"),
            ("fps", float("nan"), "fps must be an integer >= 1"),
            ("fps", float("inf"), "fps must be an integer >= 1"),
            ("fps", 0, "fps must be an integer >= 1"),
            ("seed", -1, "seed must be an integer >= 0"),
            ("num_frames", 1, "num_frames must be an integer >= 2"),
            ("num_frames", 8, r"num_frames = 4k\+1"),
            ("action_chunk_size", 7.5, "action_chunk_size must be an integer >= 1"),
            ("height", 480.5, "height must be an integer >= 1"),
        ):
            overrides = {field: value} if field != "height" else {"height": value, "width": 832}
            with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, message):
                self.prepare(**overrides)

    def test_action_rows_must_match_the_frame_count_and_unified_width(self):
        with self.assertRaisesRegex(ValueError, rf"Expected 12x{UNIFIED_ACTION_DIM} unified actions, got \(8, {UNIFIED_ACTION_DIM}\)"):
            self.prepare(num_frames=13)
        self.write_sidecar("narrow.json", [row[:29] for row in self.rows])
        with self.assertRaisesRegex(ValueError, r"got \(8, 29\)"):
            self.prepare(action_path="narrow.json")
        with self.assertRaisesRegex(ValueError, "action_chunk_size must equal num_frames - 1"):
            self.prepare(action_chunk_size=NUM_FRAMES)
        self.assertEqual(self.prepare(action_chunk_size=NUM_FRAMES - 1).num_frames, NUM_FRAMES)
        with self.assertRaisesRegex(ValueError, f"raw_action_dim must be {UNIFIED_ACTION_DIM}"):
            self.prepare(raw_action_dim=29)
        self.write_sidecar("overflow.json", [[1e39, *row[1:]] for row in self.rows])  # Finite in float64, inf in float32.
        with self.assertRaisesRegex(ValueError, "finite in float32"):
            self.prepare(action_path="overflow.json")

    def test_native_recipe_keys_inline_rows_and_contract_mismatches_reject(self):
        for overrides, message in (
            ({"camera_trajectory": []}, "earlier 29-value recipe"),
            ({"camera_pose_convention": "backward_chunk_anchored_16f"}, "earlier 29-value recipe"),
            ({"camera_translation_scale": 1.0}, "earlier 29-value recipe"),
            ({"camera_action_normalization": "none"}, "earlier 29-value recipe"),
            ({"action_space": UNIFIED_ACTION_SPACE}, "earlier 29-value recipe"),
            ({"action_normalization": "global_asinh_unified_v1"}, "action_normalization must be 'none'"),
            ({"action": self.rows}, "inline rows are not supported"),
            ({"actions": self.rows}, "inline rows are not supported"),
            ({"model_mode": "inverse_dynamics"}, "forward dynamics"),
            ({"autoregressive": False}, "autoregressive"),
            ({"resolution": "720"}, "480p canvases"),
            ({"resolution": 720}, "480p canvases"),
            ({"view_point": "top_view"}, "Unsupported view_point: 'top_view'"),
            ({"prompt": ["Place the cup."]}, "prompt must be a string"),
            ({"format_prompt_as_json": "false"}, "format_prompt_as_json must be a boolean"),
            ({"domain_name": 15}, "domain_name must be an embodiment name"),
            ({"height": 480}, "height and width must be supplied together"),
            ({"width": 832}, "height and width must be supplied together"),
            ({"height": 720, "width": 1280}, "trained 480p canvas"),
            ({"height": 544, "width": 544}, "trained 480p canvas"),  # Mixes the sides of two canvases.
        ):
            with self.subTest(**overrides), self.assertRaisesRegex(ValueError, message):
                self.prepare(**overrides)
        # Accepted spellings of the same contract.
        for overrides in (
            {"resolution": 480}, {"model_mode": "forward_dynamics"}, {"autoregressive": True},
            {"action_normalization": "none"}, {"action_condition": True, "aspect_ratio": "16,9", "num_steps": 4},
            {"view_point": "third_person_view"}, {"format_prompt_as_json": True},
        ):
            with self.subTest(**overrides):
                self.assertEqual(self.prepare(**overrides).num_frames, NUM_FRAMES)
        without_resolution = {key: value for key, value in self.record.items() if key != "resolution"}
        self.assertEqual(prepare_bimanual_input(without_resolution, base_dir=self.root).height, 480)  # Defaults to 480.
        self.assertEqual(prepare_bimanual_input({**without_resolution, "image_size": "480"}, base_dir=self.root).height, 480)
        with self.assertRaisesRegex(ValueError, "480p canvases"):
            prepare_bimanual_input({**without_resolution, "image_size": 720}, base_dir=self.root)

    def test_canvas_follows_the_image_aspect_unless_a_trained_canvas_is_named(self):
        for size, canvas, output in (
            ((640, 360), (832, 480), (640, 352)),  # Published example: 16:9 content, cropped to 16-pixel multiples.
            ((1920, 1080), (832, 480), (832, 464)),  # Larger than the canvas: scaled down, height rounded to 16.
            ((360, 640), (480, 832), (352, 640)),  # Portrait.
            ((500, 500), (640, 640), (496, 496)),  # Square.
            ((1472, 1088), (736, 544), (736, 544)),  # Twice the 4:3-ish canvas fills it exactly.
        ):
            with self.subTest(size=size):
                name = f"image_{size[0]}x{size[1]}.png"
                Image.new("RGB", size).save(self.root / name)
                prepared = self.prepare(vision_path=name)
                self.assertEqual((prepared.width, prepared.height), canvas)
                self.assertEqual((prepared.output_width, prepared.output_height), output)
                self.assertEqual(prepared.metadata["canvas"], {"height": canvas[1], "width": canvas[0]})
        for width, height in CANVASES:
            with self.subTest(canvas=(width, height)):
                prepared = self.prepare(height=height, width=width)
                self.assertEqual((prepared.width, prepared.height), (width, height))
        prepared = self.prepare(height=544, width=736)  # A named canvas never upscales the image.
        self.assertEqual(self.prepare(height=832, width=480).metadata["canvas"], {"height": 832, "width": 480})  # Portrait.
        self.assertEqual((prepared.output_width, prepared.output_height), (640, 352))

    def test_sidecar_domain_names_broadcast_or_fall_back_to_the_record(self):
        expected_rows = np.asarray(self.rows)
        single = self.write_sidecar("single.json", self.rows, "camera_pose")
        rows, names = load_action_sidecar(single)
        np.testing.assert_array_equal(rows, expected_rows)
        self.assertEqual(names, ["camera_pose"] * (NUM_FRAMES - 1))
        one = self.write_sidecar("one.json", self.rows, [" agibotworld "])
        self.assertEqual(load_action_sidecar(one)[1], ["agibotworld"] * (NUM_FRAMES - 1))  # Stripped, then broadcast.
        unnamed = self.write_sidecar("unnamed.json", self.rows, None)
        self.assertEqual(load_action_sidecar(unnamed, default_domain="agibotworld")[1], ["agibotworld"] * (NUM_FRAMES - 1))
        with self.assertRaisesRegex(ValueError, "no domain_names"):
            load_action_sidecar(unnamed)
        bare = self.root / "bare.json"
        bare.write_text(json.dumps(self.rows))
        rows, names = load_action_sidecar(bare, default_domain="agibotworld")
        np.testing.assert_array_equal(rows, expected_rows)
        self.assertEqual(names, ["agibotworld"] * (NUM_FRAMES - 1))
        with self.assertRaisesRegex(ValueError, "bare action array needs domain_name"):
            load_action_sidecar(bare)
        # Through a record: domain_name fills in missing names and labels the request.
        prepared = self.prepare(action_path="unnamed.json", domain_name="agibotworld")
        self.assertEqual(prepared.extra_args["domain_name"], "agibotworld")
        self.assertEqual(prepared.metadata["domain_runs"], [{"domain_name": "agibotworld", "rows": [0, NUM_FRAMES - 1]}])
        with self.assertRaisesRegex(ValueError, "no domain_names"):
            self.prepare(action_path="unnamed.json")
        prepared = self.prepare(domain_name="camera_pose")  # The sidecar's per-row names still route every row.
        self.assertEqual((prepared.extra_args["domain_name"], prepared.domain_names), ("camera_pose", DOMAINS))

    def test_sidecar_contract_errors_name_the_file_and_numeric_errors_name_the_actions(self):
        unified = {"action_space": UNIFIED_ACTION_SPACE, "action": self.rows, "domain_names": DOMAINS}
        for name, payload, message in (
            ("space.json", {**unified, "action_space": "native_v0"}, "action_space must be 'normalized_unified_v1', got 'native_v0'"),
            ("none.json", {**unified, "action_space": None}, "got None"),
            ("missing.json", {"action_space": UNIFIED_ACTION_SPACE, "domain_names": DOMAINS}, "needs an `action` list"),
            ("flat.json", {**unified, "action": self.rows[0]}, "actions must be a list of rows"),
            ("count.json", {**unified, "domain_names": DOMAINS[:3]}, "3 domain names for 8 action rows"),
            ("blank.json", {**unified, "domain_names": ["agibotworld", " "] * 4}, "non-empty list of embodiment names"),
            ("typed.json", {**unified, "domain_names": [15] * 8}, "non-empty list of embodiment names"),
            ("empty.json", {**unified, "domain_names": []}, "non-empty list of embodiment names"),
        ):
            with self.subTest(file=name):
                path = self.root / name
                path.write_text(json.dumps(payload))
                with self.assertRaisesRegex(ValueError, message) as raised:
                    load_action_sidecar(path)
                self.assertIn(str(path), str(raised.exception))
        for name, first, message in (
            ("bool.json", True, "actions must contain real numbers"),
            ("text.json", "0.5", "actions must contain real numbers"),
            ("nan.json", float("nan"), "actions must contain only finite values"),
            ("inf.json", float("-inf"), "actions must contain only finite values"),
        ):
            with self.subTest(file=name):
                path = self.write_sidecar(name, [[first, *self.rows[0][1:]], *self.rows[1:]])
                with self.assertRaisesRegex(ValueError, message):
                    load_action_sidecar(path)
        scalar = self.write_sidecar("scalar.json", "0.5", "agibotworld")  # A string where the rows should be.
        with self.assertRaisesRegex(ValueError, "actions must contain real numbers"):
            load_action_sidecar(scalar)

    def test_domain_runs_segments_consecutive_embodiments(self):
        self.assertEqual(domain_runs([]), [])
        self.assertEqual(domain_runs(["agibotworld"]), [("agibotworld", 0, 1)])
        self.assertEqual(domain_runs(DOMAINS), [("agibotworld", 0, 4), ("camera_pose", 4, 6), ("agibotworld", 6, 8)])

    def test_inputs_must_exist_be_non_empty_and_hydrated(self):
        with self.assertRaisesRegex(FileNotFoundError, "Missing or empty input"):
            self.prepare(action_path="absent.json")
        (self.root / "empty.json").write_bytes(b"")
        with self.assertRaisesRegex(FileNotFoundError, "Missing or empty input"):
            self.prepare(action_path="empty.json")
        (self.root / "pointer.png").write_bytes(b"version https://git-lfs.github.com/spec/v1\noid sha256:0\nsize 1\n")
        with self.assertRaisesRegex(ValueError, "Unhydrated Git LFS pointer"):
            self.prepare(vision_path="pointer.png")
        for value in (None, "", "   ", 7):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "Supply a local"):
                self.prepare(vision_path=value)
        # Relative paths resolve against base_dir; absolute paths are used as given.
        self.assertEqual(self.prepare(vision_path=str(self.image_path)).metadata["vision_path"], str(self.image_path))
        nested = self.root / "nested"
        nested.mkdir()
        self.assertEqual(
            prepare_bimanual_input({**self.record, "vision_path": "../initial.png", "action_path": "../actions.json"}, base_dir=nested).image.size,
            (640, 360),
        )

    def test_first_media_image_honors_exif_orientation_and_decodes_the_first_video_frame(self):
        rotated = self.root / "rotated.jpg"
        exif = Image.Exif()
        exif[0x0112] = 6  # Orientation: rotate 90 degrees clockwise for display.
        Image.new("RGB", (40, 20)).save(rotated, exif=exif)
        self.assertEqual(first_media_image(rotated).size, (20, 40))
        self.assertEqual((first_media_image(self.image_path).size, first_media_image(self.image_path).mode), ((640, 360), "RGB"))
        if not all(shutil.which(name) for name in ("ffmpeg", "ffprobe")):
            self.skipTest("Install ffmpeg to decode video conditioning frames.")
        with self.assertRaises((subprocess.CalledProcessError, ValueError)):
            first_media_image(self.sidecar_path)  # Neither an image nor a video.
        video = self.root / "clip.mp4"
        width, height = 64, 48
        frames = b"".join(bytes([level]) * (width * height) for level in (200, 20, 20))
        subprocess.run(
            [
                "ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pixel_format", "gray",
                "-video_size", f"{width}x{height}", "-framerate", "30", "-i", "pipe:0",
                "-an", "-c:v", "libx264", "-crf", "0", "-pix_fmt", "yuv420p", str(video),
            ],
            input=frames, capture_output=True, check=True,
        )
        frame = first_media_image(video)
        self.assertEqual((frame.size, frame.mode), ((width, height), "RGB"))
        self.assertAlmostEqual(float(np.asarray(frame).mean()), 200, delta=4)  # The first frame, not a later dark one.
        prepared = self.prepare(vision_path=video.name)
        self.assertEqual((prepared.width, prepared.height), (736, 544))  # 4:3 lands on the 736x544 canvas.
        self.assertEqual((prepared.output_width, prepared.output_height), (width, height))


class EncodingTests(BimanualFixture):
    def encodings(self, prepared: BimanualInput) -> dict[str, str]:
        """The extra_params text from orjson, when installed, and from the standard-library fallback."""
        texts = {}
        if orjson is not None:
            texts["orjson"] = encode_extra_params(prepared)
        with patch.dict(sys.modules, {"orjson": None}):  # Makes ``import orjson`` raise ImportError.
            texts["json"] = encode_extra_params(prepared)
        return texts

    def test_extra_params_round_trip_float32_rows_in_compact_form(self):
        prepared = self.prepare()
        naive = json.dumps({**prepared.extra_args, "action": prepared.action.astype(float).tolist()}, separators=(",", ":"))
        for encoder, text in self.encodings(prepared).items():
            with self.subTest(encoder=encoder):
                payload = json.loads(text)
                self.assertEqual({key: value for key, value in payload.items() if key != "action"}, prepared.extra_args)
                np.testing.assert_array_equal(np.asarray(payload["action"], dtype=np.float32), prepared.action)
                self.assertNotIn(" ", text)
                self.assertLess(len(text), 0.75 * len(naive))  # Shortest float32 decimals rather than double repr.

    def test_published_rollout_length_stays_within_the_form_field_limit(self):
        # 900 rows alternating between the robot and camera domains, the length of the published mixed-action example.
        frames = 901
        names = ["agibotworld"] * 300 + ["camera_pose"] * 300 + ["agibotworld"] * 300
        self.write_sidecar("long.json", make_rows(frames - 1, seed=1), names)
        prepared = self.prepare(num_frames=frames, action_path="long.json")
        self.assertEqual(prepared.metadata["domain_runs"], [
            {"domain_name": "agibotworld", "rows": [0, 300]}, {"domain_name": "camera_pose", "rows": [300, 600]},
            {"domain_name": "agibotworld", "rows": [600, 900]},
        ])
        naive = json.dumps({**prepared.extra_args, "action": prepared.action.astype(float).tolist()}, separators=(",", ":"))
        self.assertGreater(len(naive.encode()), bimanual_helpers.MAX_FORM_FIELD_BYTES)  # Double repr would be rejected.
        for encoder, text in self.encodings(prepared).items():
            with self.subTest(encoder=encoder):
                self.assertLessEqual(len(text.encode()), bimanual_helpers.MAX_FORM_FIELD_BYTES)
                np.testing.assert_array_equal(np.asarray(json.loads(text)["action"], dtype=np.float32), prepared.action)
        with patch.object(bimanual_helpers, "MAX_FORM_FIELD_BYTES", 4096), self.assertRaisesRegex(ValueError, "multipart-field"):
            encode_extra_params(prepared)

    def test_lossy_serialization_is_detected_before_sending(self):
        def truncate(value, **kwargs):
            return f"{float(value):.3f}"

        prepared = self.prepare()
        with (
            patch.dict(sys.modules, {"orjson": None}),
            patch.object(np, "format_float_positional", truncate),
            self.assertRaisesRegex(ValueError, "preserve every float32 value"),
        ):
            encode_extra_params(prepared)


class SubmitTests(BimanualFixture):
    def setUp(self):
        super().setUp()
        self.prepared = self.prepare(fps=30.0)  # An integral float travels as the form field "30".
        self.output = self.root / "out" / "sample" / "sample.mp4"
        self.requests = []
        self.on_submit = lambda: httpx.Response(200, json={"id": "job-7", "status": "queued"})
        self.polls = iter([{"id": "job-7", "status": "completed"}])
        self.on_content = lambda: httpx.Response(200, content=b"mock video", headers={"content-type": "video/mp4"})

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append((request.method, request.url.path))
        if (request.method, request.url.path) == ("POST", "/v1/videos"):
            self.submission = request
            return self.on_submit()
        if (request.method, request.url.path) == ("GET", "/v1/videos/job-7"):
            item = next(self.polls)
            return item if isinstance(item, httpx.Response) else httpx.Response(200, json=item)
        if (request.method, request.url.path) == ("GET", "/v1/videos/job-7/content"):
            return self.on_content()
        raise AssertionError(f"Unexpected request {request.method} {request.url}")

    def submit(self, **kwargs):
        with (
            patch("time.sleep") as sleep,
            contextlib.redirect_stdout(io.StringIO()) as stdout,
            httpx.Client(base_url="http://test", transport=httpx.MockTransport(self.handler)) as client,
        ):
            result = submit_bimanual(client, self.prepared, MODEL, self.output, **kwargs)
        return result, sleep, stdout.getvalue()

    def test_multipart_submission_polls_the_job_and_streams_the_video_with_sidecars(self):
        self.polls = iter([{"id": "job-7", "status": "in_progress", "progress": 50}, {"id": "job-7", "status": "completed", "progress": 100}])
        result, sleep, printed = self.submit(timeout=120, poll_interval=5)
        self.assertEqual(result, self.output)
        self.assertEqual(self.output.read_bytes(), b"mock video")
        self.assertIn("Submitted job-7 at http://test/v1/videos/job-7", printed)
        self.assertEqual(self.requests, [
            ("POST", "/v1/videos"), ("GET", "/v1/videos/job-7"), ("GET", "/v1/videos/job-7"), ("GET", "/v1/videos/job-7/content"),
        ])
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [5, 5])

        parts = parse_multipart(self.submission)
        form = {name: body.decode() for name, (_, _, body) in parts.items() if name != "input_reference"}
        expected_form = {
            "model": MODEL, "prompt": "Place the cup in the sink.", "size": "832x480",
            "num_frames": "9", "fps": "30", "seed": "2026", "extra_params": encode_extra_params(self.prepared),
        }
        self.assertEqual(form, expected_form)
        for name in ("num_frames", "fps", "seed"):  # Integer text: SGLang parses these fields as integers.
            self.assertTrue(form[name].isdigit(), form[name])
        filename, content_type, png = parts["input_reference"]
        self.assertEqual((filename, content_type), ("conditioning.png", "image/png"))
        with Image.open(io.BytesIO(png)) as image:
            self.assertEqual((image.format, image.size, image.getpixel((0, 0))), ("PNG", (640, 360), (40, 80, 120)))
        extra = json.loads(form["extra_params"])
        self.assertEqual({key: value for key, value in extra.items() if key != "action"}, self.prepared.extra_args)
        np.testing.assert_array_equal(np.asarray(extra["action"], dtype=np.float32), self.prepared.action)

        self.assertEqual(json.loads(self.output.with_suffix(".request.json").read_text()), expected_form)
        self.assertEqual(json.loads(self.output.with_suffix(".metadata.json").read_text()), self.prepared.metadata)
        self.assertEqual(json.loads(self.output.with_suffix(".job.json").read_text()), {"id": "job-7", "status": "completed", "progress": 100})
        self.assertEqual(
            sorted(path.name for path in self.output.parent.iterdir()),
            ["sample.job.json", "sample.metadata.json", "sample.mp4", "sample.request.json"],
        )

    def test_a_job_completed_at_submission_needs_no_polling(self):
        self.on_submit = lambda: httpx.Response(200, json={"id": "job-7", "status": "completed"})
        self.polls = iter([])
        result, sleep, _ = self.submit()
        sleep.assert_not_called()
        self.assertEqual(result.read_bytes(), b"mock video")
        self.assertEqual([path for _, path in self.requests], ["/v1/videos", "/v1/videos/job-7/content"])

    def test_rejected_submissions_failed_jobs_and_poll_errors_save_the_error_and_raise(self):
        error_path = self.output.with_suffix(".error.txt")
        self.on_submit = lambda: httpx.Response(400, text="mock submission rejected")
        with self.assertRaises(httpx.HTTPStatusError):
            self.submit()
        self.assertEqual(error_path.read_text(), "mock submission rejected")
        self.assertFalse(self.output.exists())
        self.assertTrue(self.output.with_suffix(".request.json").exists())  # Saved before sending, for replay.

        self.on_submit = lambda: httpx.Response(200, json={"id": "job-7", "status": "queued"})
        self.polls = iter([{"id": "job-7", "status": "failed", "error": {"message": "mock rollout failed"}}])
        with self.assertRaisesRegex(RuntimeError, "Video generation failed: {'message': 'mock rollout failed'}"):
            self.submit()
        self.assertEqual(json.loads(error_path.read_text())["error"], {"message": "mock rollout failed"})
        self.assertEqual(json.loads(self.output.with_suffix(".job.json").read_text())["status"], "failed")
        self.assertNotIn(("GET", "/v1/videos/job-7/content"), self.requests)

        for status, message in (("cancelled", "Video generation cancelled"), ("exploded", "Unexpected video job status")):
            with self.subTest(status=status):
                self.polls = iter([{"id": "job-7", "status": status}])
                with self.assertRaisesRegex(RuntimeError, message):
                    self.submit()
        self.assertFalse(self.output.exists())

        self.polls = iter([httpx.Response(503, text="mock poll outage")])
        with self.assertRaises(httpx.HTTPStatusError):
            self.submit()
        self.assertEqual(error_path.read_text(), "mock poll outage")

    def test_download_failures_leave_no_partial_video(self):
        self.on_submit = lambda: httpx.Response(200, json={"id": "job-7", "status": "completed"})
        self.on_content = lambda: httpx.Response(500, text="mock download failed")
        with self.assertRaises(httpx.HTTPStatusError):
            self.submit()
        self.assertEqual(self.output.with_suffix(".error.txt").read_text(), "mock download failed")
        self.on_content = lambda: httpx.Response(200, content=b"")
        with self.assertRaisesRegex(ValueError, "empty"):
            self.submit()
        self.assertEqual([path.name for path in self.output.parent.iterdir() if ".mp4" in path.name], [])

    def test_polling_timeout_raises_and_leaves_the_job_running(self):
        self.polls = itertools.repeat({"id": "job-7", "status": "in_progress"})
        clock = SimpleNamespace(monotonic=Mock(side_effect=itertools.count(100.0, 0.3)), sleep=Mock())
        with (
            patch.object(bimanual_helpers, "time", clock),
            contextlib.redirect_stdout(io.StringIO()),
            httpx.Client(base_url="http://test", transport=httpx.MockTransport(self.handler)) as client,
            self.assertRaisesRegex(TimeoutError, "job remains at http://test/v1/videos/job-7"),
        ):
            submit_bimanual(client, self.prepared, MODEL, self.output, timeout=1, poll_interval=5)
        # Deadline 101.0: sleeps shrink to the remaining time, one poll fits, then the deadline passes.
        self.assertEqual([round(call.args[0], 6) for call in clock.sleep.call_args_list], [0.7, 0.1])
        self.assertEqual(self.requests, [("POST", "/v1/videos"), ("GET", "/v1/videos/job-7")])  # No cancellation.
        self.assertEqual(json.loads(self.output.with_suffix(".job.json").read_text())["status"], "in_progress")
        self.assertFalse(self.output.with_suffix(".error.txt").exists())

    def test_polling_parameters_must_be_positive_and_finite(self):
        for kwargs in (
            {"timeout": 0}, {"timeout": -5}, {"timeout": float("inf")}, {"timeout": float("nan")},
            {"poll_interval": 0}, {"poll_interval": float("inf")},
        ):
            with (
                self.subTest(**kwargs),
                httpx.Client(base_url="http://test", transport=httpx.MockTransport(self.handler)) as client,
                self.assertRaisesRegex(ValueError, "positive and finite"),
            ):
                submit_bimanual(client, self.prepared, MODEL, self.output, **kwargs)
        self.assertEqual(self.requests, [])
        self.assertFalse(self.output.parent.exists())


class NotebookContractTests(unittest.TestCase):
    def test_notebook_imports_names_the_helper_module_defines(self):
        notebook = json.loads((BIMANUAL_ROOT / "run_bimanual_with_sglang.ipynb").read_text())
        imported = set()
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            for node in ast.walk(ast.parse("".join(cell["source"]))):
                if isinstance(node, ast.ImportFrom) and node.module == "bimanual_helpers":
                    imported.update(alias.name for alias in node.names)
        self.assertLessEqual({"domain_runs", "prepare_bimanual_input", "submit_bimanual"}, imported)
        for name in sorted(imported):
            self.assertTrue(callable(getattr(bimanual_helpers, name, None)), name)


if __name__ == "__main__":
    unittest.main()
