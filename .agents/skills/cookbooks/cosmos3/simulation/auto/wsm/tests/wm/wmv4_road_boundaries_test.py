import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
from cosmos_auto.wm import conf as wm_conf
from cosmos_auto.wm import road_boundary_utils
from cosmos_auto.wm.config import CosmosAVConfig
from hydra import compose, initialize_config_dir


def boundary_label(subtype, *, vertices=None, label_family="SHAPE3D"):
    attributes = []
    if subtype is not None:
        attributes.append({"name": "boundary_type", "text": subtype})
    if vertices is None:
        vertices = [[0.0, 0.0, 0.0], [6.0, 0.0, 0.0]]
    return {
        "labelFamily": label_family,
        "labelData": {
            "shape3d": {
                "attributes": attributes,
                "polyline3d": {"vertices": vertices},
            }
        },
    }


@pytest.mark.L0
@pytest.mark.CPU
class WMV4RoadBoundaryColorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.color_config, cls.color_groups_by_subtype = road_boundary_utils._load_color_config()

    def _load_boundaries(self, labels):
        sample = {"road_boundaries.json": {"labels": labels}}
        with mock.patch.object(road_boundary_utils, "get_sample", return_value=sample):
            return road_boundary_utils.load_colored_road_boundaries("unused.tar")

    def test_layer_routing_preserves_the_legacy_path_when_colors_are_disabled(self):
        minimap_types = ["road_boundaries", "crosswalks"]
        routed, render_colored = road_boundary_utils.route_road_boundary_layer(minimap_types, True)
        self.assertEqual(routed, ["crosswalks"])
        self.assertTrue(render_colored)

        routed, render_colored = road_boundary_utils.route_road_boundary_layer(minimap_types, False)
        self.assertEqual(routed, minimap_types)
        self.assertFalse(render_colored)

        routed, render_colored = road_boundary_utils.route_road_boundary_layer(["crosswalks"], True)
        self.assertEqual(routed, ["crosswalks"])
        self.assertFalse(render_colored)

    def test_palette_maps_every_supported_subtype_once(self):
        expected_groups = {
            "solid_single": "other",
            "dot_dashed_single": "other",
            "other_marking": "other",
            "other_physical": "other",
            "barrier": "barrier_or_fence",
            "fence": "barrier_or_fence",
            "curb": "curb",
            "low_curb": "curb",
            "tall_curb": "curb",
            "road_boundary": "road_boundary",
            "obstacle_bars": "obstacle_bars",
            "wall": "wall",
        }
        configured_subtypes = [subtype for settings in self.color_config.values() for subtype in settings["subtypes"]]
        self.assertEqual(len(configured_subtypes), len(set(configured_subtypes)))
        self.assertEqual(self.color_groups_by_subtype, expected_groups)
        self.assertEqual(
            {group: settings["rgb"] for group, settings in self.color_config.items()},
            {
                "other": [239, 191, 255],
                "barrier_or_fence": [253, 1, 232],
                "curb": [207, 255, 191],
                "road_boundary": [0, 159, 96],
                "obstacle_bars": [255, 207, 159],
                "wall": [0, 128, 159],
            },
        )

    def test_color_mode_is_the_only_road_boundary_runtime_option(self):
        fields = CosmosAVConfig.__dataclass_fields__
        self.assertTrue(fields["use_road_boundary_colors"].default)
        self.assertNotIn("use_road_boundary_styles", fields)

        with initialize_config_dir(
            version_base=None,
            config_dir=str(Path(wm_conf.__file__).parent),
        ):
            runtime_config = compose(
                config_name="render",
                overrides=["input_root=/tmp/input", "output_root=/tmp/output"],
            )
        self.assertTrue(runtime_config.use_road_boundary_colors)
        self.assertNotIn("use_road_boundary_styles", runtime_config)

    def test_color_mode_uses_solid_geometry_and_approved_palette(self):
        prepared = self._load_boundaries(
            [
                boundary_label("road_boundary"),
                boundary_label("wall"),
                boundary_label("tall_curb"),
                boundary_label("barrier"),
                boundary_label("obstacle_bars"),
                boundary_label("other_marking"),
            ]
        )
        expected_rgb = [
            [0, 159, 96],
            [0, 128, 159],
            [207, 255, 191],
            [253, 1, 232],
            [255, 207, 159],
            [239, 191, 255],
        ]
        self.assertEqual(len(prepared), 6)
        for boundary, rgb in zip(prepared, expected_rgb):
            self.assertEqual(len(boundary["segments"]), 30)
            np.testing.assert_allclose(boundary["color"], np.array(rgb) / 255.0)

    def test_unknown_and_missing_subtypes_use_other_color(self):
        prepared = self._load_boundaries([boundary_label("unknown"), boundary_label(None)])
        expected = np.array([239, 191, 255]) / 255.0
        for boundary in prepared:
            np.testing.assert_allclose(boundary["color"], expected)

    def test_loader_skips_empty_and_invalid_geometry_and_cleans_vertices(self):
        prepared = self._load_boundaries(
            [
                boundary_label("wall", label_family="EMPTY"),
                boundary_label("wall", vertices=[[0.0, 0.0, 0.0]]),
                boundary_label(
                    "wall",
                    vertices=[[0.0, 0.0, 0.0], [np.nan, 0.0, 0.0]],
                ),
                boundary_label(
                    "wall",
                    vertices=[
                        {"x": 0.0, "y": 0.0, "z": 0.0},
                        {"x": 0.0, "y": 0.0, "z": 0.0},
                        {"x": 1.0, "y": 0.0, "z": 0.0},
                    ],
                ),
            ]
        )
        self.assertEqual(len(prepared), 1)
        np.testing.assert_allclose(
            prepared[0]["polyline"],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        )

    def test_all_color_groups_batch_into_one_filtered_worldspace_object(self):
        prepared = self._load_boundaries(
            [
                boundary_label("road_boundary"),
                boundary_label("wall"),
                boundary_label("tall_curb"),
                boundary_label("barrier"),
                boundary_label("obstacle_bars"),
                boundary_label("other_marking"),
            ]
        )
        with mock.patch.object(road_boundary_utils, "WorldSpaceLines") as line_class:
            objects = road_boundary_utils.build_worldspace_road_boundaries(
                prepared,
                camera_model=object(),
                below_level_filter=True,
            )

        self.assertEqual(objects, [line_class.return_value])
        line_class.assert_called_once()
        kwargs = line_class.call_args.kwargs
        self.assertEqual(kwargs["line_width"], 12.0)
        self.assertTrue(kwargs["below_level_filter"])
        self.assertEqual(kwargs["segments"].shape, (180, 2, 3))
        self.assertEqual(kwargs["colors"].shape, (360, 3))
        for index, boundary in enumerate(prepared):
            start = index * 60
            np.testing.assert_allclose(
                kwargs["colors"][start : start + 60],
                np.tile(boundary["color"], (60, 1)),
            )

    def test_builder_returns_no_draw_call_when_every_boundary_is_filtered(self):
        prepared = self._load_boundaries([boundary_label("wall")])
        with (
            mock.patch.object(
                road_boundary_utils,
                "filter_by_height_relative_to_ego",
                return_value=True,
            ),
            mock.patch.object(road_boundary_utils, "WorldSpaceLines") as line_class,
        ):
            objects = road_boundary_utils.build_worldspace_road_boundaries(
                prepared,
                camera_model=object(),
                camera_pose_init=np.eye(4),
                below_level_filter=False,
            )

        self.assertEqual(objects, [])
        line_class.assert_not_called()


if __name__ == "__main__":
    unittest.main()
