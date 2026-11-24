import unittest
from unittest import mock

import numpy as np
import pytest
from cosmos_auto.wm import minimap_utils
from cosmos_auto.wm.pcd_utils import refine_triangle_mesh_3d


def triangle_area_xy(triangle):
    a, b, c = triangle[:, :2]
    ab = b - a
    ac = c - a
    return 0.5 * abs(ab[0] * ac[1] - ab[1] * ac[0])


def maximum_edge_length(triangles):
    edges = triangles[:, ((0, 1), (1, 2), (2, 0)), :]
    return float(np.linalg.norm(edges[:, :, 0] - edges[:, :, 1], axis=2).max())


def road_marking_label(road_marking_class, vertices):
    return {
        "labelData": {
            "shape3d": {
                "surface": {"vertices": np.asarray(vertices, dtype=float).tolist()},
                "attributes": [{"name": "road_marking_class", "text": road_marking_class}],
            }
        }
    }


@pytest.mark.L0
@pytest.mark.CPU
class TriangleMeshRefinementTest(unittest.TestCase):
    def setUp(self):
        self.mesh = np.array(
            [
                [[0.0, 0.0, 5.0], [2.0, 0.0, 5.2], [0.0, 2.0, 5.4]],
                [[2.0, 0.0, 5.2], [2.0, 2.0, 5.6], [0.0, 2.0, 5.4]],
            ]
        )

    def test_refinement_preserves_surface_and_has_no_t_junctions(self):
        refined = refine_triangle_mesh_3d(self.mesh, max_edge_length=0.75)

        self.assertLessEqual(maximum_edge_length(refined), 0.75 + 1e-10)
        self.assertAlmostEqual(
            sum(map(triangle_area_xy, refined)),
            sum(map(triangle_area_xy, self.mesh)),
            places=10,
        )
        np.testing.assert_allclose(
            refined[:, :, 2],
            5.0 + 0.1 * refined[:, :, 0] + 0.2 * refined[:, :, 1],
            atol=1e-12,
        )

        edge_counts = {}
        for triangle in refined:
            for a, b in ((0, 1), (1, 2), (2, 0)):
                endpoints = tuple(
                    sorted(
                        (
                            tuple(np.round(triangle[a], decimals=12)),
                            tuple(np.round(triangle[b], decimals=12)),
                        )
                    )
                )
                edge_counts[endpoints] = edge_counts.get(endpoints, 0) + 1

        for endpoints, count in edge_counts.items():
            self.assertIn(count, (1, 2))
            if count == 1:
                midpoint = 0.5 * (np.asarray(endpoints[0]) + np.asarray(endpoints[1]))
                on_outer_boundary = (
                    np.isclose(midpoint[0], 0.0)
                    or np.isclose(midpoint[0], 2.0)
                    or np.isclose(midpoint[1], 0.0)
                    or np.isclose(midpoint[1], 2.0)
                )
                self.assertTrue(
                    on_outer_boundary,
                    f"interior edge appears only once: {endpoints}",
                )

    def test_refinement_is_noop_when_edges_already_fit(self):
        refined = refine_triangle_mesh_3d(self.mesh, max_edge_length=3.0)
        np.testing.assert_allclose(refined, self.mesh)

    def test_refinement_rejects_invalid_limit(self):
        for limit in (0.0, -1.0, np.inf, np.nan):
            with self.subTest(limit=limit):
                with self.assertRaises(ValueError):
                    refine_triangle_mesh_3d(self.mesh, max_edge_length=limit)


@pytest.mark.L0
@pytest.mark.CPU
class WMV4RoadMarkingIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.keep_clear = road_marking_label(
            "ROI_POLYGON_KEEP_CLEAR",
            [[0, 0, 10.0], [10, 0, 10.1], [10, 4, 10.2], [0, 4, 10.1]],
        )
        self.text_paint = road_marking_label(
            "ROI_POLYGON_TEXT_PAINT",
            [[2, 1, 10.03], [4, 1, 10.05], [4, 2, 10.07], [2, 2, 10.05]],
        )
        self.sample = {"road_markings.json": {"labels": [self.keep_clear, self.text_paint]}}

    def prepare(self, **kwargs):
        defaults = {
            "use_arrow_shapes": False,
            "curve_subdivide": 10,
        }
        defaults.update(kwargs)
        with (
            mock.patch.object(minimap_utils.os.path, "exists", return_value=True),
            mock.patch.object(minimap_utils, "get_sample", return_value=self.sample),
            mock.patch.object(
                minimap_utils,
                "load_road_marking_colors",
                return_value={
                    "keep_clear": [40, 220, 80],
                    "stop": [220, 40, 40],
                    "forward": [220, 220, 40],
                    "merge_left": [244, 244, 244],
                    "merge_right": [244, 244, 244],
                },
            ),
            mock.patch.object(
                minimap_utils,
                "load_hdmap_colors",
                return_value={"road_markings": [220, 220, 220]},
            ),
        ):
            return minimap_utils.prepare_road_marking_data_v4("road_markings.tar", **defaults)

    def test_tessellation_toggle_and_keep_clear_bias(self):
        baseline = self.prepare(
            use_interior_tessellation=False,
            interior_tessellation_max_edge_m=1.0,
        )
        enabled = self.prepare(
            use_interior_tessellation=True,
            interior_tessellation_max_edge_m=1.0,
        )

        self.assertEqual(len(baseline), 2)
        self.assertEqual(len(enabled), 2)
        self.assertGreater(len(enabled[0]["tris_world"]), len(baseline[0]["tris_world"]))
        self.assertLessEqual(maximum_edge_length(enabled[0]["tris_world"]), 1.0 + 1e-10)
        self.assertEqual(enabled[0]["geometry_kind"], "footprint")
        self.assertTrue(enabled[0]["interior_tessellated"])
        self.assertEqual(baseline[0]["camera_depth_bias_m"], 0.10)
        self.assertEqual(enabled[0]["camera_depth_bias_m"], 0.10)
        self.assertEqual(enabled[1]["camera_depth_bias_m"], 0.0)

    def test_roadnet_mask_is_excluded_from_rendering(self):
        roadnet_mask = road_marking_label(
            "ROI_POLYGON_ROADNET_MASK",
            [[0, 0, 10], [20, 0, 10], [20, 10, 10], [0, 10, 10]],
        )
        self.sample = {"road_markings.json": {"labels": [self.keep_clear, roadnet_mask, self.text_paint]}}
        prepared = self.prepare(use_interior_tessellation=False)

        self.assertEqual(len(prepared), 2)
        self.assertEqual([marking["label_index"] for marking in prepared], [0, 2])
        self.assertNotIn(
            "ROI_POLYGON_ROADNET_MASK",
            {marking["road_marking_class"] for marking in prepared},
        )

    def test_single_real_label_is_not_treated_as_empty(self):
        self.sample = {"road_markings.json": {"labels": [self.keep_clear]}}
        prepared = self.prepare(use_interior_tessellation=False)
        self.assertEqual(len(prepared), 1)
        self.assertEqual(prepared[0]["road_marking_class"], "ROI_POLYGON_KEEP_CLEAR")

    def test_empty_sentinel_has_no_geometry(self):
        self.sample = {"road_markings.json": {"labels": [{"labelFamily": "EMPTY", "labelData": {}}]}}
        self.assertEqual(self.prepare(use_interior_tessellation=False), [])

    def test_tessellation_only_refines_keep_clear_and_text_paint(self):
        stop = road_marking_label(
            "ROI_POLYGON_ROAD_MARKING_STOP",
            [[0, 0, 10], [8, 0, 10], [8, 3, 10], [0, 3, 10]],
        )
        self.sample = {"road_markings.json": {"labels": [self.keep_clear, self.text_paint, stop]}}
        baseline = self.prepare(use_interior_tessellation=False)
        enabled = self.prepare(
            use_interior_tessellation=True,
            interior_tessellation_max_edge_m=1.0,
        )

        self.assertTrue(enabled[0]["interior_tessellated"])
        self.assertTrue(enabled[1]["interior_tessellated"])
        self.assertFalse(enabled[2]["interior_tessellated"])
        self.assertGreater(len(enabled[0]["tris_world"]), len(baseline[0]["tris_world"]))
        self.assertGreater(len(enabled[1]["tris_world"]), len(baseline[1]["tris_world"]))
        self.assertEqual(len(enabled[2]["tris_world"]), len(baseline[2]["tris_world"]))

    def test_glyph_mesh_is_not_tessellated_or_biased(self):
        arrow = road_marking_label(
            "ROI_POLYGON_DIRECTION_PAINT_FORWARD",
            [[0, 0, 10], [4, 0, 10], [4, 2, 10], [0, 2, 10]],
        )
        glyph = np.array([[[0, 0, 10], [5, 0, 10], [0, 5, 10]]], dtype=float)
        self.sample = {"road_markings.json": {"labels": [arrow, self.text_paint]}}
        with mock.patch.object(
            minimap_utils,
            "_wmv4_road_marking_arrow_world_triangles",
            return_value=glyph,
        ):
            prepared = self.prepare(
                use_arrow_shapes=True,
                use_interior_tessellation=True,
                interior_tessellation_max_edge_m=1.0,
            )

        self.assertEqual(prepared[0]["geometry_kind"], "arrow_glyph")
        self.assertFalse(prepared[0]["interior_tessellated"])
        np.testing.assert_array_equal(prepared[0]["tris_world"], glyph)
        self.assertEqual(prepared[0]["camera_depth_bias_m"], 0.0)

    def test_arrow_forward_uses_vertex_order_not_longest_axis(self):
        travel_axis_longer = np.array([[0, 0, 0], [0, 2, 0], [5, 2, 0], [5, 0, 0]], dtype=float)
        lateral_axis_longer = np.array([[0, 0, 0], [0, 5, 0], [2, 5, 0], [2, 0, 0]], dtype=float)

        for quad in (travel_axis_longer, lateral_axis_longer):
            with self.subTest(quad=quad.tolist()):
                np.testing.assert_allclose(
                    minimap_utils.wmv4_road_marking_arrow_forward(quad),
                    [-1.0, 0.0, 0.0],
                )

    def test_merge_classes_render_as_single_arrow_glyphs(self):
        merge_left = road_marking_label(
            "ROI_POLYGON_DIRECTION_PAINT_MERGE_LEFT",
            [[0, 0, 10], [6, 0, 10], [6, 2, 10], [0, 2, 10]],
        )
        merge_right = road_marking_label(
            "ROI_POLYGON_DIRECTION_PAINT_MERGE_RIGHT",
            [[0, 3, 10], [6, 3, 10], [6, 5, 10], [0, 5, 10]],
        )
        self.sample = {"road_markings.json": {"labels": [merge_left, merge_right]}}

        prepared = self.prepare(use_arrow_shapes=True, use_arrow_colors=True)

        self.assertEqual(len(prepared), 2)
        self.assertEqual(
            [marking["geometry_kind"] for marking in prepared],
            ["arrow_glyph", "arrow_glyph"],
        )
        for marking in prepared:
            np.testing.assert_allclose(marking["color"], np.array([244] * 3) / 255.0)

    def test_merge_arrow_templates_are_mirrored(self):
        left = minimap_utils._WMV4_ROAD_MARKING_ARROW_TEMPLATES["merge_left"]
        right = minimap_utils._WMV4_ROAD_MARKING_ARROW_TEMPLATES["merge_right"]

        left_vertices = np.unique(np.round(left.reshape(-1, 2), 10), axis=0)
        mirrored_right = right.reshape(-1, 2).copy()
        mirrored_right[:, 0] *= -1.0
        mirrored_right = np.unique(np.round(mirrored_right, 10), axis=0)
        left_vertices = left_vertices[np.lexsort((left_vertices[:, 1], left_vertices[:, 0]))]
        mirrored_right = mirrored_right[np.lexsort((mirrored_right[:, 1], mirrored_right[:, 0]))]

        np.testing.assert_allclose(left_vertices, mirrored_right)
        self.assertTrue(np.any(np.all(np.isclose(left_vertices, [-0.5, 0.5]), axis=1)))

    def test_projection_applies_bias_only_to_marked_geometry(self):
        prepared = self.prepare(use_interior_tessellation=False)

        class IdentityCamera:
            @staticmethod
            def get_xy_and_depth(points, camera_pose):
                return np.asarray(points, dtype=float)

        geometries = minimap_utils.create_road_marking_geometry_objects_from_data_v4(
            prepared, np.eye(4), IdentityCamera()
        )
        self.assertEqual(len(geometries), 2)
        np.testing.assert_allclose(
            geometries[0].triangles[:, :, 2],
            prepared[0]["tris_world"][:, :, 2] + 0.10,
        )
        np.testing.assert_allclose(
            geometries[1].triangles[:, :, 2],
            prepared[1]["tris_world"][:, :, 2],
        )


if __name__ == "__main__":
    unittest.main()
