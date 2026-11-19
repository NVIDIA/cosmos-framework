"""Preserve the traffic-light semantics used by the WMv4 rendering preset."""

import numpy as np
import pytest
from cosmos_auto.wm import minimap_utils

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


@pytest.mark.parametrize("direction", ["left", "right", "up"])
@pytest.mark.parametrize(
    ("family", "rgb"),
    [("red", [240, 50, 50]), ("yellow", [255, 200, 0]), ("green", [130, 240, 132]), ("red_yellow", [208, 136, 72])],
)
def test_directional_light_uses_signal_color_and_full_face_triangle(
    direction: str, family: str, rgb: list[int]
) -> None:
    palette = minimap_utils.load_traffic_light_colors("v4")
    housing = np.array(palette["front_housing"]) / 255.0
    state = f"{family}_{direction}_arrow"
    legacy = {state: "red" if family == "red_yellow" else family}
    face, glyph, color = minimap_utils._wmv4_tl_render_spec(state, legacy, palette, housing)

    np.testing.assert_array_equal(face, housing)
    assert glyph == direction
    np.testing.assert_array_equal(color, np.array(rgb) / 255.0)
    pieces = minimap_utils._wmv4_tl_arrow_uv(direction)
    assert len(pieces) == 1
    triangle = np.asarray(pieces[0])
    np.testing.assert_array_equal(triangle.min(axis=0), [0, 0])
    np.testing.assert_array_equal(triangle.max(axis=0), [1, 1])
    np.testing.assert_array_equal(triangle[1], {"left": [0, 0.5], "right": [1, 0.5], "up": [0.5, 1]}[direction])


@pytest.mark.parametrize(
    ("state", "glyph", "rgb"),
    [
        ("red_vru", "x", [240, 50, 50]),
        ("green_vru", "o", [130, 240, 132]),
        ("yellow_vru", "o", [255, 200, 0]),
        ("white_vru", "o", [255, 255, 255]),
    ],
)
def test_pedestrian_light_keeps_glyph_and_signal_color(state: str, glyph: str, rgb: list[int]) -> None:
    palette = minimap_utils.load_traffic_light_colors("v4")
    housing = np.array(palette["front_housing"]) / 255.0
    face, actual_glyph, color = minimap_utils._wmv4_tl_render_spec(state, {}, palette, housing)
    np.testing.assert_array_equal(face, housing)
    assert actual_glyph == glyph
    np.testing.assert_array_equal(color, np.array(rgb) / 255.0)


@pytest.mark.parametrize(
    ("state", "legacy_color", "rgb"),
    [
        ("red_solid", "red", [240, 50, 50]),
        ("yellow_solid", "yellow", [255, 200, 0]),
        ("green_solid", "green", [130, 240, 132]),
        ("red_yellow_solid", "red", [208, 136, 72]),
        ("red_yellow_left_right_arrow", "red", [208, 136, 72]),
        ("stateless", "unknown", [20, 20, 20]),
        ("unknown", "unknown", [150, 150, 150]),
        ("unrecognized_state", "unknown", [150, 150, 150]),
    ],
)
def test_solid_compound_and_unobserved_states_keep_face_colors(state: str, legacy_color: str, rgb: list[int]) -> None:
    palette = minimap_utils.load_traffic_light_colors("v4")
    housing = np.array(palette["front_housing"]) / 255.0
    face, glyph, color = minimap_utils._wmv4_tl_render_spec(state, {state: legacy_color}, palette, housing)
    np.testing.assert_array_equal(face, np.array(rgb) / 255.0)
    assert glyph is None
    assert color is None


@pytest.mark.parametrize("state", ["red", "yellow", "green", "unknown"])
def test_legacy_status_source_keeps_plain_face(state: str) -> None:
    palette = minimap_utils.load_traffic_light_colors("v4")
    housing = np.array(palette["front_housing"]) / 255.0
    face, glyph, color = minimap_utils._wmv4_tl_render_spec(state, None, palette, housing)
    np.testing.assert_array_equal(face, np.array(palette[state]) / 255.0)
    assert glyph is None
    assert color is None
