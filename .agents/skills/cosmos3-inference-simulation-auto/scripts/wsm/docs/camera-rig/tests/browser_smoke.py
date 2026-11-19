# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Exercise the static explorer with Playwright; run in a Slurm CPU allocation."""

import argparse
import hashlib
import json
import math
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from playwright.sync_api import Page, Response, expect, sync_playwright


def assert_asset_versions(page: Page, root: Path, responses: list[Response]) -> dict[str, str]:
    """Check the exact bytes loaded by the browser, including cache-version URLs."""
    urls = page.locator('script[src], link[rel="stylesheet"]').evaluate_all(
        "nodes => nodes.map(node => node.src || node.href)"
    )
    assert len(urls) == 5, urls
    loaded = {response.url: response for response in responses}
    versions: dict[str, str] = {}
    for url in urls:
        parsed = urlsplit(url)
        filename = Path(parsed.path).name
        assert filename in {"styles.css", "geometry.js", "example-rigs.js", "layouts.js", "app.js"}, url
        digest = hashlib.sha256((root / filename).read_bytes()).hexdigest()
        assert parsed.query == f"v={digest[:12]}", (filename, parsed.query, digest)
        response = loaded[url]
        assert response.ok and hashlib.sha256(response.body()).hexdigest() == digest, url
        versions[filename] = url
    assert len(versions) == 5, versions
    return versions


def fan_path(page: Page, role: str) -> str:
    return page.locator(f'[data-fan="{role}"]').evaluate(
        "node => node.ownerSVGElement.querySelector(node.getAttribute('href')).getAttribute('d')"
    )


def assert_camera_visibility(page: Page, cameras: list[int], selected: int | None = None) -> None:
    """Verify independent color and outline controls against actual SVG/Canvas draws."""
    result = page.evaluate("""async () => {
        await new Promise(requestAnimationFrame);
        const canvas = document.getElementById('spatial-view'), context = canvas.getContext('2d');
        const original = Object.fromEntries(['arc', 'beginPath', 'moveTo', 'lineTo', 'fill', 'stroke'].map(key => [key, context[key]]));
        const markers = [], outlines = [];
        let marker = false, vertices = 0;
        context.beginPath = function() { marker = false; vertices = 0; return original.beginPath.call(this); };
        context.arc = function(...args) { marker = true; return original.arc.apply(this, args); };
        for (const key of ['moveTo', 'lineTo']) context[key] = function(...args) {
            vertices++; return original[key].apply(this, args);
        };
        context.fill = function(...args) { if (marker) markers.push(this.fillStyle); return original.fill.apply(this, args); };
        context.stroke = function(...args) {
            if (vertices >= 4 && ['#69b6e8', '#627d86'].includes(this.strokeStyle)) outlines.push(this.strokeStyle);
            return original.stroke.apply(this, args);
        };
        document.getElementById('ray-length').dispatchEvent(new Event('input'));
        await new Promise(requestAnimationFrame);
        Object.assign(context, original);
        return {
            checked: [...document.querySelectorAll('[data-include]:checked')].map(node => Number(node.dataset.include)),
            fans: [...document.querySelectorAll('[data-camera-fov]')].map(node => Number(node.dataset.cameraFov)).sort((a, b) => a - b),
            markers, outlines,
            selected: Number(document.querySelector('[data-camera][aria-pressed=true]').dataset.camera),
            coloredFan: Number(document.getElementById('selected-fan').dataset.cameraColor)
        };
    }""")
    if selected is not None:
        assert result["selected"] == selected, result
    selected = result["selected"]
    assert result["checked"] == result["fans"] == cameras, result
    assert result["coloredFan"] == selected, result
    assert len(result["markers"]) == len(set(cameras) | {selected}), result
    assert result["markers"].count("#78c4f1") == 1, result
    assert len(result["outlines"]) == len(cameras), result
    assert result["outlines"].count("#69b6e8") == int(selected in cameras), result
    assert page.locator("#compare-select, #swap-pair, #intersection-fan").count() == 0


def settle_frames(page: Page) -> None:
    page.evaluate("async () => { await new Promise(requestAnimationFrame); await new Promise(requestAnimationFrame); }")


def assert_selection_persistence(page: Page) -> dict[str, bool]:
    """Keep all, none or shared camera identities across layout and calibration changes."""
    layouts = page.evaluate("CAMERA_RIG_LAYOUTS")
    values = {layout["label"]: str(index) for index, layout in enumerate(layouts)}
    page.locator("#layout-select").select_option(values["11 cam"])
    page.locator("#rig-select").select_option("0")
    page.locator("#select-all").click()
    page.locator("#camera-3").click()
    page.locator("#layout-select").select_option(values["4 cam B"])
    assert_camera_visibility(page, [0, 1, 2, 3], selected=2)
    page.locator("#rig-select").select_option("1")
    assert_camera_visibility(page, [0, 1, 2, 3], selected=2)
    page.locator("#layout-select").select_option(values["1 cam"])
    assert_camera_visibility(page, [0], selected=0)
    page.locator("#layout-select").select_option(values["11 cam"])
    assert_camera_visibility(page, list(range(11)), selected=0)

    page.locator("#deselect-all").click()
    page.locator("#layout-select").select_option(values["4 cam A"])
    assert_camera_visibility(page, [], selected=0)
    page.locator("#rig-select").select_option("0")
    assert_camera_visibility(page, [], selected=0)
    page.locator("#layout-select").select_option(values["11 cam"])
    assert_camera_visibility(page, [], selected=7)

    for index in [1, 3, 7]:
        page.locator(f"#include-{index}").check()
    page.locator("#camera-3").click()
    page.locator("#layout-select").select_option(values["4 cam B"])
    assert_camera_visibility(page, [1, 2], selected=2)
    page.locator("#rig-select").select_option("1")
    assert_camera_visibility(page, [1, 2], selected=2)
    page.locator("#layout-select").select_option(values["11 cam"])
    assert_camera_visibility(page, [1, 3], selected=3)
    page.locator("#layout-select").select_option(values["4 cam A"])
    assert_camera_visibility(page, [], selected=0)

    page.locator("#layout-select").select_option(values["11 cam"])
    page.locator("#rig-select").select_option("0")
    page.locator("#include-0").check()
    page.locator("#camera-0").click()
    assert_camera_visibility(page, [0], selected=0)
    return {
        "all_preserved": True,
        "none_preserved": True,
        "partial_intersection_by_camera_key": True,
        "disjoint_layout_stays_unchecked": True,
        "highlight_preserved_or_first_camera": True,
    }


def assert_layouts(page: Page, output: Path) -> dict:
    """Exercise every camera subset through selection, calibration changes and export."""
    output.mkdir(parents=True, exist_ok=True)
    layouts = page.evaluate("CAMERA_RIG_LAYOUTS")
    calibrations = page.evaluate("CAMERA_RIG_EXAMPLES")
    assert len(layouts) == 22
    assert {len(layout["camera_keys"]) for layout in layouts} == {1, 2, 4, 5, 6, 7, 8, 10, 11}
    assert len({tuple(layout["camera_keys"]) for layout in layouts}) == len(layouts)
    assert page.locator("#layout-select option").all_text_contents() == [layout["label"] for layout in layouts]
    for layout_index, layout in enumerate(layouts):
        count = len(layout["camera_keys"])
        page.locator("#layout-select").select_option(str(layout_index))
        for setting, calibration in enumerate(calibrations):
            page.locator("#rig-select").select_option(str(setting))
            expect(page.locator("#layout-select")).to_have_value(str(layout_index))
            page.locator("#deselect-all").click()
            page.locator("#include-0").check()
            page.locator("#camera-0").click()
            expect(page.locator("#camera-count")).to_have_text(f"1 / {count}")
            assert (
                page.locator("[data-camera]").evaluate_all("nodes => nodes.map(node => node.dataset.cameraKey)")
                == layout["camera_keys"]
            )
            assert page.locator("[data-camera] span").all_text_contents() == [
                f"{calibration['camera_keys'].index(key) + 1:02d}" for key in layout["camera_keys"]
            ]
            assert_camera_visibility(page, [0], selected=0)
            page.locator(f"#camera-{count - 1}").click()
            page.locator("#select-all").click()
            assert_camera_visibility(page, list(range(count)), selected=count - 1)
            if layout["label"] == "4 cam B" and setting == 0:
                page.screenshot(path=str(output / "four-camera.png"), full_page=True)
            page.locator("#deselect-all").click()
            assert_camera_visibility(page, [], selected=count - 1)
            with page.expect_download() as pending:
                page.locator("#export-rig").click()
            path = output / f"layout-{layout_index}-setting-{setting}.json"
            pending.value.save_as(path)
            exported = json.loads(path.read_text())
            assert exported == {
                "layout": f"{calibration['layout']} · {layout['label']}",
                "camera_keys": layout["camera_keys"],
                "geometry": [
                    calibration["geometry"][calibration["camera_keys"].index(key)] for key in layout["camera_keys"]
                ],
            }
    page.locator("#import-file").set_input_files(output / "layout-2-setting-0.json")
    expect(page.locator("#layout-select")).to_be_disabled()
    expect(page.locator("#layout-select option:checked")).to_have_text("Imported layout")
    assert_camera_visibility(page, [0], selected=0)
    assert page.locator("[data-camera]").count() == 4
    page.locator("#rig-select").select_option("0")
    expect(page.locator("#layout-select")).to_be_enabled()
    assert_camera_visibility(page, [7], selected=7)
    assert page.locator("[data-camera]").count() == 11
    page.locator("#deselect-all").click()
    page.locator("#include-0").check()
    page.locator("#camera-0").click()
    return {
        "layouts": len(layouts),
        "calibrations": len(calibrations),
        "verified_exports": len(layouts) * len(calibrations),
    }


def assert_rotation_work(page: Page) -> dict:
    """Measure idle drag work, backing-store writes and stable high-DPI redraws."""
    canvas = page.locator("#spatial-view")
    settle_frames(page)
    canvas.evaluate("""canvas => {
        canvas.resizeWrites = {width: 0, height: 0, redundant: 0};
        for (const key of ['width', 'height']) {
            const descriptor = Object.getOwnPropertyDescriptor(HTMLCanvasElement.prototype, key);
            Object.defineProperty(canvas, key, {
                configurable: true,
                get() { return descriptor.get.call(this); },
                set(value) {
                    this.resizeWrites[key]++;
                    if (value === descriptor.get.call(this)) this.resizeWrites.redundant++;
                    descriptor.set.call(this, value);
                }
            });
        }
    }""")
    expect(page.locator("#auto-rotate")).to_have_attribute("aria-pressed", "false")
    original = canvas.get_attribute("data-orientation")
    before = int(canvas.get_attribute("data-render-count"))
    page.locator("#auto-rotate").click()
    page.wait_for_function(
        "count => Number(document.getElementById('spatial-view').dataset.renderCount) >= count + 4", arg=before
    )
    assert canvas.get_attribute("data-orientation") != original
    rotating_writes = canvas.evaluate("canvas => canvas.resizeWrites")
    assert rotating_writes == {"width": 0, "height": 0, "redundant": 0}, rotating_writes
    canvas.scroll_into_view_if_needed()
    box = canvas.bounding_box()
    page.mouse.move(box["x"] + 100, box["y"] + 100)
    page.mouse.down()
    settle_frames(page)
    held = canvas.get_attribute("data-orientation")
    held_count = int(canvas.get_attribute("data-render-count"))
    page.wait_for_timeout(150)
    assert canvas.get_attribute("data-orientation") == held
    assert int(canvas.get_attribute("data-render-count")) == held_count
    page.mouse.move(box["x"] + 135, box["y"] + 120, steps=4)
    settle_frames(page)
    moved = canvas.get_attribute("data-orientation")
    moved_count = int(canvas.get_attribute("data-render-count"))
    assert moved != held and moved_count > held_count
    page.wait_for_timeout(150)
    assert int(canvas.get_attribute("data-render-count")) == moved_count
    page.mouse.up()
    page.wait_for_function("value => document.getElementById('spatial-view').dataset.orientation !== value", arg=moved)
    page.locator("#auto-rotate").click()
    settle_frames(page)
    stopped = canvas.get_attribute("data-orientation")
    stopped_count = int(canvas.get_attribute("data-render-count"))
    page.wait_for_timeout(150)
    assert canvas.get_attribute("data-orientation") == stopped
    assert int(canvas.get_attribute("data-render-count")) == stopped_count
    pixels = canvas.evaluate("canvas => canvas.toDataURL()")
    for _ in range(4):
        page.locator("#ray-length").dispatch_event("input")
        settle_frames(page)
    assert canvas.evaluate("canvas => canvas.toDataURL()") == pixels
    dpr = page.evaluate("Math.min(devicePixelRatio, 3)")
    transform = canvas.evaluate(
        "canvas => { const m = canvas.getContext('2d').getTransform(); return [m.a, m.b, m.c, m.d, m.e, m.f]; }"
    )
    assert transform == [dpr, 0, 0, dpr, 0, 0], transform
    viewport = page.viewport_size
    original_width = canvas.evaluate("canvas => canvas.width")
    page.set_viewport_size({**viewport, "width": viewport["width"] - 60})
    page.wait_for_function("width => document.getElementById('spatial-view').width !== width", arg=original_width)
    settle_frames(page)
    resized_writes = canvas.evaluate("canvas => canvas.resizeWrites")
    assert resized_writes["width"] > 0 and resized_writes["redundant"] == 0, resized_writes
    canvas.evaluate("canvas => { delete canvas.width; delete canvas.height; delete canvas.resizeWrites; }")
    page.set_viewport_size(viewport)
    settle_frames(page)
    return {
        "device_pixel_ratio": dpr,
        "writes_during_rotation": rotating_writes,
        "idle_drag_frames": 0,
        "manual_drag_frames": moved_count - held_count,
        "writes_after_resize": resized_writes,
        "repeated_redraw_pixels_identical": True,
        "transform": transform,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(SimpleHTTPRequestHandler, directory=str(root)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    errors: list[str] = []
    external: list[str] = []
    checks: list[str] = []
    rotation_metrics: dict[str, dict] = {}
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            context = browser.new_context(viewport={"width": 1500, "height": 1100}, device_scale_factor=1)
            page = context.new_page()
            responses: list[Response] = []
            page.on("response", lambda response: responses.append(response))
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.on(
                "request",
                lambda request: (
                    external.append(request.url) if not request.url.startswith("http://127.0.0.1:") else None
                ),
            )
            page.goto(f"http://127.0.0.1:{server.server_port}/index.html")
            page.wait_for_function("document.getElementById('camera-count').textContent === '1 / 11'")
            page.wait_for_function("document.getElementById('spatial-view').dataset.orientation")
            asset_versions = assert_asset_versions(page, root, responses)
            checks.append("Browser-loaded scripts and styles match their content-version URLs and source bytes")
            assert page.locator("#matrix, #strategies, #export-mask, #policy-description, #decision").count() == 0
            assert "attention" not in page.locator("body").inner_text().lower()
            assert "How links are chosen" not in page.locator("body").inner_text()
            assert page.locator("#camera-position").inner_text().endswith(" m")
            assert len(fan_path(page, "selected")) > 1000
            assert (
                page.locator("#spatial-view").evaluate(
                    "e => new Set(e.getContext('2d').getImageData(0, 0, e.width, e.height).data).size"
                )
                > 30
            )
            checks.append("First camera selected and checked by default; camera pose, calibrated SVG and 3D vehicle")
            assert_camera_visibility(page, [0], selected=0)
            page.screenshot(path=str(args.output / "desktop.png"), full_page=True)
            canvas = page.locator("#spatial-view")
            canvas.screenshot(path=str(args.output / "vehicle.png"))

            for index in range(11):
                page.locator(f"#camera-{index}").click()
                assert_camera_visibility(page, [0], selected=index)
                checkbox = page.locator(f"#include-{index}")
                checkbox.focus()
                checkbox.press("Space")
                expect(checkbox).to_be_focused()
                assert_camera_visibility(page, [] if index == 0 else [0, index], selected=index)
                checkbox.press("Space")
                assert_camera_visibility(page, [0], selected=index)
            page.locator("#select-all").click()
            assert_camera_visibility(page, list(range(11)), selected=10)
            page.screenshot(path=str(args.output / "all-outlines.png"), full_page=True)
            page.locator("#deselect-all").click()
            assert_camera_visibility(page, [], selected=10)
            page.locator("#camera-3").click()
            assert_camera_visibility(page, [], selected=3)
            page.locator("#select-all").click()
            assert_camera_visibility(page, list(range(11)), selected=3)
            checks.append(
                "Every checkbox controls only outlines; selection controls color independently, including zero outlines; Select all and Deselect all preserve selection"
            )

            assert page.locator("#preset, #source-line").count() == 0
            assert page.locator("#rig-select option").all_text_contents() == [
                "Calibration setting A",
                "Calibration setting B",
            ]
            page.locator("#calibration-details summary").click()
            for setting in range(2):
                page.locator("#rig-select").select_option(str(setting))
                page.locator("#deselect-all").click()
                page.locator("#include-0").check()
                page.locator("#camera-0").click()
                assert_camera_visibility(page, [0], selected=0)
                assert page.locator("[data-camera]").count() == 11
                for camera_index in range(11):
                    page.locator(f"#camera-{camera_index}").click()
                    expected = page.evaluate(
                        "([setting, index]) => CAMERA_RIG_EXAMPLES[setting].geometry[index]", [setting, camera_index]
                    )
                    intrinsic = expected["intrinsics"]
                    expect(page.locator("#intrinsic-size")).to_have_text(
                        f"{intrinsic['width']} × {intrinsic['height']} px"
                    )
                    center = [
                        float(value)
                        for value in page.locator("#intrinsic-center").inner_text().removesuffix(" px").split(" / ")
                    ]
                    assert all(abs(a - b) <= 5.1e-5 for a, b in zip(center, intrinsic["center"], strict=True))
                    for selector, matrix in [
                        ("#camera-rotation", expected["camera_to_rig"]),
                        ("#intrinsic-affine", intrinsic["affine"]),
                    ]:
                        actual = [float(value) for value in page.locator(selector).inner_text().split()]
                        assert all(
                            abs(a - b) <= 5.1e-7
                            for a, b in zip(actual, [v for row in matrix for v in row], strict=True)
                        )
                    coefficients = [
                        float(line.split("=")[1])
                        for line in page.locator("#intrinsic-coefficients").inner_text().splitlines()
                    ]
                    assert all(
                        math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-20)
                        for a, b in zip(coefficients, intrinsic["coefficients"], strict=True)
                    )
                    expected_formula = (
                        "θ = Σ cₖ rᵏ" if intrinsic["polynomial_type"] == "pixeldistance-to-angle" else "r = Σ cₖ θᵏ"
                    )
                    expect(page.locator("#intrinsic-polynomial")).to_have_text(expected_formula)
                    translation = [
                        float(value)
                        for value in page.locator("#camera-translation").inner_text().removesuffix(" m").split(" / ")
                    ]
                    assert all(abs(a - b) <= 5.1e-7 for a, b in zip(translation, expected["origin"], strict=True))
                with page.expect_download() as pending:
                    page.locator("#export-rig").click()
                download_path = args.output / f"setting-{setting + 1}.json"
                pending.value.save_as(download_path)
                exported_setting = json.loads(download_path.read_text())
                assert set(exported_setting) == {"layout", "camera_keys", "geometry"}
                expected_setting = page.evaluate("setting => CAMERA_RIG_EXAMPLES[setting]", setting)
                expected_setting["layout"] += " · 11 cam"
                assert exported_setting == expected_setting
            page.locator("#rig-select").select_option("0")
            page.screenshot(path=str(args.output / "parameters.png"), full_page=True)
            page.locator("#calibration-details summary").click()
            checks.append(
                "Two generically named calibrations, per-camera intrinsic/extrinsic inspection and complete numeric JSON exports"
            )
            selection_metrics = assert_selection_persistence(page)
            checks.append(
                "Calibration and layout changes retain all or no outlines, intersect partial choices by camera identity and preserve available highlights"
            )
            layout_metrics = assert_layouts(page, args.output / "layouts")
            checks.append(
                "All 22 layouts in both calibrations preserve exact camera membership, numbering and export geometry; imported layouts stay separate"
            )

            rotation_metrics["default"] = assert_rotation_work(page)
            checks.append(
                "Auto rotate stops idle drag redraws, resumes after release, reuses its backing store and preserves redraw pixels"
            )

            page.locator("#rig-select").select_option("1")
            assert page.locator("#camera-count").inner_text() == "1 / 11"
            expect(page.locator("#rig-select option:checked")).to_have_text("Calibration setting B")
            canvas = page.locator("#spatial-view")
            canvas.focus()
            original = canvas.get_attribute("data-orientation")
            canvas.press("ArrowRight")
            page.wait_for_function(
                "value => document.getElementById('spatial-view').dataset.orientation !== value", arg=original
            )
            before_drag = canvas.get_attribute("data-orientation")
            box = canvas.bounding_box()
            page.mouse.move(box["x"] + 150, box["y"] + 150)
            page.mouse.down()
            page.mouse.move(box["x"] + 220, box["y"] + 180, steps=5)
            page.mouse.up()
            page.wait_for_function(
                "value => document.getElementById('spatial-view').dataset.orientation !== value", arg=before_drag
            )
            before_zoom = canvas.get_attribute("data-orientation")
            page.mouse.wheel(0, 100)
            page.wait_for_function(
                "previous => document.getElementById('spatial-view').dataset.orientation !== previous", arg=before_zoom
            )
            for preset in ["top", "side", "orbit"]:
                page.locator(f'[data-view="{preset}"]').click()
            before = fan_path(page, "selected")
            page.locator("#ray-length").evaluate("e => { e.value = '14'; e.dispatchEvent(new Event('input')); }")
            expect(page.locator("#ray-value")).to_have_text("14 m")
            assert fan_path(page, "selected") == before
            checks.append("Second calibration, pointer/keyboard orbit, zoom, presets and display-only ray length")

            with page.expect_download() as pending:
                page.locator("#export-svg").click()
            pending.value.save_as(args.output / "camera-rig.svg")
            assert 'href="#camera-outline-0"' in (args.output / "camera-rig.svg").read_text()
            assert "clipPath" not in (args.output / "camera-rig.svg").read_text()
            with page.expect_download() as pending:
                page.locator("#export-rig").click()
            rig_path = args.output / "exported-rig.json"
            pending.value.save_as(rig_path)
            exported = json.loads(rig_path.read_text())
            assert set(exported) == {"layout", "camera_keys", "geometry"}
            page.locator("#import-file").set_input_files(rig_path)
            page.wait_for_function("document.getElementById('status').textContent.startsWith('Loaded locally')")
            assert fan_path(page, "selected") == before
            checks.append("SVG/geometry-only rig downloads and round-trip local JSON import")

            page.locator("#import-file").set_input_files(
                {"name": "broken.json", "mimeType": "application/json", "buffer": b'{"camera_keys": []}'}
            )
            page.wait_for_function("document.getElementById('status').textContent.includes('current rig is unchanged')")
            assert fan_path(page, "selected") == before
            singular = json.loads(rig_path.read_text())
            singular["geometry"][0]["intrinsics"]["affine"] = [[1, 0], [0, 0]]
            page.locator("#import-file").set_input_files(
                {"name": "singular.json", "mimeType": "application/json", "buffer": json.dumps(singular).encode()}
            )
            expect(page.locator("#status")).to_contain_text("affine matrix must be invertible")
            expect(page.locator("#status")).to_contain_text("current rig is unchanged")
            assert_camera_visibility(page, [0], selected=0)
            assert fan_path(page, "selected") == before
            rig = json.loads(rig_path.read_text())
            rig["geometry"][0] = {"status": "unresolved"}
            rig["camera_keys"][0] = '<img src=x onerror="window.injected=true">'
            page.locator("#import-file").set_input_files(
                {"name": "unresolved.json", "mimeType": "application/json", "buffer": json.dumps(rig).encode()}
            )
            page.wait_for_function("document.getElementById('status').textContent.startsWith('Loaded locally')")
            assert fan_path(page, "selected") == ""
            assert page.locator("#camera-position").inner_text() == "Unavailable"
            assert page.locator("#intrinsics-empty").evaluate("element => !element.hidden")
            assert "Missing geometry is not drawn" in page.locator("#plan-caption").inner_text()
            assert not page.evaluate("Boolean(window.injected)")
            checks.append("Invalid import preserves state; unresolved geometry and untrusted names render safely")

            vertical = json.loads(rig_path.read_text())
            vertical["geometry"][0]["camera_to_rig"] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
            page.locator("#import-file").set_input_files(
                {"name": "vertical.json", "mimeType": "application/json", "buffer": json.dumps(vertical).encode()}
            )
            expect(page.locator("#camera-heading")).to_have_text("Unavailable")
            assert "heading Unavailable" in page.locator("#plan-caption").inner_text()
            checks.append("Vertical-camera heading agrees between inspector and caption")

            dense = {"layout": "translated-dense", "camera_keys": [], "geometry": []}
            for index in range(64):
                dense["camera_keys"].append(f"camera_{index}")
                dense["geometry"].append(
                    {
                        "status": "qualified",
                        "origin": [1000 + index / 2, -500, 40],
                        "camera_to_rig": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                        "horizontal_section": {"status": "unresolved"},
                        "outline_rays": [[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]] * 1024,
                    }
                )
            payload = json.dumps(dense, separators=(",", ":")).encode()
            assert len(payload) < 4 * 1024 * 1024
            page.locator("#import-file").set_input_files(
                {"name": "dense.json", "mimeType": "application/json", "buffer": payload}
            )
            expect(page.locator("#camera-count")).to_have_text("1 / 64")
            page.locator("#select-all").click()
            expect(page.locator("#camera-count")).to_have_text("64 / 64")
            page.wait_for_function("document.getElementById('spatial-view').dataset.rayCount === '8192'")
            metrics = page.evaluate("""async () => {
                const canvas = document.getElementById('spatial-view');
                await new Promise(requestAnimationFrame);
                const context = canvas.getContext('2d');
                const original = Object.fromEntries(['arc', 'beginPath', 'moveTo', 'lineTo', 'stroke'].map(key => [key, context[key]]));
                const markers = [], outlineVertices = [];
                let path = [];
                context.arc = function(x, y, ...rest) { markers.push([x, y]); return original.arc.call(this, x, y, ...rest); };
                context.beginPath = function() { path = []; return original.beginPath.call(this); };
                for (const key of ['moveTo', 'lineTo']) {
                    context[key] = function(x, y) { path.push([x, y]); return original[key].call(this, x, y); };
                }
                context.stroke = function() {
                    if (path.length > 100) outlineVertices.push(new Set(path.map(point => point.join(','))).size);
                    return original.stroke.call(this);
                };
                const before = Number(canvas.dataset.renderCount);
                for (let i = 0; i < 100; i++) canvas.dispatchEvent(new KeyboardEvent('keydown', {key: 'ArrowRight'}));
                const immediate = Number(canvas.dataset.renderCount) - before;
                await new Promise(requestAnimationFrame);
                Object.assign(context, original);
                return {immediate, frames: Number(canvas.dataset.renderCount) - before,
                    width: canvas.clientWidth, height: canvas.clientHeight, markers, outlineVertices};
            }""")
            assert metrics["immediate"] == 0 and metrics["frames"] == 1, metrics
            assert len(metrics["markers"]) == 64
            assert all(0 <= x <= metrics["width"] and 0 <= y <= metrics["height"] for x, y in metrics["markers"])
            assert metrics["outlineVertices"] == [4] * 64, metrics
            (args.output / "dense-render.json").write_text(json.dumps(metrics, indent=2) + "\n")
            page.screenshot(path=str(args.output / "translated-dense.png"), full_page=True)
            with page.expect_download() as pending:
                page.locator("#export-rig").click()
            dense_path = args.output / "dense-export.json"
            pending.value.save_as(dense_path)
            assert json.loads(dense_path.read_text()) == dense
            checks.append(
                "Translated 64-camera rig stays in frame; periodic outlines retain four corners; 100 events coalesce to one bounded redraw; export preserves all rays"
            )
            rotation_metrics["dense_64_camera"] = assert_rotation_work(page)

            dense_plan = {
                "layout": "dense-polygons",
                "camera_keys": [],
                "geometry": [],
            }
            polygon = [[-1, -1], [1, -1], [1, 1], [-1, 1]] * 64
            for index in range(64):
                dense_plan["camera_keys"].append(f"camera_{index}")
                dense_plan["geometry"].append(
                    {
                        "status": "qualified",
                        "origin": [0, 0, 1],
                        "camera_to_rig": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                        "outline_rays": [[1, 0, 0], [0, 1, 0], [-1, 0, 0]],
                        "horizontal_section": {
                            "status": "qualified",
                            "radius_m": 8,
                            "height_m": 1,
                            "total_angle_degrees": 360,
                            "polygons_xy": [polygon] * 32,
                        },
                    }
                )
            payload = json.dumps(dense_plan, separators=(",", ":")).encode()
            assert len(payload) < 4 * 1024 * 1024
            page.locator("#import-file").set_input_files(
                {"name": "dense-polygons.json", "mimeType": "application/json", "buffer": payload}
            )
            expect(page.locator("#rig-select option:checked")).to_have_text("Imported · dense-polygons.json")
            page.locator("#select-all").click()
            plan_metrics = page.evaluate("""() => {
                const paths = [...document.querySelectorAll('[id^=camera-outline-]')];
                const observer = new MutationObserver(() => {});
                observer.observe(document.getElementById('plan-view'), {subtree: true, attributes: true, attributeFilter: ['d']});
                const start = performance.now();
                for (const index of [1, 3, 5, 7]) document.getElementById(`camera-${index}`).click();
                document.getElementById('include-8').click();
                const selectorMs = performance.now() - start;
                const mutations = observer.takeRecords().filter(record => record.target.id.startsWith('camera-outline-'));
                observer.disconnect();
                return {vertices: paths.reduce((sum, path) => sum + (path.getAttribute('d').match(/[ML]/g) || []).length, 0),
                    retained: paths.every(path => document.getElementById(path.id) === path),
                    rewritten_paths: mutations.length, selector_js_ms: selectorMs};
            }""")
            assert plan_metrics["vertices"] == 64 * 32 * 256, plan_metrics
            assert plan_metrics["retained"] and plan_metrics["rewritten_paths"] == 0, plan_metrics
            assert_camera_visibility(page, [i for i in range(64) if i != 8])
            (args.output / "dense-plan.json").write_text(json.dumps(plan_metrics, indent=2) + "\n")
            checks.append("A 524,288-vertex rig retains exact SVG geometry across selection and visibility changes")

            page.locator("#rig-select").select_option("0")
            page.locator("#ray-length").evaluate("e => { e.value = '8'; e.dispatchEvent(new Event('input')); }")
            page.set_viewport_size({"width": 390, "height": 844})
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            page.locator("#camera-5").click()
            page.locator("#calibration-details summary").click()
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            page.screenshot(path=str(args.output / "mobile.png"), full_page=True)
            checks.append("390px mobile layout without horizontal page overflow")

            touch = browser.new_context(viewport={"width": 390, "height": 844}, has_touch=True)
            mobile = touch.new_page()
            mobile.on("pageerror", lambda error: errors.append(str(error)))
            mobile.goto(f"http://127.0.0.1:{server.server_port}/index.html")
            mobile.wait_for_function("document.getElementById('spatial-view').dataset.orientation")
            mobile.locator("#include-8").tap()
            assert_camera_visibility(mobile, [0, 8], selected=0)
            mobile.locator("#camera-8").tap()
            assert mobile.locator("#camera-8").get_attribute("aria-pressed") == "true"
            assert_camera_visibility(mobile, [0, 8], selected=8)
            mobile.locator("#deselect-all").tap()
            assert_camera_visibility(mobile, [], selected=8)
            touch.close()
            checks.append("Touch selection, outlines and Deselect all stay independent on mobile")

            high_dpi = browser.new_context(viewport={"width": 1500, "height": 1100}, device_scale_factor=2)
            high_dpi_page = high_dpi.new_page()
            high_dpi_page.on("pageerror", lambda error: errors.append(str(error)))
            high_dpi_page.goto(f"http://127.0.0.1:{server.server_port}/index.html")
            high_dpi_page.wait_for_function("document.getElementById('spatial-view').dataset.orientation")
            rotation_metrics["high_dpi"] = assert_rotation_work(high_dpi_page)
            high_dpi.close()
            (args.output / "rotation-render.json").write_text(json.dumps(rotation_metrics, indent=2) + "\n")
            checks.append(
                "Dense 64-camera and 2x-DPI rotation avoid idle redraws and redundant canvas resets; real viewport changes resize correctly"
            )

            offline = browser.new_context(offline=True, viewport={"width": 1440, "height": 1000})
            local = offline.new_page()
            local.on("pageerror", lambda error: errors.append(str(error)))
            local.goto((root / "index.html").as_uri())
            local.wait_for_function("document.getElementById('camera-count').textContent === '1 / 11'")
            local.locator("#camera-7").click()
            assert "fisheye" in local.locator("#selected-title").inner_text().lower()
            assert len(fan_path(local, "selected")) > 1000
            checks.append("Offline file:// opening with all assets available locally")
            assert not errors, errors
            assert not external, external
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
    (args.output / "browser-result.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "checks": checks,
                "page_errors": errors,
                "external_requests": external,
                "asset_versions": asset_versions,
                "layouts": layout_metrics,
                "selection_persistence": selection_metrics,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"status": "PASS", "checks": checks}))


if __name__ == "__main__":
    main()
