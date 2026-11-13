/* Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. */
"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const {validateRig, heading, spatialBounds, sampleOutline} = require("../geometry.js");
const examples = require("../example-rigs.js");

for (const rig of examples) {
  test(`${rig.layout} supplies a valid calibrated 11-camera rig`, () => {
    assert.equal(validateRig(rig), rig);
    assert.equal(rig.geometry.length, 11);
    assert.equal(rig.camera_keys.filter((key) => !key.includes("fisheye")).length, 7);
    assert.ok(rig.geometry.every((camera) => camera.horizontal_section.radius_m === 8));
    assert.ok(Math.abs(heading(rig.geometry[0])) < 10);
    assert.ok(Math.abs(heading(rig.geometry[3])) > 170);
  });
}

test("heading uses the optical axis in rig FLU coordinates", () => {
  for (const degrees of [-179, -90, 0, 90, 179]) {
    const angle = degrees * Math.PI / 180;
    const camera = {status: "qualified", camera_to_rig: [[0, 0, Math.cos(angle)], [0, 0, Math.sin(angle)], [0, 0, 0]]};
    assert.ok(Math.abs(heading(camera) - degrees) < 1e-9);
  }
});

test("geometry alone is sufficient, including unresolved cameras and sections", () => {
  const {layout, camera_keys, geometry} = structuredClone(examples[0]);
  const rig = {layout, camera_keys, geometry};
  rig.geometry[0] = {status: "unresolved"};
  rig.geometry[1].horizontal_section = {status: "unresolved"};
  assert.equal(validateRig(rig), rig);
});

test("reject malformed camera identities, poses, rays and horizontal sections", () => {
  for (const corrupt of [
    (r) => { r.camera_keys[1] = r.camera_keys[0]; },
    (r) => { r.geometry[0].origin[0] = Infinity; },
    (r) => { r.geometry[0].camera_to_rig[0][0] = 8; },
    (r) => { r.geometry[0].outline_rays[0][0] = NaN; },
    (r) => { r.geometry[0].horizontal_section.radius_m = 0; },
    (r) => { r.geometry[0].horizontal_section.polygons_xy[0] = []; },
    (r) => { r.geometry[0].intrinsics.center = [0]; },
    (r) => { r.geometry[0].intrinsics.polynomial_type = "unknown"; },
    (r) => { r.geometry[0].intrinsics.coefficients = [Infinity]; },
  ]) {
    const rig = structuredClone(examples[0]); corrupt(rig);
    assert.throws(() => validateRig(rig));
  }
});

test("vertical and unresolved cameras have no horizontal heading", () => {
  assert.equal(heading({status: "unresolved"}), null);
  for (const z of [-1, 1]) {
    assert.equal(heading({status: "qualified", camera_to_rig: [[1, 0, 0], [0, z, 0], [0, 0, z]]}), null);
  }
});

test("radial intrinsics require an invertible affine matrix", () => {
  for (const affine of [[[1, 0], [0, 0]], [[1, 2], [2, 4]], [[1e308, 0], [0, 1e308]]]) {
    const rig = structuredClone(examples[0]);
    rig.geometry[0].intrinsics.affine = affine;
    assert.throws(() => validateRig(rig), /affine matrix must be invertible/);
  }
  const rig = structuredClone(examples[0]);
  rig.geometry[0].intrinsics.affine = [[0, -1], [1, 0]];
  assert.equal(validateRig(rig), rig);
});

test("3D framing contains every endpoint and follows translated camera positions", () => {
  const cameras = structuredClone(examples[0].geometry);
  const shift = [1000, -500, 40];
  for (const length of [2, 8, 20]) {
    const original = spatialBounds(examples[0].geometry, length);
    cameras.forEach((camera, i) => { camera.origin = examples[0].geometry[i].origin.map((v, k) => v + shift[k]); });
    const moved = spatialBounds(cameras, length);
    moved.center.forEach((v, k) => assert.ok(Math.abs(v - original.center[k] - shift[k]) < 1e-9));
    assert.ok(Math.abs(moved.radius - original.radius) < 1e-9);
    for (const camera of cameras) {
      for (const ray of camera.outline_rays) {
        const offset = camera.origin.map((v, k) => v + ray[k] * length - moved.center[k]);
        assert.ok(Math.hypot(...offset) <= moved.radius);
      }
    }
  }
  assert.deepEqual(spatialBounds([{status: "unresolved"}], 8), {center: [0, 0, 0], radius: 8});
});

test("dense outlines are bounded without changing the calibrated source", () => {
  const rays = Array.from({length: 4096}, (_, i) => [Math.cos(i / 4096 * 2 * Math.PI), Math.sin(i / 4096 * 2 * Math.PI), 0]);
  const original = structuredClone(rays);
  const sampled = sampleOutline(rays, 128);
  assert.equal(sampled.length * 64, 8192);
  assert.deepEqual(rays, original);
  assert.ok(sampled.every((ray, i) => i === 0 || rays.indexOf(ray) > rays.indexOf(sampled[i - 1])));
  assert.equal(sampleOutline(sampled, 1024), sampled);
});

test("display sampling retains all directions in a periodic dense outline", () => {
  const directions = [[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]];
  const rays = Array.from({length: 4096}, (_, i) => directions[i % directions.length]);
  for (const limit of [128, 136, 512, 910, 1024]) {
    const sampled = sampleOutline(rays, limit);
    assert.ok(sampled.length <= limit);
    assert.deepEqual(new Set(sampled), new Set(directions));
    for (const axis of [0, 1, 2]) {
      assert.equal(Math.min(...sampled.map((ray) => ray[axis])), Math.min(...rays.map((ray) => ray[axis])));
      assert.equal(Math.max(...sampled.map((ray) => ray[axis])), Math.max(...rays.map((ray) => ray[axis])));
    }
  }
});
