/* Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. */
"use strict";

/**
 * Pure helpers for calibrated camera rig geometry.
 * The browser draws the supplied horizontal polygons and image-border rays;
 * it never approximates a calibrated fisheye with a pinhole frustum.
 * @typedef {{status: string, polygons_xy: number[][][], radius_m: number,
 *   height_m: number, total_angle_degrees: number}} Section
 * @typedef {{status: string, origin?: number[], camera_to_rig?: number[][],
 *   horizontal_section?: Section, outline_rays?: number[][],
 *   intrinsics?: {width: number, height: number, center: number[], affine: number[][],
 *   polynomial_type: string, coefficients: number[]}}} Camera
 * @typedef {{camera_keys: string[], geometry: Camera[], layout: string}} Rig
 */
const CameraRig = (() => {
  /** @param {unknown} value @param {number} length @returns {boolean} */
  function vector(value, length) {
    return Array.isArray(value) && value.length === length && value.every(Number.isFinite);
  }

  /** Validate before replacing the currently displayed rig. @param {Rig} rig @returns {Rig} */
  function validateRig(rig) {
    const fail = (message) => { throw new Error(message); };
    if (!rig || !Array.isArray(rig.camera_keys) || !Array.isArray(rig.geometry)) {
      fail("Expected camera_keys and geometry arrays from a camera rig.");
    }
    const count = rig.camera_keys.length;
    if (count < 1 || count > 64 || rig.geometry.length !== count || new Set(rig.camera_keys).size !== count) {
      fail("Provide 1–64 unique cameras with aligned geometry.");
    }
    if (rig.camera_keys.some((key) => typeof key !== "string" || !key.length || key.length > 160)) {
      fail("Camera names must be nonempty strings of at most 160 characters.");
    }
    if (typeof rig.layout !== "string") {
      fail("Provide a layout label.");
    }
    for (const [index, geometry] of rig.geometry.entries()) {
      if (!geometry || typeof geometry.status !== "string") fail(`Camera ${index}: missing status.`);
      const intrinsic = geometry.intrinsics;
      if (intrinsic != null && (
          !Number.isInteger(intrinsic.width) || intrinsic.width <= 0 ||
          !Number.isInteger(intrinsic.height) || intrinsic.height <= 0 || !vector(intrinsic.center, 2) ||
          !Array.isArray(intrinsic.affine) || intrinsic.affine.length !== 2 || !intrinsic.affine.every((row) => vector(row, 2)) ||
          !["pixeldistance-to-angle", "angle-to-pixeldistance"].includes(intrinsic.polynomial_type) ||
          !Array.isArray(intrinsic.coefficients) || !intrinsic.coefficients.length || !intrinsic.coefficients.every(Number.isFinite))) {
        fail(`Camera ${index}: invalid radial polynomial intrinsics.`);
      }
      if (intrinsic != null) {
        const [[a, b], [c, d]] = intrinsic.affine;
        const determinant = a * d - b * c;
        if (!Number.isFinite(determinant) || determinant === 0) {
          fail(`Camera ${index}: intrinsic affine matrix must be invertible.`);
        }
      }
      if (geometry.status !== "qualified") continue;
      if (!vector(geometry.origin, 3) || geometry.origin.some((v) => Math.abs(v) > 10000) ||
          !Array.isArray(geometry.camera_to_rig) || geometry.camera_to_rig.length !== 3 ||
          !geometry.camera_to_rig.every((row) => vector(row, 3))) {
        fail(`Camera ${index}: invalid pose.`);
      }
      const rotation = geometry.camera_to_rig;
      // A proper rotation prevents a malformed input from fabricating headings.
      for (let i = 0; i < 3; i++) {
        for (let j = 0; j < 3; j++) {
          const dot = rotation[i].reduce((sum, value, k) => sum + value * rotation[j][k], 0);
          if (Math.abs(dot - Number(i === j)) > 0.001) fail(`Camera ${index}: rotation must be orthonormal.`);
        }
      }
      const determinant = rotation[0][0] * (rotation[1][1] * rotation[2][2] - rotation[1][2] * rotation[2][1]) -
        rotation[0][1] * (rotation[1][0] * rotation[2][2] - rotation[1][2] * rotation[2][0]) +
        rotation[0][2] * (rotation[1][0] * rotation[2][1] - rotation[1][1] * rotation[2][0]);
      if (Math.abs(determinant - 1) > 0.001) fail(`Camera ${index}: rotation must be right-handed.`);
      if (!Array.isArray(geometry.outline_rays) || geometry.outline_rays.length < 3 ||
          geometry.outline_rays.length > 4096 || geometry.outline_rays.some((ray) =>
            !vector(ray, 3) || Math.abs(Math.hypot(...ray) - 1) > 0.001)) {
        fail(`Camera ${index}: expected unit outline rays in rig coordinates.`);
      }
      const section = geometry.horizontal_section;
      if (!section || typeof section.status !== "string") fail(`Camera ${index}: missing horizontal section.`);
      if (section.status !== "qualified") continue;
      if (!Number.isFinite(section.radius_m) || section.radius_m <= 0 || section.radius_m > 1000 ||
          !Number.isFinite(section.height_m) ||
          !Number.isFinite(section.total_angle_degrees) || !Array.isArray(section.polygons_xy) ||
          !section.polygons_xy.length || section.polygons_xy.length > 32 || section.polygons_xy.some((polygon) =>
            !Array.isArray(polygon) || polygon.length < 3 || polygon.length > 8192 || polygon.some((p) =>
              !vector(p, 2) || p.some((v) => Math.abs(v) > 20000)))) {
        fail(`Camera ${index}: invalid horizontal polygons or radius.`);
      }
    }
    return rig;
  }

  /** @param {Camera} camera @returns {number | null} */
  function heading(camera) {
    if (camera.status !== "qualified" ||
        Math.hypot(camera.camera_to_rig[0][2], camera.camera_to_rig[1][2]) <= 1e-8) return null;
    return Math.atan2(camera.camera_to_rig[1][2], camera.camera_to_rig[0][2]) * 180 / Math.PI;
  }

  /** Conservative sphere containing all selected origins and ray endpoints.
   * @param {Camera[]} cameras @param {number} length
   * @returns {{center: number[], radius: number}} */
  function spatialBounds(cameras, length) {
    const origins = cameras.filter((camera) => camera.status === "qualified").map((camera) => camera.origin);
    if (!origins.length) return {center: [0, 0, 0], radius: Math.max(1, length)};
    const low = [0, 1, 2].map((axis) => Math.min(...origins.map((origin) => origin[axis])));
    const high = [0, 1, 2].map((axis) => Math.max(...origins.map((origin) => origin[axis])));
    return {
      center: low.map((value, axis) => (value + high[axis]) / 2),
      // Account for the unit-ray validation tolerance.
      radius: Math.max(1, Math.hypot(...low.map((value, axis) => high[axis] - value)) / 2 + length * 1.001),
    };
  }

  /** Sample only display rays; the calibrated source arrays remain intact.
   * Preserve bucket endpoints and coordinate extrema to avoid periodic aliasing.
   * @param {number[][]} rays @param {number} limit @returns {number[][]} */
  function sampleOutline(rays, limit) {
    if (rays.length <= limit) return rays;
    const bucketCount = Math.floor(limit / 8), sampled = [];
    for (let bucket = 0; bucket < bucketCount; bucket++) {
      const start = Math.floor(bucket * rays.length / bucketCount);
      const stop = Math.floor((bucket + 1) * rays.length / bucketCount);
      const chosen = new Set([start, stop - 1]);
      for (let axis = 0; axis < 3; axis++) {
        let low = start, high = start;
        for (let i = start + 1; i < stop; i++) {
          if (rays[i][axis] < rays[low][axis]) low = i;
          if (rays[i][axis] > rays[high][axis]) high = i;
        }
        chosen.add(low); chosen.add(high);
      }
      // Fill remaining slots evenly, then retain the original boundary order.
      for (let i = 0; chosen.size < 8; i++) chosen.add(start + Math.floor(i * (stop - start) / 8));
      for (const i of [...chosen].sort((a, b) => a - b)) sampled.push(rays[i]);
    }
    return sampled;
  }

  /** @param {string} key @returns {string} */
  function shortName(key) {
    return key.replace(/^camera_/, "").replace(/_\d+fov$/, "").replaceAll("_", " ");
  }
  return {validateRig, heading, spatialBounds, sampleOutline, shortName};
})();

if (typeof module !== "undefined") module.exports = CameraRig;
