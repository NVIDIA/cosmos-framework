/* Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. */
"use strict";

(() => {
  /** @param {string} id @returns {HTMLElement} */
  const $ = (id) => document.getElementById(id);
  /** @param {unknown} value @returns {string} */
  const escapeHtml = (value) => String(value).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[c]);
  const {validateRig, heading, sampleOutline, shortName} = CameraRig;
  const examples = CAMERA_RIG_EXAMPLES.map(validateRig);
  const layouts = CAMERA_RIG_LAYOUTS;
  let layoutIndex = layouts.findIndex((layout) => layout.camera_keys.length === 11);
  let cameraOrder = examples[0].camera_keys;
  let rig = examples[0];
  let indices = [0];
  let selected = 0;
  let yaw = .6, pitch = .65, zoom = 1, drag = null;
  let importVersion = 0;
  let spatialFrame = 0;
  let autoRotate = false, lastRotationTime = 0;
  let planRig = null;
  const rayCache = new WeakMap();
  const MAX_RENDER_RAYS = 8192;

  /** @param {number} i @returns {string} */
  const name = (i) => shortName(rig.camera_keys[i]);
  /** @param {number} i @returns {string} */
  const number = (i) => String(cameraOrder.indexOf(rig.camera_keys[i]) + 1).padStart(2, "0");

  /** Imported rigs retain their own coordinate convention. @returns {string} */
  const originLabel = () => Number($("rig-select").value) < CAMERA_RIG_EXAMPLES.length ? "Rear axle center" : "Rig origin";

  /** @param {import('./geometry.js').Camera} camera @returns {string} */
  const formatHeading = (camera) => {
    const angle = heading(camera);
    return angle == null ? "Unavailable" : `${angle.toFixed(1)}°`;
  };

  /** @param {unknown} data @param {string} filename @param {string} type @returns {void} */
  function download(data, filename, type = "application/json") {
    const content = type === "application/json" ? JSON.stringify(data, null, 2) + "\n" : data;
    const url = URL.createObjectURL(new Blob([content], {type}));
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = filename;
    anchor.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  /** Change checked outlines independently of the selected camera. @param {number[]} next @returns {void} */
  function setIndices(next) {
    indices = next;
    zoom = 1;
    $("status").textContent = "";
    render();
  }

  /** @param {import('./geometry.js').Rig} next @param {boolean} preserveSelection @returns {void} */
  function loadRig(next, preserveSelection = false) {
    const activeKey = rig.camera_keys[selected];
    const checkedKeys = new Set(indices.map((i) => rig.camera_keys[i]));
    const allChecked = indices.length === rig.camera_keys.length;
    rig = validateRig(next);
    indices = preserveSelection ? rig.camera_keys.map((_, i) => i).filter((i) => allChecked || checkedKeys.has(rig.camera_keys[i])) : [0];
    selected = preserveSelection ? Math.max(0, rig.camera_keys.indexOf(activeKey)) : 0;
    yaw = .6; pitch = .65; zoom = 1;
    $("status").textContent = "";
    render();
  }

  /** Apply the chosen camera subset without changing its calibrated geometry.
   * @param {boolean} preserveSelection @returns {void} */
  function loadCalibration(preserveSelection = false) {
    const index = Number($("rig-select").value), calibration = examples[index];
    const builtin = index < CAMERA_RIG_EXAMPLES.length;
    cameraOrder = calibration.camera_keys;
    $("layout-select").disabled = !builtin;
    $("layout-select").innerHTML = builtin ? layouts.map((layout, i) =>
      `<option value="${i}">${escapeHtml(layout.label)}</option>`).join("") : '<option>Imported layout</option>';
    if (!builtin) { loadRig(calibration, preserveSelection); return; }
    $("layout-select").value = String(layoutIndex);
    const layout = layouts[layoutIndex];
    loadRig({
      layout: `${calibration.layout} · ${layout.label}`,
      camera_keys: [...layout.camera_keys],
      geometry: layout.camera_keys.map((key) => calibration.geometry[cameraOrder.indexOf(key)]),
    }, preserveSelection);
  }

  /** @returns {void} */
  function render() {
    const focusedId = document.activeElement?.id;
    $("camera-count").textContent = `${indices.length} / ${rig.camera_keys.length}`;
    $("camera-list").innerHTML = rig.camera_keys.map((key, i) => `<div class="camera-row" data-included="${indices.includes(i)}">
      <label title="Outline ${escapeHtml(name(i))}"><input id="include-${i}" type="checkbox" data-include="${i}" aria-label="Outline ${escapeHtml(name(i))}" ${indices.includes(i) ? "checked" : ""}></label>
      <button id="camera-${i}" data-camera="${i}" data-camera-key="${escapeHtml(key)}" aria-pressed="${i === selected}" title="Select ${escapeHtml(key)}"><span>${number(i)}</span>${escapeHtml(name(i))}</button></div>`).join("");
    $("selected-title").textContent = `${number(selected)} · ${name(selected)}`;
    renderDetails(); renderPlan(); drawSpatial();
    if (focusedId) $(focusedId)?.focus({preventScroll: true});
  }

  /** @returns {void} */
  function renderDetails() {
    const camera = rig.geometry[selected];
    $("camera-position").textContent = camera.status === "qualified" ?
      camera.origin.map((value) => value.toFixed(2)).join(" / ") + " m" : "Unavailable";
    $("camera-heading").textContent = formatHeading(camera);
    const matrix = (rows) => rows.map((row) => row.map((value) => value.toFixed(6).padStart(10)).join(" ")).join("\n");
    $("camera-rotation").textContent = camera.status === "qualified" ? matrix(camera.camera_to_rig) : "Unavailable";
    $("camera-translation").textContent = camera.status === "qualified" ? camera.origin.map((value) => value.toFixed(6)).join(" / ") + " m" : "Unavailable";
    const intrinsic = camera.intrinsics;
    $("intrinsic-fields").hidden = !intrinsic;
    $("intrinsics-empty").hidden = Boolean(intrinsic);
    if (intrinsic) {
      $("intrinsic-size").textContent = `${intrinsic.width} × ${intrinsic.height} px`;
      $("intrinsic-center").textContent = intrinsic.center.map((value) => value.toFixed(4)).join(" / ") + " px";
      $("intrinsic-affine").textContent = matrix(intrinsic.affine);
      $("intrinsic-polynomial").textContent = intrinsic.polynomial_type === "pixeldistance-to-angle" ? "θ = Σ cₖ rᵏ" : "r = Σ cₖ θᵏ";
      $("intrinsic-coefficients").textContent = intrinsic.coefficients.map((value, index) => `c${index} = ${value.toPrecision(10)}`).join("\n");
    }
  }

  /** Draw the original horizontal polygons without reconstructing FOV from names. @returns {void} */
  function renderPlan() {
    const hasFan = (i) => rig.geometry[i].status === "qualified" && rig.geometry[i].horizontal_section?.status === "qualified";
    // Keep dense calibrated paths intact across selector changes. Only a new rig
    // rebuilds geometry; controls update references, visibility and the frame.
    if (planRig !== rig) {
      const shapes = rig.geometry.map((camera, i) => {
        const path = hasFan(i) ? camera.horizontal_section.polygons_xy.map((polygon) =>
          `M${polygon.map(([x, y]) => `${-y},${-x}`).join(" L")}Z`).join(" ") : "";
        return `<path id="camera-outline-${i}" d="${path}" vector-effect="non-scaling-stroke"/>`;
      });
      $("plan-view").innerHTML = `<title></title><rect width="600" height="500" fill="#fafcf9"/>
        <defs>${shapes.join("")}</defs><g id="plan-grid"></g><g id="plan-fans">
          <use id="selected-fan" data-fan="selected" fill="#3985b4" fill-opacity=".2" stroke="none"/>
          ${rig.camera_keys.map((_, i) => `<use id="context-fan-${i}" data-fan="context" href="#camera-outline-${i}" fill="none" stroke="#82978d" stroke-width="1.2" vector-effect="non-scaling-stroke"/>`).join("")}
        </g><g id="plan-overlay"></g>`;
      planRig = rig;
    }
    const visible = [...new Set([...indices, selected])];
    const positions = visible.filter((i) => rig.geometry[i].status === "qualified").map((i) => rig.geometry[i].origin);
    const cx = (Math.min(-1, ...positions.map((p) => p[0])) + Math.max(4, ...positions.map((p) => p[0]))) / 2;
    const cy = (Math.min(-1, ...positions.map((p) => p[1])) + Math.max(1, ...positions.map((p) => p[1]))) / 2;
    const radius = Math.max(1, ...visible.filter((i) => rig.geometry[i].status === "qualified").map((i) => rig.geometry[i].horizontal_section).filter((section) =>
      section?.status === "qualified").map((section) => section.radius_m));
    const spanX = Math.max(5, ...positions.map((p) => Math.abs(p[0] - cx) * 2));
    const spanY = Math.max(2, ...positions.map((p) => Math.abs(p[1] - cy) * 2));
    const scale = Math.min(470 / (2 * radius + spanY), 390 / (2 * radius + spanX));
    const project = (p) => [300 - (p[1] - cy) * scale, 250 - (p[0] - cx) * scale];
    const point = (p) => project(p).map((v) => v.toFixed(3)).join(",");
    const attribute = (id, key, value) => { if ($(id).getAttribute(key) !== value) $(id).setAttribute(key, value); };
    attribute("plan-fans", "transform", `translate(${300 + cy * scale} ${250 + cx * scale}) scale(${scale})`);
    $("plan-view").querySelector("title").textContent = `${name(selected)}: horizontal camera support and checked outlines; ${originLabel()} at (0, 0, 0)`;
    const showFan = (id, i, visible) => {
      attribute(id, "display", visible ? "inline" : "none");
      if (visible) attribute(id, "data-camera-fov", String(i));
      else $(id).removeAttribute("data-camera-fov");
    };
    for (let i = 0; i < rig.camera_keys.length; i++) {
      showFan(`context-fan-${i}`, i, indices.includes(i) && hasFan(i));
      attribute(`context-fan-${i}`, "stroke", i === selected ? "#226ca4" : "#82978d");
    }
    attribute("selected-fan", "display", hasFan(selected) ? "inline" : "none");
    attribute("selected-fan", "data-camera-color", String(selected));
    attribute("selected-fan", "href", `#camera-outline-${selected}`);
    const grid = [], output = [];
    for (let i = -5; i <= 5; i++) {
      const x = 300 + i * 2 * scale, y = 250 + i * 2 * scale;
      if (x > 35 && x < 565) grid.push(`<path d="M${x},40V455" stroke="#e6ece4"/>`);
      if (y > 40 && y < 455) grid.push(`<path d="M35,${y}H565" stroke="#e6ece4"/>`);
    }
    // This outline only provides orientation; it is not a measured vehicle mesh.
    const car = [[-.7, -.75], [-.5, -.94], [3.55, -.94], [3.95, -.73], [3.95, .73], [3.55, .94], [-.5, .94], [-.7, .75]];
    output.push(`<polygon points="${car.map(point).join(" ")}" fill="#f0f3ee" stroke="#779080" stroke-width="1.5"/>`);
    output.push(`<path d="M${point([2.7, -.7])}L${point([3.1, 0])}L${point([2.7, .7])}" fill="none" stroke="#97aa9a"/>`);
    for (const x of [0, 3.05]) {
      for (const side of [-1, 1]) {
        const wheel = [[x - .3, side * .85], [x + .3, side * .85], [x + .3, side * 1.06], [x - .3, side * 1.06]];
        output.push(`<polygon points="${wheel.map(point).join(" ")}" fill="#657c6c"/>`);
      }
    }
    output.push(`<path d="M${point([0, -1.06])}L${point([0, 1.06])}" stroke="#a8792d" stroke-width="1"/>`);
    for (const i of indices) {
      const origin = rig.geometry[i].origin;
      if (rig.geometry[i].status !== "qualified") continue;
      const [x, y] = project(origin);
      output.push(`<circle cx="${x}" cy="${y}" r="3.5" fill="#92a79c"><title>${number(i)} · ${escapeHtml(name(i))}</title></circle>`);
    }
    for (const [i, color, direction] of [[selected, "#226ca4", -1]]) {
      const origin = rig.geometry[i].origin;
      if (rig.geometry[i].status !== "qualified") continue;
      const [x, y] = project(origin), lx = x + 33 * direction, ly = y - 20;
      output.push(`<path d="M${x},${y}L${lx},${ly}" stroke="${color}"/><circle cx="${x}" cy="${y}" r="5" fill="${color}" stroke="white"/><circle cx="${lx}" cy="${ly}" r="12" fill="${color}"/><text x="${lx}" y="${ly + 4}" fill="white" text-anchor="middle" font-size="10">${number(i)}</text>`);
    }
    const [ox, oy] = project([0, 0]);
    const labelX = Math.max(40, Math.min(440, ox + 36)), labelY = Math.max(58, Math.min(415, oy + 34));
    output.push(`<g id="plan-origin" font-family="system-ui,sans-serif" font-size="11">
      <title>${originLabel()} · Origin (0, 0, 0)</title>
      <path d="M${ox},${oy}L${labelX - 8},${labelY - 5}" fill="none" stroke="#a8792d"/>
      <path d="M${ox},${oy - 4}l4,4l-4,4l-4,-4Z" fill="#a8792d" stroke="#fafcf9" stroke-width="1.2"/>
      <text x="${labelX}" y="${labelY}" fill="#805a1f">${originLabel()}</text>
      <text x="${labelX}" y="${labelY + 16}" fill="#657c6c" font-size="10">Origin (0, 0, 0)</text></g>`);
    output.push(`<g font-family="system-ui,sans-serif" font-size="10" fill="#657c6c"><text x="300" y="22" text-anchor="middle">FRONT (+X) ↑</text><text x="300" y="486" text-anchor="middle">REAR (−X)</text><text x="12" y="250">LEFT</text><text x="588" y="250" text-anchor="end">RIGHT</text><path d="M22,463h${2 * scale}m0,-4v8M22,459v8" stroke="#657c6c"/><text x="${22 + scale}" y="453" text-anchor="middle">2 m</text><text x="578" y="466" text-anchor="end">XY projection</text></g>`);
    $("plan-grid").innerHTML = grid.join("");
    $("plan-overlay").innerHTML = output.join("");
    const g = rig.geometry[selected], section = g.horizontal_section;
    $("plan-caption").textContent = g.status === "qualified" && section?.status === "qualified" ?
      `${section.total_angle_degrees.toFixed(1)}° horizontal cut · mounting height ${section.height_m.toFixed(2)} m · heading ${formatHeading(g)}.` :
      "No qualified horizontal section for the selected camera. Missing geometry is not drawn.";
  }

  /** Coalesce interactions and optional rotation into one draw per animation frame. @returns {void} */
  function drawSpatial() {
    if (spatialFrame) return;
    spatialFrame = requestAnimationFrame((time) => {
      spatialFrame = 0;
      const rotating = autoRotate && !drag && !document.hidden;
      if (rotating) {
        if (lastRotationTime) yaw += Math.min(time - lastRotationTime, 50) * .00022;
        lastRotationTime = time;
      } else lastRotationTime = 0;
      drawSpatialFrame();
      if (rotating) drawSpatial();
    });
  }

  /** A schematic vehicle with its rear axle center at the rig origin and +X at the hood.
   * @returns {{points: number[][], color: string}[]} */
  function vehicleFaces() {
    const faces = [];
    // The template's rear axle center is [-0.3, 0, 0.36]; align it to [0, 0, 0].
    const face = (points, color) => faces.push({points: points.map(([x, y, z]) => [x + .3, y, z - .36]), color});
    const bottom = [[-1, -.75, .35], [-.8, -.94, .35], [3.25, -.94, .35], [3.65, -.73, .38],
      [3.65, .73, .38], [3.25, .94, .35], [-.8, .94, .35], [-1, .75, .35]];
    const top = bottom.map(([x, y]) => [x, y, x > 3.3 ? .7 : x > 3 ? .88 : .94]);
    face(top, "#cbd7db");
    for (let i = 0; i < bottom.length; i++) {
      const next = (i + 1) % bottom.length;
      face([bottom[i], bottom[next], top[next], top[i]], i < 4 ? "#a5b5bd" : "#d3dce0");
    }
    const cabin = [[-.5, -.83, .94], [2.05, -.83, .94], [2.05, .83, .94], [-.5, .83, .94]];
    const roof = [[-.15, -.7, 1.48], [1.35, -.7, 1.5], [1.35, .7, 1.5], [-.15, .7, 1.48]];
    face(roof, "#e6ecec");
    for (let i = 0; i < 4; i++) {
      const next = (i + 1) % 4;
      face([cabin[i], cabin[next], roof[next], roof[i]], i === 1 ? "#507888" : "#345260");
    }
    for (const side of [-1, 1]) {
      // Window pillars and four tires make the schematic readable while orbiting.
      face([[.48, side * .84, .95], [.59, side * .84, .95], [.59, side * .71, 1.49], [.48, side * .71, 1.49]], "#bccbd1");
      for (const x of [-.3, 2.75]) {
        const ring = (y, radius) => Array.from({length: 20}, (_, i) => {
          const angle = i / 20 * Math.PI * 2;
          return [x + Math.cos(angle) * radius, y, .36 + Math.sin(angle) * radius];
        });
        const inner = ring(side * .82, .35), outer = ring(side * 1.05, .35);
        face(inner, "#182126"); face(outer, "#202d33");
        for (let i = 0; i < 20; i++) face([inner[i], inner[(i + 1) % 20], outer[(i + 1) % 20], outer[i]], "#26343a");
        face(ring(side * 1.06, .18), "#80949e");
      }
      face([[3.66, side * .34, .55], [3.66, side * .65, .55], [3.66, side * .65, .69], [3.66, side * .34, .69]], "#fff2c7");
      face([[-1.01, side * .4, .64], [-1.01, side * .7, .64], [-1.01, side * .7, .84], [-1.01, side * .4, .84]], "#b9655c");
    }
    return faces;
  }
  const vehicle = vehicleFaces();

  /** Orthographic coverage with a solid vehicle and only checked boundary lines. @returns {void} */
  function drawSpatialFrame() {
    const canvas = $("spatial-view"), rect = canvas.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 3);
    const pixelWidth = Math.round(rect.width * dpr), pixelHeight = Math.round(rect.height * dpr);
    if (canvas.width !== pixelWidth) canvas.width = pixelWidth;
    if (canvas.height !== pixelHeight) canvas.height = pixelHeight;
    const context = canvas.getContext("2d");
    context.setTransform(dpr, 0, 0, dpr, 0, 0);
    const width = rect.width, height = rect.height, length = Number($("ray-length").value);
    context.fillStyle = "#15262b"; context.fillRect(0, 0, width, height);
    $("ray-value").textContent = `${length} m`;
    const visible = [...new Set([...indices, selected])].filter((i) => rig.geometry[i].status === "qualified");
    const rayLimit = Math.min(256, Math.floor(MAX_RENDER_RAYS / Math.max(1, visible.length)));
    const cameras = visible.map((i) => {
      const g = rig.geometry[i];
      let cached = rayCache.get(g);
      if (!cached || cached.limit !== rayLimit) {
        cached = {limit: rayLimit, rays: sampleOutline(g.outline_rays, rayLimit)};
        rayCache.set(g, cached);
      }
      return {i, origin: g.origin, end: cached.rays.map((ray) => ray.map((v, k) => g.origin[k] + v * length))};
    });
    const rotate = ([x, y, z]) => {
      const across = -Math.sin(yaw) * x + Math.cos(yaw) * y;
      const forward = Math.cos(yaw) * x + Math.sin(yaw) * y;
      return [-across, -(Math.sin(pitch) * forward + Math.cos(pitch) * z), Math.cos(pitch) * forward - Math.sin(pitch) * z];
    };
    const frame = [...vehicle.flatMap((face) => face.points), [5, 0, -.35], ...cameras.flatMap((camera) => [camera.origin, ...camera.end])];
    const center = [0, 1, 2].map((k) => (Math.min(...frame.map((p) => p[k])) + Math.max(...frame.map((p) => p[k]))) / 2);
    const radius = Math.max(1, ...frame.map((p) => Math.hypot(...p.map((v, k) => v - center[k]))));
    // A fixed sphere keeps orbiting smooth without zooming or recentering each frame.
    const scale = (Math.min(width, height) - 64) / (2 * radius) * zoom;
    const project = (point) => rotate(point.map((v, k) => v - center[k])).slice(0, 2).map((v, k) => (k ? height : width) / 2 + v * scale);
    const path = (points) => {
      context.beginPath();
      points.forEach((p, i) => { const xy = project(p); i ? context.lineTo(...xy) : context.moveTo(...xy); });
    };
    const line = (points, color, thickness = 1) => {
      path(points); context.strokeStyle = color; context.lineWidth = thickness; context.stroke();
    };
    const fill = (points, color) => {
      path(points); context.closePath(); context.fillStyle = color; context.fill();
    };
    // A quiet ground reference replaces long axes and a grid through the camera volumes.
    for (let x = -2; x <= 5; x++) line([[x, -2, -.35], [x, 2, -.35]], "#20373c");
    for (let y = -2; y <= 2; y++) line([[-2, y, -.35], [5, y, -.35]], "#20373c");
    fill([[-1, -1.15, -.35], [4.1, -1.15, -.35], [4.1, 1.15, -.35], [-1, 1.15, -.35]], "#101e22");
    const active = cameras.find((camera) => camera.i === selected);
    if (active) {
      // Color only the selected camera's calibrated coverage; no internal mesh edges.
      for (let i = 0; i < active.end.length; i++) {
        fill([active.origin, active.end[i], active.end[(i + 1) % active.end.length]], "rgba(88,172,223,.12)");
      }
    }
    for (const camera of cameras) {
      if (!indices.includes(camera.i)) continue;
      line([...camera.end, camera.end[0]], camera.i === selected ? "#69b6e8" : "#627d86", camera.i === selected ? 1.6 : 1);
      // Four corner guides belong only to the selected outline; other cameras keep one clean perimeter.
      if (camera.i === selected) {
        for (let k = 0; k < 4; k++) line([camera.origin, camera.end[Math.floor(k * camera.end.length / 4)]], "#69b6e8", .8);
      }
    }
    const depth = (face) => face.points.reduce((sum, p) => sum + rotate(p)[2], 0) / face.points.length;
    for (const face of [...vehicle].sort((a, b) => depth(b) - depth(a))) {
      fill(face.points, face.color);
      context.strokeStyle = "rgba(21,38,43,.28)"; context.lineWidth = .65; context.stroke();
    }
    line([[4.3, 0, -.34], [4.95, 0, -.34]], "#a2b9ac", 1.5);
    line([[4.7, -.16, -.34], [4.95, 0, -.34], [4.7, .16, -.34]], "#a2b9ac", 1.5);
    const front = project([4.95, 0, -.34]);
    context.fillStyle = "#a2b9ac"; context.font = "10px system-ui"; context.fillText("FRONT", front[0] + 7, front[1] + 3);
    for (const camera of cameras) {
      const [x, y] = project(camera.origin);
      context.fillStyle = camera.i === selected ? "#78c4f1" : "#93aaa1";
      context.beginPath(); context.arc(x, y, camera.i === selected ? 4.5 : 2.5, 0, 2 * Math.PI); context.fill();
      if (camera.i === selected) {
        context.strokeStyle = "#e1f4ff"; context.lineWidth = 1.5; context.stroke();
        context.font = "bold 12px system-ui"; context.fillText(number(camera.i), x + 9, y - 8);
      }
    }
    // Keep the origin annotation visible as the view rotates, even behind the schematic body.
    const [ox, oy] = project([0, 0, 0]);
    const labelX = Math.max(16, Math.min(width - 135, ox + 34)), labelY = Math.max(25, Math.min(height - 30, oy + 44));
    context.beginPath(); context.moveTo(ox, oy); context.lineTo(labelX - 8, labelY - 5);
    context.strokeStyle = "#e6bd73"; context.lineWidth = 1; context.stroke();
    context.beginPath(); context.moveTo(ox, oy - 4); context.lineTo(ox + 4, oy); context.lineTo(ox, oy + 4); context.lineTo(ox - 4, oy); context.closePath();
    context.fillStyle = "#e6bd73"; context.fill(); context.strokeStyle = "#15262b"; context.lineWidth = 1.2; context.stroke();
    context.fillStyle = "#15262b"; context.fillRect(labelX - 4, labelY - 13, 126, 34);
    context.fillStyle = "#e6bd73"; context.font = "11px system-ui"; context.fillText(originLabel(), labelX, labelY);
    context.fillStyle = "#a2b9ac"; context.font = "10px system-ui"; context.fillText("Origin (0, 0, 0)", labelX, labelY + 16);
    canvas.dataset.renderCount = String(Number(canvas.dataset.renderCount || 0) + 1);
    canvas.dataset.rayCount = String(cameras.reduce((sum, camera) => sum + camera.end.length, 0));
    canvas.dataset.orientation = `${yaw.toFixed(3)},${pitch.toFixed(3)},${zoom.toFixed(3)}`;
  }

  $("rig-select").innerHTML = examples.map((example, i) => `<option value="${i}">${escapeHtml(example.layout)}</option>`).join("");
  $("rig-select").addEventListener("change", () => { importVersion++; loadCalibration(true); });
  $("layout-select").addEventListener("change", () => { importVersion++; layoutIndex = Number($("layout-select").value); loadCalibration(true); });
  $("select-all").addEventListener("click", () => setIndices(rig.camera_keys.map((_, i) => i)));
  $("deselect-all").addEventListener("click", () => setIndices([]));
  $("auto-rotate").addEventListener("click", () => {
    autoRotate = !autoRotate;
    $("auto-rotate").setAttribute("aria-pressed", String(autoRotate));
    lastRotationTime = 0;
    drawSpatial();
  });
  document.addEventListener("visibilitychange", () => { lastRotationTime = 0; if (!document.hidden) drawSpatial(); });
  $("camera-list").addEventListener("change", (event) => {
    if (!event.target.matches("[data-include]")) return;
    const i = Number(event.target.dataset.include);
    setIndices(rig.camera_keys.map((_, k) => k).filter((k) => k === i ? event.target.checked : indices.includes(k)));
  });
  document.addEventListener("click", (event) => {
    const button = event.target.closest("button");
    if (!button) return;
    if (button.hasAttribute("data-camera")) {
      selected = Number(button.dataset.camera);
      zoom = 1;
      render();
    }
    if (button.hasAttribute("data-view")) {
      yaw = button.dataset.view === "orbit" ? .6 : button.dataset.view === "side" ? Math.PI / 2 : 0;
      pitch = button.dataset.view === "orbit" ? .65 : button.dataset.view === "top" ? Math.PI / 2 : 0;
      zoom = 1; drawSpatial();
    }
  });
  $("ray-length").addEventListener("input", drawSpatial);
  $("export-rig").addEventListener("click", () => download(rig, "camera-rig.json"));
  $("export-svg").addEventListener("click", () => {
    const svg = $("plan-view").cloneNode(true); svg.setAttribute("width", "600"); svg.setAttribute("height", "500");
    download(new XMLSerializer().serializeToString(svg), "camera-rig.svg", "image/svg+xml");
  });
  $("import-button").addEventListener("click", () => $("import-file").click());
  $("import-file").addEventListener("change", async (event) => {
    const file = event.target.files[0], version = ++importVersion;
    if (!file) return;
    try {
      if (file.size > 4 * 1024 * 1024) throw new Error("Rig JSON must be smaller than 4 MiB.");
      const next = validateRig(JSON.parse(await file.text()));
      if (version !== importVersion) return;
      examples.push(next);
      const option = document.createElement("option"); option.value = examples.length - 1; option.textContent = `Imported · ${file.name}`;
      $("rig-select").append(option); $("rig-select").value = option.value;
      loadCalibration();
      $("status").textContent = "Loaded locally. Imported geometry is supplied by the file; this viewer does not certify its derivation.";
    } catch (error) {
      if (version === importVersion) $("status").textContent = `Could not open rig: ${error.message} The current rig is unchanged.`;
    } finally { event.target.value = ""; }
  });
  const canvas = $("spatial-view");
  canvas.addEventListener("pointerdown", (event) => { drag = [event.clientX, event.clientY]; canvas.setPointerCapture(event.pointerId); });
  canvas.addEventListener("pointermove", (event) => {
    if (!drag) return;
    yaw += (event.clientX - drag[0]) * .008; pitch = Math.max(-1.4, Math.min(Math.PI / 2, pitch + (event.clientY - drag[1]) * .008));
    drag = [event.clientX, event.clientY]; drawSpatial();
  });
  for (const event of ["pointerup", "pointercancel", "lostpointercapture"]) canvas.addEventListener(event, () => {
    if (!drag) return;
    drag = null; lastRotationTime = 0;
    if (autoRotate) drawSpatial();
  });
  canvas.addEventListener("wheel", (event) => { event.preventDefault(); zoom = Math.max(.4, Math.min(3, zoom * Math.exp(-event.deltaY * .001))); drawSpatial(); }, {passive: false});
  canvas.addEventListener("keydown", (event) => {
    if (!["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "+", "=", "-"].includes(event.key)) return;
    event.preventDefault();
    if (event.key === "ArrowLeft") yaw -= .12;
    if (event.key === "ArrowRight") yaw += .12;
    if (event.key === "ArrowUp") pitch += .12;
    if (event.key === "ArrowDown") pitch -= .12;
    if (["+", "="].includes(event.key)) zoom *= 1.1;
    if (event.key === "-") zoom /= 1.1;
    pitch = Math.max(-1.4, Math.min(Math.PI / 2, pitch)); zoom = Math.max(.4, Math.min(3, zoom));
    drawSpatial();
  });
  new ResizeObserver(drawSpatial).observe(canvas);
  loadCalibration();
})();
