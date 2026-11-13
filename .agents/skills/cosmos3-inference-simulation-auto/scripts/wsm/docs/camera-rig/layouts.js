/* Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. */
"use strict";

const CAMERA_RIG_LAYOUTS = [
  {"label": "1 cam", "camera_keys": ["front_wide"]},
  {"label": "2 cam", "camera_keys": ["front_wide", "front_tele"]},
  {"label": "4 cam A", "camera_keys": ["front_fisheye", "left_fisheye", "right_fisheye", "rear_fisheye"]},
  {"label": "4 cam B", "camera_keys": ["front_wide", "cross_right", "rear_tele", "cross_left"]},
  {"label": "5 cam A", "camera_keys": ["front_wide", "cross_right", "rear_right", "rear_left", "cross_left"]},
  {"label": "5 cam B", "camera_keys": ["front_wide", "front_fisheye", "left_fisheye", "right_fisheye", "rear_fisheye"]},
  {"label": "6 cam A", "camera_keys": ["front_wide", "cross_right", "rear_right", "rear_left", "cross_left", "rear_fisheye"]},
  {"label": "6 cam B", "camera_keys": ["front_wide", "cross_right", "cross_left", "left_fisheye", "right_fisheye", "rear_fisheye"]},
  {"label": "6 cam C", "camera_keys": ["front_wide", "cross_right", "rear_right", "rear_tele", "rear_left", "cross_left"]},
  {"label": "6 cam D", "camera_keys": ["front_wide", "cross_right", "rear_right", "rear_tele", "rear_left", "front_tele"]},
  {"label": "6 cam E", "camera_keys": ["front_wide", "cross_right", "rear_right", "rear_tele", "cross_left", "front_tele"]},
  {"label": "6 cam F", "camera_keys": ["front_wide", "cross_right", "rear_right", "rear_left", "cross_left", "front_tele"]},
  {"label": "6 cam G", "camera_keys": ["front_wide", "cross_right", "rear_tele", "rear_left", "cross_left", "front_tele"]},
  {"label": "6 cam H", "camera_keys": ["front_wide", "rear_right", "rear_tele", "rear_left", "cross_left", "front_tele"]},
  {"label": "6 cam I", "camera_keys": ["cross_right", "rear_right", "rear_tele", "rear_left", "cross_left", "front_tele"]},
  {"label": "7 cam A", "camera_keys": ["front_wide", "cross_right", "rear_right", "rear_left", "cross_left", "front_tele", "rear_fisheye"]},
  {"label": "7 cam B", "camera_keys": ["front_wide", "cross_right", "rear_right", "rear_tele", "rear_left", "cross_left", "front_tele"]},
  {"label": "8 cam A", "camera_keys": ["front_wide", "cross_right", "rear_right", "rear_left", "cross_left", "front_tele", "front_fisheye", "rear_fisheye"]},
  {"label": "8 cam B", "camera_keys": ["front_wide", "cross_right", "rear_right", "rear_tele", "rear_left", "cross_left", "front_tele", "front_fisheye"]},
  {"label": "10 cam A", "camera_keys": ["front_wide", "cross_right", "rear_right", "rear_left", "cross_left", "front_tele", "front_fisheye", "left_fisheye", "right_fisheye", "rear_fisheye"]},
  {"label": "10 cam B", "camera_keys": ["front_wide", "cross_right", "rear_right", "rear_tele", "rear_left", "cross_left", "front_tele", "front_fisheye", "left_fisheye", "right_fisheye"]},
  {"label": "11 cam", "camera_keys": ["front_wide", "cross_right", "rear_right", "rear_tele", "rear_left", "cross_left", "front_tele", "front_fisheye", "left_fisheye", "right_fisheye", "rear_fisheye"]}
];
if (typeof module !== "undefined") module.exports = CAMERA_RIG_LAYOUTS;
