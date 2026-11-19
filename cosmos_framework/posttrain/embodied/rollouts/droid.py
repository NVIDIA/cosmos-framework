# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Connect a RoboLab DROID simulator to the public Cosmos WebSocket policy server."""

from __future__ import annotations

import numpy as np

from cosmos_framework.posttrain.embodied.rollout import RolloutInfo


class DroidEndpoint:
    def __init__(self, host: str, port: int) -> None:
        from robolab.eval.websocket_transport import MsgPackWebSocketTransport

        self.client = MsgPackWebSocketTransport(f"ws://{host}:{port}")
        self.client.connect()

    def describe(self):
        return RolloutInfo(name="cosmos-droid", action_dim=8, horizon=32, embodiment="robolab_droid")

    def close(self):
        self.client.close()

    def act(self, observations):
        import torch
        from robolab.core.utils.image_utils import resize_with_pad
        from torch.nn.functional import interpolate

        chunks = []
        for obs in observations:
            views = []
            for name in ("wrist_cam", "over_shoulder_left_camera", "over_shoulder_right_camera"):
                image = np.asarray(obs[name])
                if image.ndim != 3 or image.shape[-1] not in (3, 4) or image.dtype != np.uint8:
                    raise ValueError(f"Invalid RGB/RGBA image for {name}")
                views.append(resize_with_pad(np.ascontiguousarray(image[..., :3]), 360, 640))
            sides = []
            for image in views[1:]:
                tensor = torch.from_numpy(image).permute(2, 0, 1)[None].float()
                small = interpolate(tensor, size=(180, 320), mode="bilinear", align_corners=False)
                sides.append(small[0].permute(1, 2, 0).numpy().astype(np.uint8))
            frame = np.concatenate((views[0], np.concatenate(sides, axis=1)))
            state = np.asarray(obs["state"], dtype=np.float32)
            if state.shape != (8,) or not np.isfinite(state).all():
                raise ValueError("DROID expects seven joints and one gripper position")
            response = self.client.request(
                {
                    "observation/image": frame,
                    "observation/joint_position": state[:7],
                    "observation/gripper_position": state[7:],
                    "prompt": obs["prompt"],
                }
            )
            action = np.array(response["action"], dtype=np.float32, copy=True)
            if action.ndim != 2 or not len(action) or action.shape[1] != 8 or not np.isfinite(action).all():
                raise ValueError("Invalid DROID action chunk")
            action[:, -1] = action[:, -1] > 0.5
            chunks.append(action)
        return chunks
