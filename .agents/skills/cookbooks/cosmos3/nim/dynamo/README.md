<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Cosmos3 Reasoner With Dynamo

Refer to [Dynamo Configuration](../configuration.md#dynamo-configuration) for worker
enablement, discovery, proxy settings, and
[streaming controls](../configuration.md#dynamo-streaming).

Choose a deployment:

| Guide | Manifest | Frontend and Behavior |
| --- | --- | --- |
| [Regular Dynamo](dynamo_deployment.md) | [reasoner.yaml](reasoner.yaml) | Pinned Dynamo frontend running `dynamo.frontend`; Chat Completions, no persistent video sessions. |
| [Dynamo With Streaming](dynamo_with_streaming.md) | [reasoner_with_streaming.yaml](reasoner_with_streaming.yaml) | Same NIM image for frontend and workers; frontend runs `streaming.dynamo.frontend` for Chat Completions plus REST video sessions. |

Both use [model-cache.yaml](model-cache.yaml), the
[model-download Job](model-download.yaml), and an NVIDIA Dynamo operator serving the
`nvidia.com/v1alpha1` `spec.services` schema. Both use the same NIM image for
the download Job and workers. The regular frontend uses the pinned Dynamo
image. The streaming frontend uses the NIM image. The manifests use
`nvcr.io/nim/nvidia/cosmos3:2.1.0`. Keep the same image in your chosen variant and the
download Job. Validate streaming sessions separately. Replace `<angle-bracket>` placeholders before applying. These are alternative
manifests for the same `cosmos3-reasoner` deployment, so apply only one.
Switching variants replaces pods and loses active streaming sessions.

Here, **streaming** means persistent video sessions, not token streaming in a
Chat Completions response. The Dynamo video-session path uses REST only.
