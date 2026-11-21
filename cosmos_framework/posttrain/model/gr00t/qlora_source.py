# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Exact public N1.7 LIBERO/backbone source pin for native QLoRA qualification.

File identities come from the retained gr00t-public-assets-v1 staging attestation
(2026-10-31). Backbone files are byte-identical public metadata, not substituted weights.
"""

from __future__ import annotations

from pathlib import Path

BASE_REPOSITORY = "nvidia/GR00T-N1.7-LIBERO"
BASE_REVISION = "2ea293aa20ba7cf5bbf3ba17a5fbcb1a01cbfe21"
BACKBONE_REPOSITORY = "nvidia/Cosmos-Reason2-2B"
BACKBONE_REVISION = "9ce19a195e423419c349abfc86fd07178b230561"
PUBLIC_METADATA_REPOSITORY = "Qwen/Qwen3-VL-2B-Instruct"
PUBLIC_METADATA_REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"
BASE_FILES = {
    "config.json": "ce763f74ad1cda2ca9e67491229cbc5810d83cae675cb4dad839a69a9731fd97",
    "embodiment_id.json": "abc4a749389837416a102d9455a8c089336f2e417afb18565703b9944974bc35",
    "model-00001-of-00002.safetensors": "b7c7314a989ae3b577e3265e3f25c42a018ab4a476ab99e4d0fa61266d38a878",
    "model-00002-of-00002.safetensors": "ac70555aee124ef20d23473c85dbcfda42dc8d9f859384822e8aef19fd2d45dd",
    "model.safetensors.index.json": "407804ea5a62f4f8823f48811ae0edbb82fac101e9cf4d7273e6e2f692bb4d59",
    "processor_config.json": "545bb44c7d8caaf5ac82df55b90831271a1469a4e48adcc9a5f469f064f4e121",
    "statistics.json": "853bd068c7a5be4c7657fae634b654de38ea3e64c5e57409cbd3f4b57b6461c2",
}
BACKBONE_FILES = {
    "chat_template.json": "6f8a6a55027e3da5160105556cda5dd69f6423f1c32645f6730d32de7773d0c4",
    "config.json": "bec4b3d446efa05807365c9e1cec03ac590836879d02f3a6da879971154bdd3b",
    "generation_config.json": "1e241830b48b397cb0900101421df5450baddc7adf01e5fc86b5615865f3bae4",
    "merges.txt": "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3",
    "preprocessor_config.json": "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516",
    "tokenizer.json": "a5d85b6dcc535e6b93115a9ef287e6132fdbf30270da6218194ba742261173c7",
    "tokenizer_config.json": "c2da771801886ad9ae98181793ffd3dfb7f1af30f6f7c6a4e15d7dbba52e2399",
    "video_preprocessor_config.json": "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
}


def recorded_source():
    return {
        "format": "cosmos.gr00t-native-source.v1",
        "base": {"repository": BASE_REPOSITORY, "revision": BASE_REVISION, "files": dict(BASE_FILES)},
        "backbone": {
            "repository": BACKBONE_REPOSITORY,
            "revision": BACKBONE_REVISION,
            "public_identical_metadata": {
                "repository": PUBLIC_METADATA_REPOSITORY,
                "revision": PUBLIC_METADATA_REVISION,
            },
            "files": dict(BACKBONE_FILES),
            "weights": "from-complete-GR00T-checkpoint",
        },
    }


def verify_metadata(checkpoint, backbone):
    from cosmos_framework.posttrain.model.gr00t.source import _metadata

    native = _metadata(Path(checkpoint), gr00t=True)
    expected = {
        name: digest
        for name, digest in BASE_FILES.items()
        if not name.endswith(".safetensors") and name != "model.safetensors.index.json"
    }
    if native != expected:
        raise ValueError("GR00T native processor/config/statistics differ from the pinned LIBERO checkpoint")
    if _metadata(Path(backbone)) != BACKBONE_FILES:
        raise ValueError("GR00T backbone config/tokenizer differ from exact Reason2-2B metadata")


def pinned_source_identity(checkpoint, backbone):
    from cosmos_framework.posttrain.model.gr00t.source import _metadata, _weights

    verify_metadata(checkpoint, backbone)
    observed = {**_metadata(Path(checkpoint), gr00t=True), **_weights(Path(checkpoint))}
    if observed != BASE_FILES:
        raise ValueError("GR00T weights differ from pinned nvidia/GR00T-N1.7-LIBERO revision " + BASE_REVISION)
    return recorded_source()
