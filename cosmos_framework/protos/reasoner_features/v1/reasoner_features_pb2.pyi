# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from collections.abc import Iterable as _Iterable
from collections.abc import Mapping as _Mapping
from typing import ClassVar as _ClassVar
from typing import Optional as _Optional
from typing import Union as _Union

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper

DESCRIPTOR: _descriptor.FileDescriptor

class DType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    DTYPE_UNSPECIFIED: _ClassVar[DType]
    DTYPE_INT32: _ClassVar[DType]
    DTYPE_INT64: _ClassVar[DType]
    DTYPE_FLOAT16: _ClassVar[DType]
    DTYPE_BFLOAT16: _ClassVar[DType]
    DTYPE_FLOAT32: _ClassVar[DType]
    DTYPE_FLOAT64: _ClassVar[DType]

class TensorKind(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    TENSOR_KIND_UNSPECIFIED: _ClassVar[TensorKind]
    TENSOR_KIND_CROSS_K: _ClassVar[TensorKind]
    TENSOR_KIND_CROSS_V: _ClassVar[TensorKind]

DTYPE_UNSPECIFIED: DType
DTYPE_INT32: DType
DTYPE_INT64: DType
DTYPE_FLOAT16: DType
DTYPE_BFLOAT16: DType
DTYPE_FLOAT32: DType
DTYPE_FLOAT64: DType
TENSOR_KIND_UNSPECIFIED: TensorKind
TENSOR_KIND_CROSS_K: TensorKind
TENSOR_KIND_CROSS_V: TensorKind

class TensorPayload(_message.Message):
    __slots__ = ("dtype", "shape", "data")
    DTYPE_FIELD_NUMBER: _ClassVar[int]
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    dtype: DType
    shape: _containers.RepeatedScalarFieldContainer[int]
    data: bytes
    def __init__(
        self,
        dtype: _Optional[_Union[DType, str]] = ...,
        shape: _Optional[_Iterable[int]] = ...,
        data: _Optional[bytes] = ...,
    ) -> None: ...

class CacheIdentity(_message.Message):
    __slots__ = ("reasoner", "tokenizer", "framing")
    REASONER_FIELD_NUMBER: _ClassVar[int]
    TOKENIZER_FIELD_NUMBER: _ClassVar[int]
    FRAMING_FIELD_NUMBER: _ClassVar[int]
    reasoner: str
    tokenizer: str
    framing: str
    def __init__(
        self, reasoner: _Optional[str] = ..., tokenizer: _Optional[str] = ..., framing: _Optional[str] = ...
    ) -> None: ...

class FeatureSignature(_message.Message):
    __slots__ = ("dtype", "num_layers", "num_kv_heads", "head_dim")
    DTYPE_FIELD_NUMBER: _ClassVar[int]
    NUM_LAYERS_FIELD_NUMBER: _ClassVar[int]
    NUM_KV_HEADS_FIELD_NUMBER: _ClassVar[int]
    HEAD_DIM_FIELD_NUMBER: _ClassVar[int]
    dtype: DType
    num_layers: int
    num_kv_heads: int
    head_dim: int
    def __init__(
        self,
        dtype: _Optional[_Union[DType, str]] = ...,
        num_layers: _Optional[int] = ...,
        num_kv_heads: _Optional[int] = ...,
        head_dim: _Optional[int] = ...,
    ) -> None: ...

class Capability(_message.Message):
    __slots__ = ("name", "version")
    NAME_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    name: str
    version: int
    def __init__(self, name: _Optional[str] = ..., version: _Optional[int] = ...) -> None: ...

class GetInfoRequest(_message.Message):
    __slots__ = ("protocol_version",)
    PROTOCOL_VERSION_FIELD_NUMBER: _ClassVar[int]
    protocol_version: int
    def __init__(self, protocol_version: _Optional[int] = ...) -> None: ...

class GetInfoResponse(_message.Message):
    __slots__ = (
        "protocol_version",
        "service_instance_id",
        "identity",
        "signature",
        "capabilities",
        "max_batch_tokens",
        "max_queued_tokens",
        "max_chunk_bytes",
        "ready",
        "max_requests",
    )
    PROTOCOL_VERSION_FIELD_NUMBER: _ClassVar[int]
    SERVICE_INSTANCE_ID_FIELD_NUMBER: _ClassVar[int]
    IDENTITY_FIELD_NUMBER: _ClassVar[int]
    SIGNATURE_FIELD_NUMBER: _ClassVar[int]
    CAPABILITIES_FIELD_NUMBER: _ClassVar[int]
    MAX_BATCH_TOKENS_FIELD_NUMBER: _ClassVar[int]
    MAX_QUEUED_TOKENS_FIELD_NUMBER: _ClassVar[int]
    MAX_CHUNK_BYTES_FIELD_NUMBER: _ClassVar[int]
    READY_FIELD_NUMBER: _ClassVar[int]
    MAX_REQUESTS_FIELD_NUMBER: _ClassVar[int]
    protocol_version: int
    service_instance_id: str
    identity: CacheIdentity
    signature: FeatureSignature
    capabilities: _containers.RepeatedCompositeFieldContainer[Capability]
    max_batch_tokens: int
    max_queued_tokens: int
    max_chunk_bytes: int
    ready: bool
    max_requests: int
    def __init__(
        self,
        protocol_version: _Optional[int] = ...,
        service_instance_id: _Optional[str] = ...,
        identity: _Optional[_Union[CacheIdentity, _Mapping]] = ...,
        signature: _Optional[_Union[FeatureSignature, _Mapping]] = ...,
        capabilities: _Optional[_Iterable[_Union[Capability, _Mapping]]] = ...,
        max_batch_tokens: _Optional[int] = ...,
        max_queued_tokens: _Optional[int] = ...,
        max_chunk_bytes: _Optional[int] = ...,
        ready: bool = ...,
        max_requests: _Optional[int] = ...,
    ) -> None: ...

class ReasonerFeatureRequest(_message.Message):
    __slots__ = ("sample_key", "token_ids", "position_ids", "causal_offsets", "fingerprint")
    SAMPLE_KEY_FIELD_NUMBER: _ClassVar[int]
    TOKEN_IDS_FIELD_NUMBER: _ClassVar[int]
    POSITION_IDS_FIELD_NUMBER: _ClassVar[int]
    CAUSAL_OFFSETS_FIELD_NUMBER: _ClassVar[int]
    FINGERPRINT_FIELD_NUMBER: _ClassVar[int]
    sample_key: str
    token_ids: TensorPayload
    position_ids: TensorPayload
    causal_offsets: TensorPayload
    fingerprint: str
    def __init__(
        self,
        sample_key: _Optional[str] = ...,
        token_ids: _Optional[_Union[TensorPayload, _Mapping]] = ...,
        position_ids: _Optional[_Union[TensorPayload, _Mapping]] = ...,
        causal_offsets: _Optional[_Union[TensorPayload, _Mapping]] = ...,
        fingerprint: _Optional[str] = ...,
    ) -> None: ...

class GenerateRequest(_message.Message):
    __slots__ = ("protocol_version", "request_id", "identity", "requests")
    PROTOCOL_VERSION_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    IDENTITY_FIELD_NUMBER: _ClassVar[int]
    REQUESTS_FIELD_NUMBER: _ClassVar[int]
    protocol_version: int
    request_id: str
    identity: CacheIdentity
    requests: _containers.RepeatedCompositeFieldContainer[ReasonerFeatureRequest]
    def __init__(
        self,
        protocol_version: _Optional[int] = ...,
        request_id: _Optional[str] = ...,
        identity: _Optional[_Union[CacheIdentity, _Mapping]] = ...,
        requests: _Optional[_Iterable[_Union[ReasonerFeatureRequest, _Mapping]]] = ...,
    ) -> None: ...

class GenerateHeader(_message.Message):
    __slots__ = (
        "protocol_version",
        "request_id",
        "identity",
        "signature",
        "fingerprints",
        "causal_offsets",
        "total_tensor_bytes",
        "service_instance_id",
    )
    PROTOCOL_VERSION_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    IDENTITY_FIELD_NUMBER: _ClassVar[int]
    SIGNATURE_FIELD_NUMBER: _ClassVar[int]
    FINGERPRINTS_FIELD_NUMBER: _ClassVar[int]
    CAUSAL_OFFSETS_FIELD_NUMBER: _ClassVar[int]
    TOTAL_TENSOR_BYTES_FIELD_NUMBER: _ClassVar[int]
    SERVICE_INSTANCE_ID_FIELD_NUMBER: _ClassVar[int]
    protocol_version: int
    request_id: str
    identity: CacheIdentity
    signature: FeatureSignature
    fingerprints: _containers.RepeatedScalarFieldContainer[str]
    causal_offsets: TensorPayload
    total_tensor_bytes: int
    service_instance_id: str
    def __init__(
        self,
        protocol_version: _Optional[int] = ...,
        request_id: _Optional[str] = ...,
        identity: _Optional[_Union[CacheIdentity, _Mapping]] = ...,
        signature: _Optional[_Union[FeatureSignature, _Mapping]] = ...,
        fingerprints: _Optional[_Iterable[str]] = ...,
        causal_offsets: _Optional[_Union[TensorPayload, _Mapping]] = ...,
        total_tensor_bytes: _Optional[int] = ...,
        service_instance_id: _Optional[str] = ...,
    ) -> None: ...

class TensorChunk(_message.Message):
    __slots__ = ("chunk_index", "kind", "layer_index", "dtype", "shape", "byte_offset", "total_bytes", "data", "sha256")
    CHUNK_INDEX_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    LAYER_INDEX_FIELD_NUMBER: _ClassVar[int]
    DTYPE_FIELD_NUMBER: _ClassVar[int]
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    BYTE_OFFSET_FIELD_NUMBER: _ClassVar[int]
    TOTAL_BYTES_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    SHA256_FIELD_NUMBER: _ClassVar[int]
    chunk_index: int
    kind: TensorKind
    layer_index: int
    dtype: DType
    shape: _containers.RepeatedScalarFieldContainer[int]
    byte_offset: int
    total_bytes: int
    data: bytes
    sha256: bytes
    def __init__(
        self,
        chunk_index: _Optional[int] = ...,
        kind: _Optional[_Union[TensorKind, str]] = ...,
        layer_index: _Optional[int] = ...,
        dtype: _Optional[_Union[DType, str]] = ...,
        shape: _Optional[_Iterable[int]] = ...,
        byte_offset: _Optional[int] = ...,
        total_bytes: _Optional[int] = ...,
        data: _Optional[bytes] = ...,
        sha256: _Optional[bytes] = ...,
    ) -> None: ...

class ServerTiming(_message.Message):
    __slots__ = ("queue_ns", "compute_ns", "device_to_host_ns", "serialization_ns")
    QUEUE_NS_FIELD_NUMBER: _ClassVar[int]
    COMPUTE_NS_FIELD_NUMBER: _ClassVar[int]
    DEVICE_TO_HOST_NS_FIELD_NUMBER: _ClassVar[int]
    SERIALIZATION_NS_FIELD_NUMBER: _ClassVar[int]
    queue_ns: int
    compute_ns: int
    device_to_host_ns: int
    serialization_ns: int
    def __init__(
        self,
        queue_ns: _Optional[int] = ...,
        compute_ns: _Optional[int] = ...,
        device_to_host_ns: _Optional[int] = ...,
        serialization_ns: _Optional[int] = ...,
    ) -> None: ...

class GenerateTrailer(_message.Message):
    __slots__ = ("total_chunks", "total_tensor_bytes", "response_sha256", "timing", "cache_hits", "cache_misses")
    TOTAL_CHUNKS_FIELD_NUMBER: _ClassVar[int]
    TOTAL_TENSOR_BYTES_FIELD_NUMBER: _ClassVar[int]
    RESPONSE_SHA256_FIELD_NUMBER: _ClassVar[int]
    TIMING_FIELD_NUMBER: _ClassVar[int]
    CACHE_HITS_FIELD_NUMBER: _ClassVar[int]
    CACHE_MISSES_FIELD_NUMBER: _ClassVar[int]
    total_chunks: int
    total_tensor_bytes: int
    response_sha256: bytes
    timing: ServerTiming
    cache_hits: int
    cache_misses: int
    def __init__(
        self,
        total_chunks: _Optional[int] = ...,
        total_tensor_bytes: _Optional[int] = ...,
        response_sha256: _Optional[bytes] = ...,
        timing: _Optional[_Union[ServerTiming, _Mapping]] = ...,
        cache_hits: _Optional[int] = ...,
        cache_misses: _Optional[int] = ...,
    ) -> None: ...

class GenerateResponse(_message.Message):
    __slots__ = ("header", "tensor_chunk", "trailer")
    HEADER_FIELD_NUMBER: _ClassVar[int]
    TENSOR_CHUNK_FIELD_NUMBER: _ClassVar[int]
    TRAILER_FIELD_NUMBER: _ClassVar[int]
    header: GenerateHeader
    tensor_chunk: TensorChunk
    trailer: GenerateTrailer
    def __init__(
        self,
        header: _Optional[_Union[GenerateHeader, _Mapping]] = ...,
        tensor_chunk: _Optional[_Union[TensorChunk, _Mapping]] = ...,
        trailer: _Optional[_Union[GenerateTrailer, _Mapping]] = ...,
    ) -> None: ...
