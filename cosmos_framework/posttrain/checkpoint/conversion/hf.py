# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Hugging Face conversion of consolidated ``model.pt`` for vLLM teachers and offline scorers.

``overlay_consolidated_onto_hf`` memory-maps a consolidated checkpoint (``torch.save({"net": state_dict})``,
Hugging Face key names), overlays it onto the HF base with strict key accounting, saves model and processor with
``save_pretrained`` and rewrites the fused MoE expert tensors in the saved shards back to the CHECKPOINT layout: the
rollout publishes exactly those tensors into vLLM every step without a transpose, so the checkpoint layout is what
vLLM expects on disk, while the HF module only accepts the (1, 2)-transposed layout in memory.

The remaining functions verify such a directory without a model. ``sync_sidecars`` / ``verify_sidecars`` make the
tokenizer, processor and generation sidecars byte-identical to the base: transformers >= 5 re-serializes them and
writes ``chat_template.jinja`` and ``processor_config.json``, which shadow the base's files on load and make the
teacher fail the shared-vocabulary identity check of ``config/build.py``. ``write_unified_chat_template`` is the one
ruled exception (the materializer's ``--unified-chat-template``): ``tokenizer_config.json``'s AND ``chat_template.json``'s
``chat_template`` set to the processor template in its canonical form (``processor-canonical``, the reasoning branch trimmed),
every other key of each file unchanged, verified as that rule instead of byte identity; under that option the preflight
(``unified_template_conflicts``) and ``verify_sidecars`` decide about ``additional_chat_templates/`` through ONE helper,
``unified_template_directory_conflict`` (the directory is refused whatever it holds). ``safetensors_header``, ``shard_status``
and ``verify_index`` check the shards' shape torch-free and ``safetensors_reader_error`` asks the pinned safetensors reader
itself; ``verify_with_torch`` probes config identity, the on-disk expert
layout, one tensor bitwise and the processor identity on the OUTPUT directory.

Every failure raises ``ConversionError`` with a message naming the real path. torch, transformers and safetensors
are imported inside the functions that need them.
"""

from __future__ import annotations

import difflib
import filecmp
import gc
import hashlib
import json
import os
import shutil
import struct
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, BinaryIO

from cosmos_framework.posttrain.utils.log import logger

if TYPE_CHECKING:
    import torch

INDEX_FILE = "model.safetensors.index.json"
# What an OPD teacher directory must carry (the OPD launcher probes these) plus the tokenizer config.
REQUIRED_FILES = (
    "config.json",
    INDEX_FILE,
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
# Every tokenizer / processor / generation sidecar transformers reads or writes. The output must present EXACTLY the
# base's set: files the base has are copied verbatim (over what save_pretrained re-serialized), files the base lacks
# are removed (chat_template.jinja shadows tokenizer_config.json's chat_template on load, processor_config.json shadows
# preprocessor_config.json). config.json, the shards and the index are NOT sidecars.
SIDECAR_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.json",
    "chat_template.jinja",
    "merges.txt",
    "vocab.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "tokenizer.model",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "processor_config.json",
    "generation_config.json",
)
SIDECAR_DIRS = ("additional_chat_templates",)
# Compatibility files that may be overlaid onto an already-materialized export. Unlike sync_sidecars, this includes
# config.json and never removes a file the base lacks; it serves existing exports without changing their weight layout.
COMPATIBILITY_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json")
TRAINER_BACKUP_SUFFIX = ".trainer.bak"
PROCESSOR_TEMPLATE_FILE = "chat_template.json"  # the processor template: what ``AutoProcessor`` renders with
TOKENIZER_CONFIG_FILE = "tokenizer_config.json"  # its ``chat_template`` key: what a tokenizer-only loader renders with
CHAT_TEMPLATE_KEY = "chat_template"
# Recorded in .materialized beside the template shas of a unified export: the serve must set no
# enable_thinking. Under the unified (processor) template a request with enable_thinking=true is prefilled with the opener
# token itself, <|im_start|>assistant\n<think> (id 151667, no newline after it; 251 vs 250 ids on the round's probe; row
# rule R4), while the rows supervise the opener as the student's first token; dsh sends no enable_thinking and the
# student's serve recipe must not set it.
UNIFIED_SERVE_ASSUMPTION = (
    "the serve sets no enable_thinking: under the unified (processor) template a request with enable_thinking=true is "
    "prefilled with the opener token <think> itself (no newline after it) while the rows supervise the opener as the "
    "student's first token (review R151-05)"
)
# Row rule R1 of record (the lab's brain program, 2026-10-02): the unified template is the processor
# template with ONE change in its reasoning branch, the replayed ``reasoning_content`` rendered trimmed, so a replay string
# that arrives with a newline at both ends (dsh's form) renders the canonical block ``<think>\n<text>\n</think>\n`` = the
# student's own emission form (on the captured request the four positions 7248 / 7254 / 8016 / 8025 read 198 / 624 / 198 /
# 624 instead of 271 / 382 / 271 / 382, 8,969 ids either way). Both lines are template SOURCE: the ``\n`` inside the Jinja
# string literals are the two characters backslash + n.
PROCESSOR_REASONING_LINE = "{{- '<think>\\n' ~ message.reasoning_content ~ '\\n</think>\\n' }}"
CANONICAL_REASONING_LINE = "{{- '<think>\\n' ~ (message.reasoning_content | trim) ~ '\\n</think>\\n' }}"
TEMPLATE_MODE_CANONICAL = "processor-canonical"
# The two template files a unified export rewrites: a tokenizer-only loader (vLLM with a tools array) reads
# tokenizer_config.json, a processor loader (vLLM without tools, AutoProcessor) reads chat_template.json; with both carrying
# the canonical text a request with or without tools renders the same reasoning form. In SIDECAR_FILES order.
UNIFIED_TEMPLATE_FILES = (TOKENIZER_CONFIG_FILE, PROCESSOR_TEMPLATE_FILE)
# Template sidecars a loader prefers over the two JSON files, read on transformers 4.57.6 and 5.8.1 (the two pinned
# versions). tokenization_utils_base.py, both versions: "If independent chat template file(s)
# exist, they take priority over template entries in the tokenizer config" -- chat_template.jinja becomes the default and
# additional_chat_templates/<name>.jinja the named templates (default.jinja the default), replacing tokenizer_config.json's
# key. processing_utils.py, both versions: chat_template.json (legacy) wins and a chat_template.jinja beside it is not read;
# additional_chat_templates/ beside chat_template.json raises ValueError at load; on 5.8.1 a chat_template key inside
# processor_config.json wins over the files. Under --unified-chat-template each of these must be absent or equal to the
# canonical text, or the export's effective template is not the one the option promises: a unified export built from a base
# with a divergent chat_template.jinja ships both canonical JSON files while AutoTokenizer renders the Jinja.
PREFERRED_TEMPLATE_SIDECARS = (
    "chat_template.jinja",
    "additional_chat_templates/",
    "processor_config.json:chat_template",
)
# The preferred_by texts of the named-template conflict records: one file under the directory, and the directory itself
# (the directory is a conflict whatever it holds; see unified_template_directory_conflict).
NAMED_TEMPLATE_PREFERRED_BY = (
    "the tokenizer loader, as a named template over tokenizer_config.json's key; beside "
    "chat_template.json the processor loader refuses the combination (4.57.6 and 5.8.1)"
)
TEMPLATE_DIRECTORY_PREFERRED_BY = (
    "the tokenizer loader (named templates over tokenizer_config.json's key) and the processor loader (refuses them "
    "beside chat_template.json), 4.57.6 and 5.8.1"
)
PROCESSOR_CONFIG_FILE = "processor_config.json"
MODEL_IDENTITY_FIELDS = (
    "vocab_size",
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "intermediate_size",
)
REQUIRED_MODEL_IDENTITY_FIELDS = ("vocab_size", "hidden_size", "num_hidden_layers")
_MAX_HEADER_BYTES = 256 * 2**20


class ConversionError(RuntimeError):
    """The checkpoint, the base or the produced directory violates the overlay contract; the message names the path."""


def model_identity_from_config(config: Mapping[str, Any]) -> dict[str, object] | None:
    """Return a structural identity strong enough to guard metadata overlay across model families."""
    model_type = config.get("model_type")
    architectures = config.get("architectures")
    if not isinstance(model_type, str) or not model_type or not isinstance(architectures, list):
        return None
    if not architectures or not all(isinstance(name, str) and name for name in architectures):
        return None
    text = config.get("text_config")
    text_config = text if isinstance(text, dict) else config
    identity: dict[str, object] = {"model_type": model_type, "architectures": list(architectures)}
    for name in MODEL_IDENTITY_FIELDS:
        value = text_config.get(name, config.get(name))
        if value is not None:
            identity[name] = value
    if any(name not in identity for name in REQUIRED_MODEL_IDENTITY_FIELDS):
        return None
    return identity


def model_identity(directory: str) -> dict[str, object] | None:
    """Read structural identity from one HF directory, or None when it is unreadable/incomplete."""
    try:
        with open(os.path.join(directory, "config.json")) as handle:
            config = json.load(handle)
    except (OSError, ValueError):
        return None
    return model_identity_from_config(config) if isinstance(config, dict) else None


def validate_model_identity(export: str, base: str) -> dict[str, object]:
    """Require equal structural identity before base metadata may replace export metadata."""
    export_identity, base_identity = model_identity(export), model_identity(base)
    if export_identity is None or base_identity is None or export_identity != base_identity:
        raise ConversionError(
            f"model identity mismatch; export={export_identity!r}, base={base_identity!r}; nothing copied"
        )
    return export_identity


def overlay_compatibility_files(
    export: str,
    base: str,
    files: tuple[str, ...] = COMPATIBILITY_FILES,
    backup_suffix: str = TRAINER_BACKUP_SUFFIX,
) -> list[str]:
    """Copy differing base compatibility files after validation, preserving each original export once."""
    validate_model_identity(export, base)
    copied: list[str] = []
    for name in files:
        source, destination = os.path.join(base, name), os.path.join(export, name)
        if not os.path.exists(source):
            continue
        if os.path.exists(destination) and filecmp.cmp(source, destination, shallow=False):
            continue
        backup = destination + backup_suffix
        if os.path.exists(destination) and not os.path.exists(backup):
            shutil.copy2(destination, backup)
        shutil.copy2(source, destination)
        copied.append(name)
    return copied


def write_config_from_base(export: str, base: str, backup_suffix: str = TRAINER_BACKUP_SUFFIX) -> dict[str, Any]:
    """Make ``<export>/config.json`` the base's ``config.json`` verbatim, keeping the file transformers wrote once as
    ``config.json<backup_suffix>``.

    Why: ``save_pretrained`` under transformers 5 re-serializes the config in its own format (``num_local_experts`` for
    ``num_experts``, ``rope_parameters`` for ``rope_scaling`` + ``rope_theta``, ``vision_config.model_type`` renamed,
    ``text_config.tie_word_embeddings: true`` added beside the top-level ``false``), which the evaluation service's
    transformers 4.57 reads as a different model. The weights are the base's architecture, so the
    base's config is the one true description. Requires equal structural identity; idempotent (a kept backup is never
    overwritten).
    """
    validate_model_identity(export, base)
    source, destination = os.path.join(base, "config.json"), os.path.join(export, "config.json")
    backup = destination + backup_suffix
    kept = False
    if os.path.isfile(destination) and not filecmp.cmp(source, destination, shallow=False):
        if not os.path.exists(backup):
            shutil.copy2(destination, backup)
            kept = True
    part = destination + ".copying"
    shutil.copy2(source, part)
    os.replace(part, destination)
    return {
        "config_source": "base",
        "transformers_config_kept_as": os.path.basename(backup) if os.path.exists(backup) else None,
        "kept_now": kept,
    }


def _sha256_text(text: str | None) -> str | None:
    return None if text is None else hashlib.sha256(text.encode("utf-8")).hexdigest()


def describe_json_value(value: Any) -> str:
    """What a parsed JSON value is, for a refusal text that says what was found instead of what the rule needs: ``a JSON
    array``, ``an empty JSON object``, ``a JSON object``, ``an empty JSON string``, ``a JSON string``, ``a JSON number``,
    ``a JSON boolean`` or ``absent or null``."""
    if value is None:
        return "absent or null"
    if isinstance(value, bool):
        return "a JSON boolean"
    if isinstance(value, dict):
        return "a JSON object" if value else "an empty JSON object"
    if isinstance(value, list):
        return "a JSON array"
    if isinstance(value, str):
        return "a JSON string" if value else "an empty JSON string"
    return "a JSON number"


def read_processor_template(base: str) -> str:
    """The processor chat template's text: ``<base>/chat_template.json``'s ``chat_template`` string, what the processor
    loader's legacy path indexes as ``chat_template_json["chat_template"]`` (processing_utils.py 4.57.6:1072, 5.8.1:1083; a
    KeyError at load when the key is absent). ``ConversionError`` when the file is missing or unreadable or is not a JSON
    object (``_read_json_file``'s texts), or when the key is not a non-empty string, naming what was found
    (``{}`` used to pass the materializer's preflight and fail the loader, after an earlier fix)."""
    path = os.path.join(base, PROCESSOR_TEMPLATE_FILE)
    payload, _ = _read_json_file(path)
    template = payload.get(CHAT_TEMPLATE_KEY)
    if not isinstance(template, str) or not template:
        raise ConversionError(
            f"{path} carries no non-empty '{CHAT_TEMPLATE_KEY}' string (found {describe_json_value(template)}); the processor "
            f"loader reads chat_template_json['{CHAT_TEMPLATE_KEY}'] from it"
        )
    return template


def canonicalize_processor_template(text: str) -> str:
    """The processor template with ONE change in its reasoning branch (row rule R1 of record): ``reasoning_content``
    rendered ``| trim``, so a replayed string with a newline at both ends renders the canonical block. ``ConversionError``
    unless the exact reasoning line occurs once in ``text``: a template this change does not describe is not unified."""
    if text.count(PROCESSOR_REASONING_LINE) != 1:
        raise ConversionError(
            f"the processor template does not carry the reasoning line {PROCESSOR_REASONING_LINE!r} exactly once "
            f"({text.count(PROCESSOR_REASONING_LINE)} found); --unified-chat-template ({TEMPLATE_MODE_CANONICAL}) cannot "
            f"canonicalize it"
        )
    return text.replace(PROCESSOR_REASONING_LINE, CANONICAL_REASONING_LINE)


def canonical_processor_template(base: str) -> str:
    """``canonicalize_processor_template`` of the base's ``chat_template.json`` text: what a unified export ships."""
    return canonicalize_processor_template(read_processor_template(base))


def canonical_template_diff(processor_template: str) -> list[str]:
    """The one-hunk unified diff (no context lines) from the processor template to its canonical form, kept in the marker."""
    return list(
        difflib.unified_diff(
            processor_template.splitlines(),
            canonicalize_processor_template(processor_template).splitlines(),
            fromfile="chat_template.json (processor template)",
            tofile=f"tokenizer_config.json chat_template ({TEMPLATE_MODE_CANONICAL})",
            n=0,
            lineterm="",
        )
    )


UTF8_BOM = b"\xef\xbb\xbf"


class ByteOrderMarkError(ValueError):
    """The bytes start with a UTF-8 byte-order mark: valid for ``json.loads(bytes)`` (it switches to ``utf-8-sig`` and strips it),
    rejected by every UTF-8 TEXT reader (``json.load`` on an ``open()`` handle, transformers' loaders: ``Unexpected UTF-8 BOM``)."""


def decode_json_bytes(raw: bytes) -> Any:
    """The ONE JSON decoder of the materializer's preflight and of this module's readers: the bytes decoded as strict UTF-8 text,
    then ``json.loads`` (what ``model_identity``, ``_read_json_file`` and transformers' ``open(..., encoding="utf-8")`` readers do
    in the staged build). ``ByteOrderMarkError`` (a ``ValueError``) when the bytes start with the UTF-8 byte-order
    mark, named: the case where the preflight's ``json.loads(bytes)`` stripped the mark and passed a ``config.json`` that
    ``write_config_from_base`` then refused, after an earlier fix; ``UnicodeDecodeError`` / ``JSONDecodeError`` (both ``ValueError``)
    for bytes that are not UTF-8 JSON."""
    if raw.startswith(UTF8_BOM):
        raise ByteOrderMarkError(
            "starts with a UTF-8 byte-order mark (bytes EF BB BF), which the UTF-8 text readers of the loaders and of this module reject"
        )
    return json.loads(raw.decode("utf-8"))


def _read_json_file(path: str) -> tuple[dict[str, Any], bytes]:
    """``(mapping, raw bytes)`` of one JSON object file; ``ConversionError`` when missing, unreadable (a byte-order mark included, by
    name) or not an object. Decodes through ``decode_json_bytes``."""
    try:
        with open(path, "rb") as handle:
            raw = handle.read()
        mapping = decode_json_bytes(raw)
    except (OSError, ValueError) as error:
        raise ConversionError(f"{path} missing or unreadable ({error})") from error
    if not isinstance(mapping, dict):
        raise ConversionError(f"{path} is not a JSON object")
    return mapping, raw


def read_tokenizer_config(directory: str) -> tuple[dict[str, Any], bytes]:
    """``(mapping, raw bytes)`` of ``<directory>/tokenizer_config.json``."""
    return _read_json_file(os.path.join(directory, TOKENIZER_CONFIG_FILE))


def read_processor_template_file(directory: str) -> tuple[dict[str, Any], bytes]:
    """``(mapping, raw bytes)`` of ``<directory>/chat_template.json`` (the processor template's file)."""
    return _read_json_file(os.path.join(directory, PROCESSOR_TEMPLATE_FILE))


def _dumps_like(mapping: dict[str, Any], reference: dict[str, Any], raw: bytes) -> tuple[bytes, bool]:
    """Serialize ``mapping`` in the ``json.dumps`` formatting that reproduces ``raw`` from ``reference`` (indent, key order,
    ASCII escaping and trailing newline tried in turn): ``(bytes, True)`` when one does, so every unchanged key keeps its
    bytes line for line; ``(indent-2 UTF-8 bytes with a trailing newline, False)`` when none does."""
    for indent in (2, 4, None, 1, 3):
        for sort_keys in (False, True):
            for ensure_ascii in (False, True):
                for newline in ("", "\n"):
                    text = (
                        json.dumps(reference, indent=indent, sort_keys=sort_keys, ensure_ascii=ensure_ascii) + newline
                    )
                    if text.encode("utf-8") == raw:
                        out = (
                            json.dumps(mapping, indent=indent, sort_keys=sort_keys, ensure_ascii=ensure_ascii) + newline
                        )
                        return out.encode("utf-8"), True
    return (json.dumps(mapping, indent=2, ensure_ascii=False) + "\n").encode("utf-8"), False


def read_base_template_file(base: str, name: str) -> tuple[dict[str, Any], bytes, str]:
    """``(mapping, raw bytes, the file's own template text)`` of ``<base>/<name>`` for a name in ``UNIFIED_TEMPLATE_FILES``:
    what ``write_unified_chat_template`` reads, exposed so the materializer's preflight can refuse a base it cannot unify
    before anything is published or removed. ``ConversionError`` when the file is missing, unreadable, not
    an object or carries no ``chat_template`` string."""
    mapping, raw = _read_json_file(os.path.join(base, name))
    template = mapping.get(CHAT_TEMPLATE_KEY)
    if not isinstance(template, str):
        raise ConversionError(f"{os.path.join(base, name)} carries no '{CHAT_TEMPLATE_KEY}' string to unify")
    return mapping, raw, template


def _write_unified_file(out: str, base: str, name: str, unified_text: str) -> tuple[str, bool]:
    """Write ``<out>/<name>`` as the base's ``<name>`` mapping with ``chat_template`` set to ``unified_text``, every other key
    unchanged and the base file's ``json.dumps`` formatting reproduced when possible. Returns ``(the base file's own
    template text, formatting reproduced)``."""
    base_mapping, raw, base_template = read_base_template_file(base, name)
    unified = dict(base_mapping)
    unified[CHAT_TEMPLATE_KEY] = unified_text
    data, reproduced = _dumps_like(unified, base_mapping, raw)
    destination = os.path.join(out, name)
    part = destination + ".writing"
    with open(part, "wb") as handle:
        handle.write(data)
    os.replace(part, destination)
    return base_template, reproduced


def write_unified_chat_template(out: str, base: str) -> dict[str, Any]:
    """The materializer's ``--unified-chat-template``: write BOTH template files of the export,
    ``<out>/tokenizer_config.json`` and ``<out>/chat_template.json``, as the base's mappings with ``chat_template`` set to
    the base's PROCESSOR template in its CANONICAL form (``canonicalize_processor_template``: the one reasoning-branch
    change of row rule R1, ``template_mode`` ``processor-canonical``), every other key of each file unchanged and, when
    the base's file is a plain ``json.dumps`` rendering, every other line byte-equal (before it only the
    tokenizer file was rewritten and a request without tools still rendered the base's padded form).

    Why: a tokenizer-only loader (vLLM's chat path for a request that carries a tools array) renders the tokenizer
    template, which renders no reasoning in the history; a processor loader (vLLM without tools, ``AutoProcessor``)
    renders ``chat_template.json``, which pads the replayed reasoning with its own newlines. With the canonical text in
    both files a request with or without tools renders the history reasoning in the student's own emission form, and
    no serve-side template flag is needed. Idempotent; returns the record ``.materialized`` keeps.
    """
    processor_template = read_processor_template(base)
    unified_text = canonicalize_processor_template(processor_template)
    base_tokenizer_template, tokenizer_reproduced = _write_unified_file(out, base, TOKENIZER_CONFIG_FILE, unified_text)
    _, processor_reproduced = _write_unified_file(out, base, PROCESSOR_TEMPLATE_FILE, unified_text)
    return {
        "unified": True,
        "template_mode": TEMPLATE_MODE_CANONICAL,
        "files": list(UNIFIED_TEMPLATE_FILES),
        "processor_template_sha256": _sha256_text(processor_template),
        "base_tokenizer_template_sha256": _sha256_text(base_tokenizer_template),
        "tokenizer_template_sha256": _sha256_text(unified_text),  # the shipped text's sha (tokenizer_config.json)
        "shipped": {name: _sha256_text(unified_text) for name in UNIFIED_TEMPLATE_FILES},
        "canonical_diff": canonical_template_diff(processor_template),
        "processor_template_chars": len(processor_template),
        "shipped_template_chars": len(unified_text),
        "base_tokenizer_template_chars": len(base_tokenizer_template),
        "templates_already_equal": unified_text == base_tokenizer_template,
        "formatting_reproduced": {
            TOKENIZER_CONFIG_FILE: tokenizer_reproduced,
            PROCESSOR_TEMPLATE_FILE: processor_reproduced,
        },
    }


def chat_template_record(out: str, base: str, *, unified_chat_template: bool) -> dict[str, Any]:
    """What ``.materialized`` records about the chat templates, torch-free: the option, the base's processor template
    sha256 (None when the base ships no ``chat_template.json`` and the option is off), the base's tokenizer template
    sha256 and the export's tokenizer template sha256 (the shipped canonical text's under the option), and under the
    option the ``template_mode``, the two ``files`` with the ``shipped`` sha256 of each, the one-hunk ``canonical_diff``
    (row rule R1) and the serve assumption ``UNIFIED_SERVE_ASSUMPTION``."""
    try:
        processor_template: str | None = read_processor_template(base)
    except ConversionError:
        if unified_chat_template:
            raise
        processor_template = None
    base_config, _ = read_tokenizer_config(base)
    out_config, _ = read_tokenizer_config(out)
    base_template, out_template = base_config.get(CHAT_TEMPLATE_KEY), out_config.get(CHAT_TEMPLATE_KEY)
    record = {
        "unified": unified_chat_template,
        "processor_template_sha256": _sha256_text(processor_template),
        "base_tokenizer_template_sha256": _sha256_text(base_template if isinstance(base_template, str) else None),
        "tokenizer_template_sha256": _sha256_text(out_template if isinstance(out_template, str) else None),
    }
    if unified_chat_template:
        assert processor_template is not None
        out_processor_file, _ = read_processor_template_file(out)
        shipped_processor = out_processor_file.get(CHAT_TEMPLATE_KEY)
        record["template_mode"] = TEMPLATE_MODE_CANONICAL
        record["files"] = list(UNIFIED_TEMPLATE_FILES)
        record["shipped"] = {
            TOKENIZER_CONFIG_FILE: _sha256_text(out_template if isinstance(out_template, str) else None),
            PROCESSOR_TEMPLATE_FILE: _sha256_text(shipped_processor if isinstance(shipped_processor, str) else None),
        }
        record["canonical_diff"] = canonical_template_diff(processor_template)
        record["serve_assumption"] = UNIFIED_SERVE_ASSUMPTION
        # The loader-preferred sidecars the base carries with the canonical text (none for the base of record); a divergent
        # one was refused before the build.
        record["preferred_sidecars"] = unified_template_conflicts(base)[1]
    return record


def _verify_unified_file(out: str, base: str, name: str) -> None:
    """The rule for one template file of a unified export, asserted: ``<out>/<name>`` parses to the base's ``<name>``
    mapping with exactly one key allowed to differ, ``chat_template``, and that value equals the base's processor
    template (``chat_template.json``) in its canonical form (``canonicalize_processor_template``). ``ConversionError``
    names the first violation (keys, another key's value, the template)."""
    base_mapping, _ = _read_json_file(os.path.join(base, name))
    out_mapping, _ = _read_json_file(os.path.join(out, name))
    destination = os.path.join(out, name)
    if set(out_mapping) != set(base_mapping):
        missing, extra = sorted(set(base_mapping) - set(out_mapping)), sorted(set(out_mapping) - set(base_mapping))
        raise ConversionError(f"verify: {destination} keys differ from base {base}: missing={missing} extra={extra}")
    other = sorted(k for k in base_mapping if k != CHAT_TEMPLATE_KEY and base_mapping[k] != out_mapping[k])
    if other:
        raise ConversionError(
            f"verify: {destination} differs from base {base} on {other}; --unified-chat-template may change "
            f"'{CHAT_TEMPLATE_KEY}' only"
        )
    if out_mapping.get(CHAT_TEMPLATE_KEY) != canonical_processor_template(base):
        raise ConversionError(
            f"verify: {destination} {CHAT_TEMPLATE_KEY} is not the base's processor template "
            f"({os.path.join(base, PROCESSOR_TEMPLATE_FILE)}) in its canonical form ({TEMPLATE_MODE_CANONICAL}: the "
            f"reasoning branch trimmed) as --unified-chat-template requires"
        )


def unified_template_directory_conflict(root: str, name: str = SIDECAR_DIRS[0]) -> dict[str, Any] | None:
    """The ONE directory rule of unified mode, shared by the preflight (``unified_template_conflicts``) and the final
    verifier (``verify_sidecars``) so the two cannot drift again (previously the preflight recorded only direct regular
    files, so an empty or nested-only ``additional_chat_templates/`` passed it and failed the verifier after the
    overlay): ``<root>/<name>`` is a conflict whenever it IS A DIRECTORY, whatever it holds.
    Returns ``{file: "additional_chat_templates/", sha256: None, preferred_by, equals_canonical: False, entries}`` with
    its direct entries as ``{name, kind, sha256}`` (``kind`` ``file`` with the sha256 of the bytes, ``directory``, or
    ``other``), or ``None`` when no such directory exists.

    Loader facts, read on the two pinned versions (``docker/posttrain/environment/pyproject.toml:64-65``): both loaders
    discover named templates of a local path with ``Path(root, "additional_chat_templates").is_dir()`` followed by
    ``glob("*.jinja")`` (tokenization_utils_base.py 4.57.6:2030-2037 and 5.8.1:1665-1670; processing_utils.py
    4.57.6:953-958 and 5.8.1:968-973), so an EMPTY directory, one holding sub-directories only, and files without the
    ``.jinja`` suffix yield no template: the tokenizer keeps ``tokenizer_config.json``'s key (4.57.6:2183-2200,
    5.8.1:1790-1807) and the processor reads ``chat_template.json`` without the ValueError it raises when a ``*.jinja``
    stands beside it (4.57.6:1068-1079, 5.8.1:1079-1090). Both loaders therefore ignore such a directory; the rule still
    refuses it, fail closed: the discovery is a glob a newer transformers may widen, the base of record carries no such
    directory, and one rule in both checks is what keeps the preflight equal to the verifier. No case was found that the
    final verifier must allow. A sub-directory whose NAME ends in ``.jinja`` is NOT ignored: ``Path.glob("*.jinja")``
    returns it, the tokenizer's ``open()`` on it raises IsADirectoryError (4.57.6:2195, 5.8.1:1802) and the processor refuses
    the combination beside ``chat_template.json`` (4.57.6:1073-1079, 5.8.1:1084-1090); the record lists it as ``kind``
    ``directory`` and both checks refuse it before a loader sees it. The default path reads the same
    record to refuse a directory with entries that are not regular files (the materializer's ``preflight``)."""
    directory = os.path.join(root, name)
    if not os.path.isdir(directory):
        return None
    entries: list[dict[str, Any]] = []
    for entry in sorted(os.listdir(directory)):
        path = os.path.join(directory, entry)
        if os.path.isfile(path):
            with open(path, "rb") as handle:
                entries.append({"name": entry, "kind": "file", "sha256": hashlib.sha256(handle.read()).hexdigest()})
        else:
            entries.append({"name": entry, "kind": "directory" if os.path.isdir(path) else "other", "sha256": None})
    return {
        "file": f"{name}/",
        "sha256": None,
        "preferred_by": TEMPLATE_DIRECTORY_PREFERRED_BY,
        "equals_canonical": False,
        "entries": entries,
    }


def describe_template_conflict(record: Mapping[str, Any]) -> str:
    """One conflict record as BOTH refusal messages name it (the preflight's and the verifier's, one text): a file with
    its sha256 prefix; the directory with its direct entries (``<name>/`` for a sub-directory, ``none`` when empty)."""
    entries = record.get("entries")
    if entries is None:
        return f"{record['file']} (sha256 {str(record['sha256'])[:12]}, preferred by {record['preferred_by']})"
    listed = ", ".join(f"{e['name']}/" if e["kind"] == "directory" else e["name"] for e in entries) or "none"
    return (
        f"{record['file']} (a directory, refused whatever it holds; direct entries: {listed}; preferred by "
        f"{record['preferred_by']})"
    )


def unified_template_conflicts(base: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """``(conflicts, allowed)`` among the template sidecars a loader prefers over the two JSON files
    (``PREFERRED_TEMPLATE_SIDECARS``), each as ``{file, sha256, preferred_by, equals_canonical}``: ``chat_template.jinja``
    and a ``chat_template`` key in ``processor_config.json`` are allowed when their text equals the canonical processor
    template; the ``additional_chat_templates/`` DIRECTORY is a conflict whatever it holds (the record of
    ``unified_template_directory_conflict``, the helper the final verifier decides by too), followed by one
    record per direct regular file in it with its sha256 (the tokenizer loader prefers named templates over the config
    key; beside ``chat_template.json`` the processor loader refuses the combination outright). Torch-free; reads the BASE
    only, so the materializer can refuse before anything is published or removed."""
    canonical = canonical_processor_template(base)
    canonical_sha256 = _sha256_text(canonical)
    conflicts: list[dict[str, Any]] = []
    allowed: list[dict[str, Any]] = []
    jinja = os.path.join(base, SIDECAR_FILES[3])  # chat_template.jinja
    if os.path.isfile(jinja):
        with open(jinja, encoding="utf-8") as handle:
            text = handle.read()
        record = {
            "file": "chat_template.jinja",
            "sha256": _sha256_text(text),
            "preferred_by": "the tokenizer loader, over tokenizer_config.json's chat_template (4.57.6 and 5.8.1)",
            "equals_canonical": text == canonical,
        }
        (allowed if text == canonical else conflicts).append(record)
    # additional_chat_templates/: the directory itself is a conflict, whatever it holds (one rule with the verifier)
    directory = unified_template_directory_conflict(base)
    if directory is not None:
        conflicts.append(directory)
        for entry in directory["entries"]:
            if entry["kind"] == "file":
                conflicts.append(
                    {
                        "file": f"{SIDECAR_DIRS[0]}/{entry['name']}",
                        "sha256": entry["sha256"],
                        "preferred_by": NAMED_TEMPLATE_PREFERRED_BY,
                        "equals_canonical": entry["sha256"] == canonical_sha256,
                    }
                )
    processor_config = os.path.join(base, PROCESSOR_CONFIG_FILE)
    if os.path.isfile(processor_config):
        mapping, _ = _read_json_file(processor_config)
        key = mapping.get(CHAT_TEMPLATE_KEY)
        if key is not None:
            record = {
                "file": f"{PROCESSOR_CONFIG_FILE}:{CHAT_TEMPLATE_KEY}",
                "sha256": _sha256_text(key) if isinstance(key, str) else None,
                "preferred_by": "the processor loader, over the template files (5.8.1)",
                "equals_canonical": key == canonical,
            }
            (allowed if key == canonical else conflicts).append(record)
    return conflicts, allowed


def refuse_unified_template_conflicts(base: str) -> list[dict[str, Any]]:
    """Raise ``ConversionError`` naming every conflicting sidecar of ``base`` (``unified_template_conflicts``, each as
    ``describe_template_conflict`` names it); returns the allowed ones. Called by the materializer's preflight before
    anything is published or removed, with or without ``--no-verify``."""
    conflicts, allowed = unified_template_conflicts(base)
    if conflicts:
        named = "; ".join(describe_template_conflict(c) for c in conflicts)
        raise ConversionError(
            f"REFUSED before anything is published: --unified-chat-template needs a base that ships its templates in "
            f"{PROCESSOR_TEMPLATE_FILE} and {TOKENIZER_CONFIG_FILE} only; {base} carries template sidecar(s) a loader prefers "
            f"over the two JSON files (a file is allowed only when its text is the canonical processor template; the "
            f"{SIDECAR_DIRS[0]}/ directory never): {named}"
        )
    return allowed


def preflight_unified_base(base: str) -> list[dict[str, Any]]:
    """Every torch-free refusal of unified mode that reads the BASE alone, in one call for the materializer's preflight:
    both template files parse and carry a template string (``read_base_template_file``, what
    ``write_unified_chat_template`` needs after the overlay), the processor template canonicalizes and no loader-preferred
    sidecar conflicts (``refuse_unified_template_conflicts``, which calls ``canonical_processor_template``). Returns the
    allowed sidecars."""
    for name in UNIFIED_TEMPLATE_FILES:
        read_base_template_file(base, name)
    return refuse_unified_template_conflicts(base)


def loader_template_files(root: str) -> list[str]:
    """The Jinja template files the tokenizer and processor loaders read as UTF-8 TEXT from ``root``:
    ``chat_template.jinja`` and every regular ``additional_chat_templates/*.jinja`` named template, sorted.
    Read on the two pinned transformers (``pyproject.toml``): tokenization_utils_base.py opens ``chat_template.jinja`` with
    ``encoding="utf-8"`` (4.57.6:2188, 5.8.1:1795) and each named template it discovered as
    ``additional_chat_templates/<name>.jinja`` (discovery 4.57.6:2047, 5.8.1:1680; the open 4.57.6:2195 with ``encoding="utf8"``,
    5.8.1:1802 with the locale's codec, UTF-8 in the container); processing_utils.py opens ``chat_template.jinja`` (4.57.6:1070,
    5.8.1:1081) and the named templates (4.57.6:1082, 5.8.1:1093) the same way, plus the legacy ``chat_template.json``
    (4.57.6:1086, 5.8.1:1097), which the preflight parses as JSON; the container's 5.13.0 reads the same files
    (tokenization_utils_base.py 1672-1684 and 1799 / 1806, processing_utils.py 1261-1274 and 1371 / 1383 / 1387; the names
    ``CHAT_TEMPLATE_FILE`` / ``CHAT_TEMPLATE_DIR`` in utils/hub.py:66-67). A file under the directory without the ``.jinja`` suffix is
    not discovered by either loader and is not listed; an entry that is not a regular file is the directory rule's business
    (``unified_template_directory_conflict``)."""
    files: list[str] = []
    jinja = os.path.join(root, "chat_template.jinja")
    if os.path.isfile(jinja):
        files.append(jinja)
    directory = os.path.join(root, SIDECAR_DIRS[0])
    if os.path.isdir(directory):
        for name in sorted(os.listdir(directory)):
            path = os.path.join(directory, name)
            if name.endswith(".jinja") and os.path.isfile(path):
                files.append(path)
    return files


def utf8_text_error(path: str) -> str | None:
    """None when the bytes of ``path`` decode as strict UTF-8 (what the loaders' ``open(path, encoding="utf-8")`` yields; a
    byte-order mark decodes, as U+FEFF text, and is not an error for them), else the codec's own reason with the byte and its
    position, e.g. ``'utf-8' codec can't decode byte 0xff in position 25: invalid start byte``."""
    with open(path, "rb") as handle:
        raw = handle.read()
    try:
        raw.decode("utf-8")
    except UnicodeDecodeError as error:
        return str(error)
    return None


def verify_unified_tokenizer_config(out: str, base: str) -> None:
    """The rule (``_verify_unified_file``) for ``tokenizer_config.json``, the file a tokenizer-only loader reads."""
    _verify_unified_file(out, base, TOKENIZER_CONFIG_FILE)


def verify_unified_template_files(out: str, base: str) -> list[str]:
    """The rule for BOTH template files of a unified export, in ``SIDECAR_FILES`` order; returns the
    names checked."""
    for name in UNIFIED_TEMPLATE_FILES:
        _verify_unified_file(out, base, name)
    return list(UNIFIED_TEMPLATE_FILES)


def service_processor_config(base: str) -> dict[str, Any]:
    """A ``processor_config.json`` composed from the base's own sidecars: the image block is the base's
    ``preprocessor_config.json`` (its ``image_processor_type`` included, verbatim), the video block the base's
    ``video_preprocessor_config.json``; ``processor_class`` is the base's. Per-call media kwargs a run sets
    (``rl.processor_kwargs``) are call-time overrides recorded in its ``config_resolved.yaml``, not processor defaults,
    and are deliberately not baked in.

    Scope of record (this helper's output was measured against the file served beside the A1a export): a unit-tested
    helper with NO production caller. The materializer CLI copies the base's sidecars verbatim and removes a
    ``processor_config.json`` the base lacks (``sync_sidecars``; ``verify_sidecars`` refuses any sidecar the base
    lacks), so an export of the Cosmos 3.5 Super base carries none; the file served beside the A1a export was placed
    by hand (a byte-copy of another export's) and differs from this helper's output in the image processor class
    (``Qwen2VLImageProcessor`` served vs ``Qwen2VLImageProcessorFast`` from the base's sidecar) and in 17
    explicit-default fields it spells out (``do_*``,
    ``resample``, ``rescale_factor``, ``fps``, ``min_frames``, ``max_frames``, ``do_sample_frames``, ``return_metadata``).
    Connecting this helper to the CLI is therefore a separate change: a ``CONVERSION_VERSION`` bump, ``verify_sidecars``
    accepting the composed file, and a re-anchored serve.
    """
    with open(os.path.join(base, "preprocessor_config.json")) as handle:
        image = json.load(handle)
    video_path = os.path.join(base, "video_preprocessor_config.json")
    video = None
    if os.path.isfile(video_path):
        with open(video_path) as handle:
            video = json.load(handle)
    processor_class = image.pop("processor_class", None) or (video or {}).get("processor_class") or "Qwen3VLProcessor"
    if video is not None:
        video.pop("processor_class", None)
    result: dict[str, Any] = {"image_processor": image, "processor_class": processor_class}
    if video is not None:
        result["video_processor"] = video
    return result


def _log(msg: str) -> None:
    logger.info(f"[hf_overlay] {msg}")


def _read_weight_map(index_path: str) -> dict[str, str]:
    with open(index_path) as handle:
        return json.load(handle)["weight_map"]


# ---------------------------------------------------------------- overlay -------------------------------------------
def maybe_transpose_expert(
    tensor: torch.Tensor, want: tuple[int, ...]
) -> torch.Tensor | None:  # tensor: [*source_shape], returns: [*want] or None
    """Return ``tensor`` when its shape is ``want``, its (1, 2)-transpose when the two are exact transposes, else None.

    The consolidated checkpoint stores fused expert weights as ``[experts, in, out]`` while the HF module holds
    ``[experts, out, in]``; only that exact relation is bridged, any other mismatch is left for strict accounting.
    """
    if tuple(tensor.shape) == tuple(want):
        return tensor  # [*want]
    if tensor.dim() == 3 and len(want) == 3 and (tensor.shape[0], tensor.shape[2], tensor.shape[1]) == tuple(want):
        return tensor.transpose(1, 2).contiguous()  # [experts,out,in]
    return None


def open_file_reference(payload: BinaryIO) -> str | None:
    """The path through which the file ``payload`` has open can be opened AGAIN as the same inode: ``/proc/self/fd/<fd>`` on Linux
    (the kernel resolves it to the open file description, whatever the original path names now), else None.
    ``torch.load(mmap=True)`` takes only a path (torch 2.11: "f must be a file path in order to use
    the mmap argument"), so this is how the loader memory-maps exactly the file object the materializer hashed."""
    reference = f"/proc/self/fd/{payload.fileno()}"
    return reference if os.path.exists(reference) else None


def load_consolidated_state_dict(
    model_pt: str, payload: BinaryIO | None = None
) -> dict[str, torch.Tensor]:  # values: [*shape]
    """Memory-map one consolidated ``model.pt`` and return its ``net`` state dict (``net_ema`` is tolerated, not used).

    With ``payload`` the tensors come from THAT open file object, never from a fresh
    resolution of ``model_pt``: the file is memory-mapped through ``open_file_reference`` (``/proc/self/fd/<fd>``, the same inode
    even after the path was replaced by the merge tool's tmp + os.replace), or, where that reference does not exist, read whole
    through the file object (no mmap; the payload is held in memory). ``model_pt`` then only names the file in messages.
    """
    import torch

    if payload is not None:
        reference = open_file_reference(payload)
        if reference is None:
            _log(
                f"{model_pt}: no /proc/self/fd reference for the open payload; reading it whole through the file object (no mmap)"
            )
            payload.seek(0)
            obj = torch.load(payload, map_location="cpu", weights_only=False, mmap=False)  # net values: [*shape]
        else:
            obj = torch.load(reference, map_location="cpu", weights_only=False, mmap=True)  # net values: [*shape]
    else:
        if not os.path.isfile(model_pt):
            raise ConversionError(
                f"{model_pt} does not exist; expected a consolidated checkpoint directory holding model.pt"
            )
        obj = torch.load(model_pt, map_location="cpu", weights_only=False, mmap=True)  # net values: [*shape]
    if not isinstance(obj, dict) or "net" not in obj:
        found = sorted(obj) if isinstance(obj, dict) else type(obj).__name__
        raise ConversionError(f"{model_pt}: top level is {found}; expected a dict containing 'net'")
    extra_top = sorted(set(obj) - {"net", "net_ema"})
    if extra_top:
        raise ConversionError(
            f"{model_pt}: unrecognized top-level keys {extra_top}; expected 'net' (plus optional 'net_ema')"
        )
    if "net_ema" in obj:
        _log(f"WARNING: {model_pt} also contains 'net_ema'; using the raw 'net' weights, not EMA")
    state_dict = obj["net"]  # values: [*shape]
    if not isinstance(state_dict, dict) or not state_dict:
        raise ConversionError(f"{model_pt}['net'] is not a non-empty state dict")
    return state_dict  # values: [*shape]


def rewrite_expert_shards(staging_dir: str, tensors: Mapping[str, torch.Tensor]) -> int:  # tensors: [*shape] per key
    """Replace the named tensors inside the saved safetensors shards and return how many were rewritten.

    Shards are located through the index when present, else every ``*.safetensors`` file is scanned. Each shard is
    rewritten through a same-directory temporary and renamed, so a crash leaves the previous complete shard. A name
    that no shard holds raises ``ConversionError``: a silently missing rewrite would ship the HF layout to vLLM.
    """
    from safetensors.torch import load_file, save_file

    index_path = os.path.join(staging_dir, INDEX_FILE)
    if os.path.isfile(index_path):
        weight_map = _read_weight_map(index_path)
        shards = sorted({weight_map[name] for name in tensors if name in weight_map})
    else:
        shards = sorted(name for name in os.listdir(staging_dir) if name.endswith(".safetensors"))
    rewritten = 0
    for shard in shards:
        path = os.path.join(staging_dir, shard)
        loaded = load_file(path)  # values: [*shape]
        hit = False
        for name in list(loaded):
            if name in tensors:
                loaded[name] = tensors[name].to(loaded[name].dtype).contiguous()  # [*checkpoint_shape]
                rewritten += 1
                hit = True
        if hit:
            temporary = path + ".tmp"
            save_file(loaded, temporary, metadata={"format": "pt"})
            os.replace(temporary, path)
    if rewritten != len(tensors):
        raise ConversionError(
            f"expert shard rewrite incomplete under {staging_dir}: {rewritten} of {len(tensors)} tensors found in shards"
        )
    return rewritten


def overlay_consolidated_onto_hf(
    ckpt_dir: str, model_id: str, staging_dir: str, payload: BinaryIO | None = None
) -> str:
    """Overlay ``<ckpt_dir>/model.pt`` onto the HF base ``model_id`` and write an HF directory into ``staging_dir``.

    Strict key accounting: after the overlay only ``lm_head.weight`` may be missing or unexpected, and only when the
    base ties word embeddings. Fused expert tensors are transposed for the in-memory module and written back to the
    saved shards in the checkpoint layout. Returns ``staging_dir``. Peak memory is the bf16 base plus the transposed
    expert copies; the mmap'd checkpoint pages are reclaimable. With ``payload`` (an open ``model.pt``)
    the state dict is loaded from that file object, so the caller's identity of the bytes is the identity of what is
    overlaid, whatever ``<ckpt_dir>/model.pt`` names by the time the load runs.
    """
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model_pt = os.path.join(ckpt_dir, "model.pt")
    _log(
        f"loading checkpoint state dict (mmap{' through the open file object' if payload is not None else ''}): {model_pt}"
    )
    state_dict = load_consolidated_state_dict(model_pt, payload)  # values: [*shape]
    _log(f"checkpoint state dict has {len(state_dict)} keys")

    _log(f"loading HF base model {model_id}")
    try:
        model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=torch.bfloat16, low_cpu_mem_usage=True)
    except TypeError:  # older transformers spell the dtype argument torch_dtype
        model = AutoModelForImageTextToText.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
        )
    params = dict(model.named_parameters())  # values: [*shape]
    # A trained checkpoint should move these; bitwise equality to the base afterwards is only a warning.
    probe_keys = [k for k in ("lm_head.weight", "model.language_model.norm.weight") if k in state_dict and k in params]
    base_probe = {k: params[k].detach().clone() for k in probe_keys}  # values: [*shape]

    expert_orig: dict[str, torch.Tensor] = {}  # values: [experts,in,out]
    for key in list(state_dict):
        if ".mlp.experts." in key and key in params:
            fixed = maybe_transpose_expert(state_dict[key], tuple(params[key].shape))  # [experts,out,in] or None
            if fixed is None:
                raise ConversionError(
                    f"{model_pt}: expert tensor {key} shape {tuple(state_dict[key].shape)} is neither equal to nor a "
                    f"(1,2)-transpose of the model's {tuple(params[key].shape)}"
                )
            if fixed is not state_dict[key]:
                expert_orig[key] = state_dict[key]  # [experts,in,out]; the layout the saved shards get back
                state_dict[key] = fixed  # [experts,out,in]
    if expert_orig:
        _log(
            f"transposed {len(expert_orig)} fused MoE expert tensors for the in-memory overlay (shards keep the checkpoint layout)"
        )

    result = model.load_state_dict(state_dict, strict=False, assign=True)
    tied = bool(getattr(model.config, "tie_word_embeddings", False))
    allowed = {"lm_head.weight"} if tied else set()
    missing = sorted(set(result.missing_keys) - allowed)
    unexpected = sorted(set(result.unexpected_keys) - allowed)
    _log(
        f"state dict overlay: missing={len(result.missing_keys)} unexpected={len(result.unexpected_keys)} "
        f"(tie_word_embeddings={tied})"
    )
    if missing or unexpected:
        raise ConversionError(
            f"{model_pt}: state dict overlay onto {model_id} differs beyond embedding tying; missing={missing[:20]} "
            f"unexpected={unexpected[:20]}"
        )
    params = dict(model.named_parameters())  # values: [*shape]
    changed = sum(
        1 for k in probe_keys if not torch.equal(base_probe[k].to(params[k].dtype), params[k].detach())
    )  # [*shape] probes
    if probe_keys and changed == 0:
        _log(
            "WARNING: probe tensors are bitwise identical to the base model; the checkpoint may equal the base weights"
        )
    del base_probe

    _log(f"writing HF-format directory: {staging_dir}")
    os.makedirs(staging_dir, exist_ok=True)
    model.save_pretrained(staging_dir, safe_serialization=True)
    AutoProcessor.from_pretrained(model_id).save_pretrained(staging_dir)
    if expert_orig:
        rewritten = rewrite_expert_shards(staging_dir, expert_orig)
        _log(f"rewrote {rewritten} fused MoE expert tensors in the saved shards to the checkpoint layout")
    del model, state_dict, expert_orig
    gc.collect()
    return staging_dir


# ---------------------------------------------------------------- shards: torch-free safetensors checks -------------
def _is_count(value: Any) -> bool:
    """A JSON non-negative integer (``True`` / ``False`` are ``int`` in Python and are not counts)."""
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def safetensors_header_shape_error(header: Any) -> str | None:
    """None when ``header`` has the SHAPE of a safetensors header, else the reason (any JSON used to be accepted,
    and a tensor entry without ``data_offsets`` escaped as KeyError, an entry that is ``5`` as TypeError, a list header as TypeError
    and ``data_offsets`` = ``["x", "y"]`` as an unprefixed ValueError, out of ``shard_status`` and so out of the materializer's
    preflight and the output verifier). The shape, and only the shape: a JSON object whose entries are objects carrying a string
    ``dtype``, a ``shape`` list of non-negative integers and ``data_offsets`` = two non-negative integers with begin <= end, plus
    an optional ``__metadata__`` object (the safetensors format's string map). Names the first tensor that breaks it. It claims
    no more than the shape: the dtype names, string metadata values, contiguous offsets and the byte count of
    each tensor are the pinned reader's verdict, ``safetensors_reader_error``, which the materializer's preflight asks for every
    base shard."""
    if not isinstance(header, dict):
        return f"the header is {describe_json_value(header)}, not a JSON object"
    for name, entry in header.items():
        if name == "__metadata__":
            if not isinstance(entry, dict):
                return f"__metadata__ is {describe_json_value(entry)}, not a JSON object"
            continue
        if not isinstance(entry, dict):
            return f"tensor {name!r} is {describe_json_value(entry)}, not a JSON object"
        if not isinstance(entry.get("dtype"), str) or not entry.get("dtype"):
            return f"tensor {name!r} carries no dtype string"
        shape = entry.get("shape")
        if not isinstance(shape, list) or not all(_is_count(dim) for dim in shape):
            return (
                f"tensor {name!r} carries no shape list of non-negative integers (found {describe_json_value(shape)})"
            )
        offsets = entry.get("data_offsets")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(_is_count(o) for o in offsets)
            or offsets[0] > offsets[1]
        ):
            return (
                f"tensor {name!r} carries no data_offsets of two non-negative integers begin <= end (found {offsets!r})"
            )
    return None


def safetensors_header(path: str) -> tuple[dict[str, Any], int]:
    """Parse a safetensors header without torch: ``({tensor: {dtype, shape, data_offsets}}, header_len)``. ``ValueError`` names
    what is wrong: a file shorter than the 8-byte length, an implausible length, header bytes that are not strict UTF-8 JSON (a
    byte-order mark included, ``decode_json_bytes``) or a header without the safetensors shape (``safetensors_header_shape_error``)."""
    with open(path, "rb") as handle:
        raw = handle.read(8)
        if len(raw) < 8:
            raise ValueError("shorter than the 8-byte header length")
        n = struct.unpack("<Q", raw)[0]
        if n > _MAX_HEADER_BYTES:
            raise ValueError(f"header length {n} is implausible")
        header = decode_json_bytes(handle.read(n))
    reason = safetensors_header_shape_error(header)
    if reason is not None:
        raise ValueError(reason)
    header.pop("__metadata__", None)
    return header, n


def shard_status(path: str, expect_keys: set[str] | None = None) -> str | None:
    """Return None when ``path`` is a safetensors file whose header has the shape, whose size is 8 + header + data and which
    holds exactly ``expect_keys``, else the reason. Cheap checks, not a validation of the format's semantics: that is
    ``safetensors_reader_error`` (the pinned reader), which the materializer's preflight runs after this on every base shard."""
    if not os.path.isfile(path):
        return f"{path} does not exist"
    try:
        header, n = safetensors_header(path)
    except (OSError, ValueError) as error:
        return f"{path}: unreadable safetensors header ({error})"
    if not header:
        return f"{path}: header lists no tensors"
    end = max(int(value["data_offsets"][1]) for value in header.values())
    size = os.path.getsize(path)
    if size != 8 + n + end:
        return f"{path}: size {size} != 8 + header {n} + data {end} = {8 + n + end} (truncated or partially written)"
    if expect_keys is not None and set(header) != expect_keys:
        missing, extra = sorted(expect_keys - set(header)), sorted(set(header) - expect_keys)
        return f"{path}: tensors differ from the index; missing={missing[:5]} extra={extra[:5]}"
    return None


def safetensors_reader_error(path: str) -> str | None:
    """None when the pinned safetensors reader (``safetensors.safe_open(path, framework="np")``, the library every
    ``from_pretrained`` reads a shard with; ``safetensors==0.8.0`` in ``pyproject.toml``) opens ``path`` and lists its tensors
    and metadata, else the reader's own refusal as ``<ExceptionClass>: <message>``. The Rust reader validates what
    ``safetensors_header_shape_error`` does not claim to: the dtype
    names, string metadata values, offsets contiguous from 0 in order (no gap, no overlap), every byte range equal to the dtype
    size x the element count, and the data end equal to the file length. Header-only (the file is mapped, no tensor is read, so
    a BF16 shard needs no numpy bfloat16); torch-free. A reader that cannot be imported is a reason too (fail closed: the overlay
    cannot run without it either)."""
    try:
        from safetensors import safe_open
    except ImportError as error:  # a bare interpreter without the project's pinned dependency
        return f"the pinned safetensors reader cannot be imported ({type(error).__name__}: {error})"
    try:
        with safe_open(path, framework="np") as handle:
            list(handle.keys())
            handle.metadata()
    except Exception as error:  # noqa: BLE001 -- the reader's own class and text are the verdict, whatever they are
        return f"{type(error).__name__}: {error}"
    return None


def hf_export_status(directory: str) -> str | None:
    """Return None when config, index and every shard named by the index are present, else a reason."""
    config = os.path.join(directory, "config.json")
    index = os.path.join(directory, INDEX_FILE)
    if not os.path.isfile(config):
        return f"{config} missing"
    if not os.path.isfile(index):
        return f"{index} missing"
    try:
        with open(index) as handle:
            weight_map = json.load(handle).get("weight_map") or {}
    except (OSError, ValueError) as error:
        return f"{index} unreadable: {error}"
    if not isinstance(weight_map, dict) or not weight_map:
        return f"{index} has no weight_map"
    missing = sorted({name for name in weight_map.values() if not os.path.isfile(os.path.join(directory, name))})
    return f"{directory} missing indexed shards {missing[:5]}" if missing else None


def hf_export_complete(directory: str) -> bool:
    """Whether the minimal served HF directory is complete, without loading model tensors."""
    return hf_export_status(directory) is None


# ---------------------------------------------------------------- sidecars: exact base identity ---------------------
def sync_sidecars(base: str, out: str) -> dict[str, list[str]]:
    """Make ``out``'s sidecars EXACTLY the base's: copy every base sidecar verbatim, remove every one the base lacks.

    Idempotent; ``config.json``, the shards and the index are never touched; every file lands through a temporary and a
    replace, the template directory through a staged copy and renames (a recovery never half-writes). Returns
    ``{"copied", "removed"}``.
    """
    copied, removed = [], []
    for name in SIDECAR_FILES:
        src, dst = os.path.join(base, name), os.path.join(out, name)
        if os.path.isfile(src):
            part = dst + ".copying"
            shutil.copy2(src, part)
            os.replace(part, dst)
            copied.append(name)
        elif os.path.lexists(dst):
            os.remove(dst)
            removed.append(name)
    for name in SIDECAR_DIRS:
        src, dst = os.path.join(base, name), os.path.join(out, name)
        if os.path.isdir(src):
            # Through a staged copy and renames: the directory is never absent or half-written in ``out``.
            part, replaced = dst + ".copying", dst + ".replaced"
            for leftover in (part, replaced):
                if os.path.isdir(leftover):
                    shutil.rmtree(leftover)
            shutil.copytree(src, part)
            if os.path.isdir(dst):
                os.rename(dst, replaced)
            os.rename(part, dst)
            if os.path.isdir(replaced):
                shutil.rmtree(replaced)
            copied.append(name + "/")
        elif os.path.isdir(dst):
            shutil.rmtree(dst)
            removed.append(name + "/")
    return {"copied": copied, "removed": removed}


def verify_sidecars(out: str, base: str, *, unified_chat_template: bool = False) -> dict[str, Any]:
    """Torch-free: every base sidecar exists in ``out`` byte-identical and no sidecar the base lacks exists there. Under
    ``unified_chat_template`` the ruled exceptions are the two template files (``UNIFIED_TEMPLATE_FILES``): checked by
    ``_verify_unified_file`` instead of byte identity and listed under ``sidecars_unified_chat_template``; and the
    ``additional_chat_templates/`` directory is refused, in the base and in the export, by the helper the preflight uses
    (``unified_template_directory_conflict``: whatever it holds)."""
    n = 0
    unified: list[str] = []
    for name in SIDECAR_FILES:
        src, dst = os.path.join(base, name), os.path.join(out, name)
        if os.path.isfile(src):
            if not os.path.isfile(dst):
                raise ConversionError(f"verify: {dst} missing (base has {src})")
            if unified_chat_template and name in UNIFIED_TEMPLATE_FILES:
                _verify_unified_file(out, base, name)
                unified.append(name)
                continue
            if not filecmp.cmp(src, dst, shallow=False):
                raise ConversionError(f"verify: {dst} differs from base {src}")
            if unified_chat_template and name == "chat_template.jinja":
                # The tokenizer loader prefers this file over tokenizer_config.json: under the option it may exist only
                # with the canonical text.
                with open(dst, encoding="utf-8") as handle:
                    if handle.read() != canonical_processor_template(base):
                        raise ConversionError(
                            f"verify: {dst} is a chat_template.jinja whose text is not the canonical processor template; the "
                            f"tokenizer loader prefers it over tokenizer_config.json, so the export would not render the "
                            f"template --unified-chat-template promises"
                        )
            n += 1
        elif os.path.lexists(dst):
            raise ConversionError(
                f"verify: {dst} exists but the base has no {name}; it would shadow the base's tokenizer/processor files on load"
            )
    for name in SIDECAR_DIRS:
        src, dst = os.path.join(base, name), os.path.join(out, name)
        if unified_chat_template:
            # ONE rule with the preflight: the directory is refused whatever it holds, base and export.
            for role, root in (("base", base), ("export", out)):
                record = unified_template_directory_conflict(root, name)
                if record is not None:
                    raise ConversionError(
                        f"verify: the {role} {root} carries {describe_template_conflict(record)}; not allowed under "
                        f"--unified-chat-template (the rule the preflight applies before anything is published)"
                    )
            continue
        if os.path.isdir(src):
            if not os.path.isdir(dst):
                raise ConversionError(f"verify: {dst} missing (base has {src})")
            files = sorted(os.listdir(src))
            if files != sorted(os.listdir(dst)):
                raise ConversionError(f"verify: {dst} lists different files than base {src}")
            _, mismatched, errors = filecmp.cmpfiles(src, dst, files, shallow=False)
            if mismatched or errors:
                raise ConversionError(f"verify: {dst} differs from base {src} on {sorted(mismatched + errors)}")
            n += 1
        elif os.path.isdir(dst):
            raise ConversionError(f"verify: {dst} exists but the base has no {name}/")
    report: dict[str, Any] = {"sidecars_identical": n}
    if unified_chat_template:
        report["sidecars_unified_chat_template"] = unified
    return report


# ---------------------------------------------------------------- verification --------------------------------------
def verify_index(
    out: str, expect_keys: set[str] | None = None, expect_tensors: int = 0, min_tensors: int = 800
) -> dict[str, int]:
    """Torch-free structural checks on ``out``: required files, complete shards holding exactly the indexed tensors,
    tensor-count floor (and exact count when ``expect_tensors`` > 0), weight-map coverage of ``expect_keys``."""
    for name in REQUIRED_FILES:
        if not os.path.isfile(os.path.join(out, name)):
            raise ConversionError(f"verify: {out}/{name} missing")
    index_path = os.path.join(out, INDEX_FILE)
    with open(index_path) as handle:
        weight_map = json.load(handle).get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ConversionError(f"verify: weight_map missing/empty in {index_path}")
    shards = sorted(set(weight_map.values()))
    total = 0
    for shard in shards:
        path = os.path.join(out, shard)
        reason = shard_status(path, {k for k, v in weight_map.items() if v == shard})
        if reason:
            raise ConversionError(f"verify: shard listed in {index_path}: {reason}")
        total += os.path.getsize(path)
    if len(weight_map) < min_tensors:
        raise ConversionError(
            f"verify: {index_path} weight_map lists {len(weight_map)} tensors, fewer than the {min_tensors} floor"
        )
    if expect_tensors and len(weight_map) != expect_tensors:
        raise ConversionError(
            f"verify: {index_path} weight_map has {len(weight_map)} tensors, expected exactly {expect_tensors}"
        )
    if expect_keys is not None:
        missing = sorted(expect_keys - set(weight_map))
        extra = sorted(set(weight_map) - expect_keys)
        if missing or extra:
            raise ConversionError(
                f"verify: {index_path} weight_map != model.pt keys; missing={missing[:10]} extra={extra[:10]}"
            )
    return {"n_tensors": len(weight_map), "n_shards": len(shards), "bytes": total}


def _placeholder(config: Any, name: str) -> Any:
    # The same placeholder projection config/build.py hashes: top-level first, then vision_config.
    value = getattr(config, name, None)
    if value is None and getattr(config, "vision_config", None) is not None:
        value = getattr(config.vision_config, name, None)
    return value


def tensor_sha256(tensor: Any) -> str:
    """sha256 of a tensor's bytes (row-major, its own dtype), the digest ``.materialized`` keeps per tensor."""
    import torch

    flat = tensor.detach().reshape(-1).contiguous().cpu()  # [numel]
    if flat.numel() == 0:
        return hashlib.sha256(b"").hexdigest()
    return hashlib.sha256(flat.view(torch.uint8).numpy()).hexdigest()


def verify_with_torch(
    out: str,
    base: str,
    model_pt: str,
    n_expert_samples: int = 4,
    *,
    compare_base: bool = True,
    unified_chat_template: bool = False,
    payload: BinaryIO | None = None,
) -> dict[str, Any]:
    """Run-time-relevant checks on the OUTPUT directory: config identity with the base, on-disk expert layout equal to
    the checkpoint layout, EVERY tensor of ``model.pt`` bitwise equal to its shard tensor (with per-tensor sha256
    digests kept in the report), optionally which tensors are bitwise equal to the base's, and the processor /
    tokenizer identity ``build_vllm_teacher`` demands (processor type, chat templates, full vocabulary, special ids,
    image and video processor dicts).

    With ``payload`` the checkpoint tensors come from THAT open file object
    (``load_consolidated_state_dict``; the same one the materializer hashed under its lock and the overlay consumed), never
    from a fresh resolution of ``model_pt``: a path replaced by the merge tool meanwhile can neither fail a correct build
    nor pass a wrong one. Without it the path is loaded, for standalone callers.

    The earlier single-tensor probe (``model.language_model.norm.weight``) was the one tensor a bf16 run does not move,
    so it passed against the base too; the full comparison runs before any caller deletes
    ``model.pt``. Dense checkpoints have no expert layout to check. The base's weight index, not absence in a possibly
    incomplete checkpoint, determines whether expert tensors are required. The caller applies its configured
    tensor-count floor; this check instead requires exact output/checkpoint key equality.
    """
    import torch
    from safetensors import safe_open
    from transformers import AutoConfig, AutoProcessor

    report: dict[str, Any] = {}
    config, base_config = AutoConfig.from_pretrained(out), AutoConfig.from_pretrained(base)
    if config.architectures != base_config.architectures or config.model_type != base_config.model_type:
        raise ConversionError(
            f"verify: {out}/config.json identity {config.architectures}/{config.model_type} differs from base "
            f"{base_config.architectures}/{base_config.model_type}"
        )
    for name in (
        "image_token_id",
        "video_token_id",
        "vision_start_token_id",
        "vision_end_token_id",
        "spatial_merge_size",
        "patch_size",
        "temporal_patch_size",
    ):
        if _placeholder(config, name) != _placeholder(base_config, name):
            raise ConversionError(
                f"verify: {out}/config.json placeholder {name}={_placeholder(config, name)!r} differs from base "
                f"{_placeholder(base_config, name)!r}"
            )
    report["architectures"] = list(config.architectures or [])

    state_dict = load_consolidated_state_dict(model_pt, payload)  # values: [*shape]
    weight_map = _read_weight_map(os.path.join(out, INDEX_FILE))
    verify_index(out, expect_keys=set(state_dict), min_tensors=0)
    expert_keys = sorted(k for k in state_dict if ".mlp.experts." in k)
    base_has_experts = any(".mlp.experts." in key for key in _read_weight_map(os.path.join(base, INDEX_FILE)))
    if base_has_experts and not expert_keys:
        raise ConversionError(f"verify: no fused MoE expert tensors in {model_pt} (base {base} contains experts)")
    step = max(1, len(expert_keys) // n_expert_samples)
    checked = 0
    for key in expert_keys[::step][:n_expert_samples]:
        shard = os.path.join(out, weight_map[key])
        with safe_open(shard, framework="pt") as handle:
            disk_shape = list(handle.get_slice(key).get_shape())
        if disk_shape != list(state_dict[key].shape):
            raise ConversionError(
                f"verify: {shard}: expert {key} shape {disk_shape} != checkpoint shape {list(state_dict[key].shape)}; "
                f"the vLLM layout rewrite did not land"
            )
        checked += 1
    report["expert_shape_checks"] = checked
    # Every tensor of the checkpoint against the shard that holds it, shard by shard, bitwise; the digests stay.
    digests: dict[str, str] = {}
    unequal: list[str] = []
    by_shard: dict[str, list[str]] = {}
    for key in state_dict:
        by_shard.setdefault(weight_map[key], []).append(key)
    for shard_name, keys in sorted(by_shard.items()):
        shard = os.path.join(out, shard_name)
        with safe_open(shard, framework="pt") as handle:
            for key in sorted(keys):
                disk = handle.get_tensor(key)  # [*shape]
                expected = state_dict[key]  # [*shape]
                if (
                    tuple(disk.shape) != tuple(expected.shape)
                    or disk.dtype != expected.dtype
                    or not torch.equal(disk, expected)
                ):
                    unequal.append(key)
                digests[key] = tensor_sha256(expected)
    if unequal:
        raise ConversionError(
            f"verify: {len(unequal)} of {len(state_dict)} tensors differ bitwise between {model_pt} and the shards under "
            f"{out}; each listed tensor differs bitwise from the checkpoint: {unequal[:10]}"
        )
    report["bitwise_equal_tensors"] = len(digests)
    report["tensor_sha256"] = digests
    # The classic probe name is kept for readers of older markers; it now names the whole comparison.
    report["bitwise_probe"] = "all_tensors"
    if compare_base:
        base_map = _read_weight_map(os.path.join(base, INDEX_FILE))
        equal_to_base: list[str] = []
        compared = 0
        base_by_shard: dict[str, list[str]] = {}
        for key in state_dict:
            if key in base_map:
                base_by_shard.setdefault(base_map[key], []).append(key)
        for shard_name, keys in sorted(base_by_shard.items()):
            with safe_open(os.path.join(base, shard_name), framework="pt") as handle:
                for key in sorted(keys):
                    reference = handle.get_tensor(key)  # [*shape]
                    expected = state_dict[key]  # [*shape] (the checkpoint layout, as the base's)
                    compared += 1
                    if (
                        tuple(reference.shape) == tuple(expected.shape)
                        and reference.dtype == expected.dtype
                        and torch.equal(reference, expected)
                    ):
                        equal_to_base.append(key)
        report["tensors_compared_to_base"] = compared
        report["tensors_equal_to_base"] = len(equal_to_base)
        report["tensors_equal_to_base_names"] = equal_to_base
        report["tensors_moved_vs_base"] = compared - len(equal_to_base)
    del state_dict

    # Shared-vocabulary identity (config/build.py) must hold or build_vllm_teacher refuses the teacher: processor type,
    # tokenizer chat_template (exact string), full vocab, the special-id fields, component types and to_dict().
    processor_out, processor_base = AutoProcessor.from_pretrained(out), AutoProcessor.from_pretrained(base)
    if type(processor_out).__name__ != type(processor_base).__name__:
        raise ConversionError(
            f"verify: {out} processor type {type(processor_out).__name__} differs from base {type(processor_base).__name__}"
        )
    # Under --unified-chat-template the loaded processor's template (chat_template.json) must be the canonical text too.
    expected_processor_template = (
        canonical_processor_template(base) if unified_chat_template else getattr(processor_base, "chat_template", None)
    )
    if getattr(processor_out, "chat_template", None) != expected_processor_template:
        against = (
            f"the canonical processor template of base {base} (--unified-chat-template, {TEMPLATE_MODE_CANONICAL})"
            if unified_chat_template
            else f"base {base}"
        )
        raise ConversionError(f"verify: {out} processor chat_template differs from {against}")
    tokenizer_out, tokenizer_base = processor_out.tokenizer, processor_base.tokenizer
    if type(tokenizer_out).__name__ != type(tokenizer_base).__name__:
        raise ConversionError(
            f"verify: {out} tokenizer type {type(tokenizer_out).__name__} differs from base {type(tokenizer_base).__name__}"
        )
    template_out = getattr(tokenizer_out, "chat_template", None)
    template_base = getattr(tokenizer_base, "chat_template", None)
    # Under --unified-chat-template the loaded tokenizer must carry the base's PROCESSOR template in its canonical form,
    # not the base's tokenizer template (the ruled sidecar differences; verify_sidecars asserted the file-level rule).
    expected_template = canonical_processor_template(base) if unified_chat_template else template_base
    if template_out != expected_template:
        source = (
            "chat_template.jinja"
            if os.path.isfile(os.path.join(out, "chat_template.jinja"))
            else "tokenizer_config.json"
        )
        against = (
            f"the canonical processor template of base {base} (--unified-chat-template, {TEMPLATE_MODE_CANONICAL})"
            if unified_chat_template
            else f"base {base}"
        )
        raise ConversionError(
            f"verify: {out} tokenizer chat_template (len {len(template_out or '')}, read from {out}/{source}) differs from "
            f"{against} (len {len(expected_template or '')}); a chat_template.jinja shadows tokenizer_config.json's "
            f"chat_template on load"
        )
    if len(tokenizer_out) != len(tokenizer_base) or tokenizer_out.get_vocab() != tokenizer_base.get_vocab():
        raise ConversionError(f"verify: {out} tokenizer vocabulary differs from base {base}")
    for name in (
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
        "unk_token_id",
        "additional_special_tokens_ids",
        "all_special_ids",
    ):
        if getattr(tokenizer_out, name, None) != getattr(tokenizer_base, name, None):
            raise ConversionError(
                f"verify: {out} tokenizer {name}={getattr(tokenizer_out, name, None)!r} differs from base "
                f"{getattr(tokenizer_base, name, None)!r}"
            )
    for component in ("image_processor", "video_processor"):
        a, b = getattr(processor_out, component, None), getattr(processor_base, component, None)
        if (a is None) != (b is None):
            raise ConversionError(f"verify: {component} present on one side only ({out} vs {base})")
        if a is not None:
            if type(a).__name__ != type(b).__name__:
                raise ConversionError(
                    f"verify: {out} {component} type {type(a).__name__} differs from base {type(b).__name__}"
                )
            da, db = a.to_dict(), b.to_dict()
            if da != db:
                diff = sorted(k for k in set(da) | set(db) if da.get(k) != db.get(k))
                raise ConversionError(f"verify: {out} {component} config differs from base {base} on keys {diff}")
    report["processor_identity"] = "equal"
    report["tokenizer_chat_template"] = TEMPLATE_MODE_CANONICAL if unified_chat_template else "base"
    report["processor_chat_template"] = TEMPLATE_MODE_CANONICAL if unified_chat_template else "base"
    return report
