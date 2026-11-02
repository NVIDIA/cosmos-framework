import json
import os

from cosmos_framework.utils.env_parsers.env_parser import EnvParser
from cosmos_framework.utils.validator import Int, String


class _TinyParser(EnvParser):
    NAME = String(default="svc")
    PORT = Int(default=8000)


def test_to_json_serializes_subclass_validators(monkeypatch):
    """to_json must serialize the concrete parser's validators, not the empty
    base class __dict__ (which made every real parser emit "{}")."""
    monkeypatch.setenv("PORT", "1234")
    data = json.loads(_TinyParser().to_json())
    assert set(data) == {"NAME", "PORT"}
    assert data["PORT"] == 1234
    assert data["NAME"] == "svc"


def test_to_b64_round_trips(monkeypatch):
    """to_b64 -> reconstruct must restore values (was broken because to_json
    serialized nothing)."""
    monkeypatch.setenv("PORT", "4321")
    b64 = _TinyParser().to_b64()
    monkeypatch.delenv("PORT", raising=False)
    restored = _TinyParser(b64)
    assert restored.PORT == 4321
