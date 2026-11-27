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


class _FakeCredentialParser(EnvParser):
    ACCESS_KEY = String(default=None)
    SECRET_KEY = String(default=None)


def test_serialization_round_trip_does_not_echo_credentials(monkeypatch, capsys, caplog):
    """A credential parser must round-trip through to_b64/reconstruction with
    neither the plaintext secret nor its reversible base64 payload appearing
    on stdout or in logs (previously both were printed, and reconstruction
    log.critical'd the payload)."""
    import base64
    import logging

    secret = "FAKE-SECRET-sentinel-123"
    monkeypatch.setenv("ACCESS_KEY", "FAKE-AK")
    monkeypatch.setenv("SECRET_KEY", secret)

    with caplog.at_level(logging.DEBUG):
        b64 = _FakeCredentialParser().to_b64()
        monkeypatch.delenv("SECRET_KEY", raising=False)
        restored = _FakeCredentialParser(b64)

    assert restored.SECRET_KEY == secret

    captured = capsys.readouterr()
    leaked_text = captured.out + captured.err + caplog.text
    assert secret not in leaked_text
    assert b64 not in leaked_text
    # Belt and braces: no encoding of the secret anywhere in the output.
    assert base64.b64encode(secret.encode()).decode()[:16] not in leaked_text
