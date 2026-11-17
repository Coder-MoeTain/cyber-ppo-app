"""Tests for experiment artifact provenance manifests."""
import importlib.util
from pathlib import Path

import pytest

MODULE = Path(__file__).parents[1] / "scripts" / "artifact_manifest.py"
SPEC = importlib.util.spec_from_file_location("artifact_manifest", MODULE)
MANIFEST = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MANIFEST)


def test_manifest_create_and_verify(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "episode.jsonl").write_text('{"seed": 501}\n', encoding="utf-8")
    output = tmp_path / "results" / "manifest.json"

    created = MANIFEST.create_manifest(["artifacts"], output)
    assert len(created["files"]) == 1
    assert MANIFEST.verify_manifest(output) == 1


def test_manifest_detects_modified_artifact(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    artifact = tmp_path / "checkpoint.bin"
    artifact.write_bytes(b"checkpoint-v1")
    output = tmp_path / "manifest.json"
    MANIFEST.create_manifest(["checkpoint.bin"], output)
    artifact.write_bytes(b"checkpoint-v2")
    with pytest.raises(ValueError, match="modified"):
        MANIFEST.verify_manifest(output)


def test_manifest_rejects_missing_artifact(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    artifact = tmp_path / "record.jsonl"
    artifact.write_text("episode\n", encoding="utf-8")
    output = tmp_path / "manifest.json"
    MANIFEST.create_manifest(["record.jsonl"], output)
    artifact.unlink()
    with pytest.raises(ValueError, match="missing"):
        MANIFEST.verify_manifest(output)
