from __future__ import annotations

import json
from pathlib import Path

from scripts.verify_generalized_homotopy_checkpoint import verify_checkpoint

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "runs/generalized_solver/n3_unequal_waypoint_checkpoint.json"


def test_published_checkpoint_is_internally_consistent() -> None:
    assert verify_checkpoint(CHECKPOINT) == []


def test_checkpoint_detects_artifact_hash_tampering(tmp_path: Path) -> None:
    payload = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    payload["artifacts"]["controller"]["sha256"] = "0" * 64
    tampered = tmp_path / "checkpoint.json"
    tampered.write_text(json.dumps(payload), encoding="utf-8")
    assert "controller artifact hash mismatch" in verify_checkpoint(tampered)
