"""Regression tests for dashboard-facing training telemetry."""
import importlib.util
import json
from pathlib import Path

import torch

MODULE = Path(__file__).parents[1] / "scripts" / "training_telemetry.py"
SPEC = importlib.util.spec_from_file_location("training_telemetry", MODULE)
TELEMETRY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TELEMETRY)


def test_update_writes_progress_and_atomic_checkpoint(tmp_path):
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    progress = tmp_path / "progress.jsonl"
    checkpoint = tmp_path / "latest.pt"
    telemetry = TELEMETRY.TrainingTelemetry(
        progress, checkpoint, model, optimizer, 1000, "cpu",
        metadata={"algorithm": "MAPPO", "seed": 11},
    )

    telemetry.on_update({
        "environment_steps": 128,
        "actor_loss": 0.25,
        "critic_loss": 0.5,
        "episode_returns": [1.0, 2.0],
    })

    row = json.loads(progress.read_text().splitlines()[0])
    assert row["event"] == "training_update"
    assert row["progress_percent"] == 12.8
    assert row["update_index"] == 1
    assert row["gpu"]["requested_device"] == "cpu"
    assert checkpoint.is_file()
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert saved["environment_steps"] == 128
    assert saved["metadata"]["algorithm"] == "MAPPO"
    assert "optimizer_state_dict" in saved


def test_finish_event_is_appended(tmp_path):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters())
    progress = tmp_path / "progress.jsonl"
    telemetry = TELEMETRY.TrainingTelemetry(
        progress, tmp_path / "latest.pt", model, optimizer, 10, "cpu"
    )
    telemetry.finish("completed", {"wall_seconds": 2.0})
    row = json.loads(progress.read_text().splitlines()[-1])
    assert row["event"] == "training_finished"
    assert row["status"] == "completed"
