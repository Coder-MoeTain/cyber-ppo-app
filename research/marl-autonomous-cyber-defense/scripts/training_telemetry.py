"""Atomic checkpoint and JSONL progress telemetry for long-running training jobs."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import time

import torch


def gpu_telemetry(device: str) -> dict:
    requested = str(device)
    available = bool(torch.cuda.is_available())
    data = {
        "requested_device": requested,
        "cuda_available": available,
        "cuda_device_count": torch.cuda.device_count() if available else 0,
        "gpu_name": None,
        "gpu_memory_allocated_bytes": None,
        "gpu_memory_reserved_bytes": None,
        "gpu_memory_total_bytes": None,
        "gpu_utilization_percent": None,
    }
    if requested.startswith("cuda") and available:
        index = torch.device(requested).index or 0
        props = torch.cuda.get_device_properties(index)
        data.update({
            "gpu_name": props.name,
            "gpu_memory_allocated_bytes": int(torch.cuda.memory_allocated(index)),
            "gpu_memory_reserved_bytes": int(torch.cuda.memory_reserved(index)),
            "gpu_memory_total_bytes": int(props.total_memory),
        })
    return data


class TrainingTelemetry:
    def __init__(self, progress_file, checkpoint_file, model, optimizer,
                 total_steps, device, metadata=None):
        self.progress_file = Path(progress_file) if progress_file else None
        self.checkpoint_file = Path(checkpoint_file)
        self.model, self.optimizer = model, optimizer
        self.total_steps, self.device = int(total_steps), str(device)
        self.metadata = dict(metadata or {})
        self.started = time.monotonic()
        self.latest = {}
        self.update_count = 0

    def _append(self, record):
        if self.progress_file is None:
            return
        self.progress_file.parent.mkdir(parents=True, exist_ok=True)
        with self.progress_file.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, default=float, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def on_update(self, metrics):
        steps = int(metrics["environment_steps"])
        self.update_count += 1
        elapsed = max(time.monotonic() - self.started, 1e-9)
        record = {
            "event": "training_update",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "environment_steps": steps,
            "update_index": self.update_count,
            "total_steps": self.total_steps,
            "progress_percent": min(100.0, 100.0 * steps / self.total_steps),
            "elapsed_seconds": elapsed,
            "steps_per_second": steps / elapsed,
            "eta_seconds": max(0.0, (self.total_steps - steps) / max(steps / elapsed, 1e-9)),
            "metrics": metrics,
            "gpu": gpu_telemetry(self.device),
            "model": self.metadata,
        }
        self.latest = record
        self._append(record)
        self.save_checkpoint(steps, record)

    def save_checkpoint(self, steps, progress=None):
        self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "environment_steps": int(steps),
            "total_steps": self.total_steps,
            "progress": progress or self.latest,
            "metadata": self.metadata,
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        fd, temporary = tempfile.mkstemp(
            prefix=self.checkpoint_file.name + ".", suffix=".tmp",
            dir=str(self.checkpoint_file.parent),
        )
        os.close(fd)
        try:
            torch.save(payload, temporary)
            os.replace(temporary, self.checkpoint_file)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def finish(self, status, details=None):
        record = {
            "event": "training_finished",
            "status": status,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "environment_steps": int(self.latest.get("environment_steps", 0)),
            "total_steps": self.total_steps,
            "progress_percent": float(self.latest.get("progress_percent", 0.0)),
            "details": details or {},
        }
        self._append(record)
