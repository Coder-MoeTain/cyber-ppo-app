"""Authenticated local control plane for CC4 GPU training jobs.

Bind to 127.0.0.1 and use an SSH tunnel for remote access. Never expose this
service directly to the public Internet.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import hmac
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time
import uuid

import torch
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
DASHBOARD = ROOT / "dashboard"
RUNS = ROOT / "artifacts" / "dashboard_runs"
SCRIPTS = ROOT / "scripts"
TOKEN = os.environ.get("SENTINEL_DASHBOARD_TOKEN", "")
if len(TOKEN) < 24:
    raise RuntimeError("Set SENTINEL_DASHBOARD_TOKEN to a random value of at least 24 characters")

app = FastAPI(title="Sentinel MARL Training Control Plane", docs_url=None, redoc_url=None)
app.mount("/assets", StaticFiles(directory=str(DASHBOARD)), name="assets")
LOCK = threading.RLock()
JOB = None
PROCESS = None


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def require_auth(authorization: str | None = Header(default=None)):
    expected = "Bearer " + TOKEN
    if not authorization or not hmac.compare_digest(authorization, expected):
        raise HTTPException(status_code=401, detail="Authentication required")


class StartRequest(BaseModel):
    algorithm: str = Field(default="RA-HMAPPO", pattern="^(RA-HMAPPO|MAPPO|IPPO)$")
    steps: int = Field(default=1_000_000, ge=256, le=50_000_000)
    rollout_steps: int = Field(default=256, ge=32, le=4096)
    episode_steps: int = Field(default=500, ge=50, le=2000)
    seed: int = Field(default=11, ge=0, le=2_147_483_647)
    device: str = Field(default="auto", pattern="^(auto|cpu|cuda(:[0-9]+)?)$")
    risk_checkpoint: str | None = None
    run_name: str = Field(default="cc4-training", min_length=1, max_length=48, pattern="^[A-Za-z0-9_-]+$")


def job_status():
    global JOB, PROCESS
    with LOCK:
        if JOB is None:
            return {"status": "idle"}
        state = dict(JOB)
        if PROCESS is not None and PROCESS.poll() is not None and JOB["status"] not in {
            "completed", "failed", "stopped"
        }:
            code = PROCESS.returncode
            JOB["status"] = "completed" if code == 0 else "failed"
            JOB["exit_code"] = code
            JOB["finished_at"] = utcnow()
            state = dict(JOB)
        progress_path = Path(JOB["progress_file"])
        latest = None
        if progress_path.is_file():
            try:
                lines = progress_path.read_text(encoding="utf-8").splitlines()
                for line in reversed(lines):
                    row = json.loads(line)
                    if row.get("event") == "training_update":
                        latest = row
                        break
                    if row.get("event") == "training_finished":
                        state["trainer_event"] = row
                        break
            except (OSError, ValueError, json.JSONDecodeError):
                pass
        state["progress"] = latest
        state["process_alive"] = PROCESS is not None and PROCESS.poll() is None
        return state


@app.get("/")
def index():
    return FileResponse(DASHBOARD / "index.html")


@app.get("/api/health")
def health():
    return {"service": "sentinel-marl", "time": utcnow(), "auth": "bearer-token"}


@app.get("/api/system")
def system(_: None = Depends(require_auth)):
    cuda = bool(torch.cuda.is_available())
    devices = []
    if cuda:
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            devices.append({
                "id": f"cuda:{index}", "name": props.name,
                "memory_total_bytes": int(props.total_memory),
                "memory_allocated_bytes": int(torch.cuda.memory_allocated(index)),
                "memory_reserved_bytes": int(torch.cuda.memory_reserved(index)),
            })
    cyborg_path = ROOT / "vendor" / "CybORG"
    return {
        "project_root": str(ROOT),
        "python": sys.version.split()[0],
        "pytorch": torch.__version__,
        "cuda_available": cuda,
        "cuda_version": torch.version.cuda,
        "devices": devices,
        "cyborg_source_present": cyborg_path.is_dir(),
        "risk_checkpoint_default": str(ROOT / "artifacts" / "standard_cc4" / "cc4_risk.pt"),
        "disk_free_bytes": __import__("shutil").disk_usage(ROOT).free,
    }


@app.get("/api/status")
def status(_: None = Depends(require_auth)):
    return job_status()


@app.post("/api/train/start")
def start_training(request: StartRequest, _: None = Depends(require_auth)):
    global JOB, PROCESS
    with LOCK:
        current = job_status()
        if current.get("status") in {"starting", "running", "paused", "stopping"}:
            raise HTTPException(409, "A training job is already active")
        cyborg = ROOT / "vendor" / "CybORG"
        if not cyborg.is_dir():
            raise HTTPException(400, "Pinned CybORG source is missing. Follow the CC4 setup guide first.")
        device = request.device
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise HTTPException(400, "CUDA was requested but PyTorch cannot see an NVIDIA GPU")
            index = torch.device(device).index or 0
            if index >= torch.cuda.device_count():
                raise HTTPException(400, f"GPU index {index} is unavailable")
        risk_path = None
        if request.algorithm == "RA-HMAPPO":
            raw = request.risk_checkpoint or "artifacts/standard_cc4/cc4_risk.pt"
            candidate = (ROOT / raw).resolve()
            if ROOT not in candidate.parents or not candidate.is_file():
                raise HTTPException(400, "RA-HMAPPO requires an existing risk checkpoint inside the project directory")
            risk_path = candidate

        job_id = uuid.uuid4().hex[:12]
        run_dir = RUNS / job_id
        run_dir.mkdir(parents=True, exist_ok=False)
        checkpoint = run_dir / f"{request.algorithm.lower().replace('-', '_')}_seed{request.seed}.pt"
        progress_file = run_dir / "progress.jsonl"
        log_file = run_dir / "training.log"
        script = {
            "RA-HMAPPO": "train_ra_hmappo_cc4.py",
            "MAPPO": "train_mappo_cc4.py",
            "IPPO": "train_ippo_cc4.py",
        }[request.algorithm]
        command = [
            sys.executable, str(SCRIPTS / script),
            "--steps", str(request.steps),
            "--rollout-steps", str(request.rollout_steps),
            "--episode-steps", str(request.episode_steps),
            "--seed", str(request.seed),
            "--device", device,
            "--output", str(checkpoint),
            "--progress-file", str(progress_file),
            "--rolling-checkpoint", str(checkpoint) + ".latest.pt",
        ]
        if risk_path:
            command += ["--risk-checkpoint", str(risk_path)]
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join([
            str(ROOT / "src"), str(cyborg), env.get("PYTHONPATH", "")
        ]).rstrip(os.pathsep)
        env["PYTHONUNBUFFERED"] = "1"
        metadata = {
            "job_id": job_id, "run_name": request.run_name,
            "algorithm": request.algorithm, "status": "starting",
            "device": device, "seed": request.seed,
            "total_steps": request.steps, "rollout_steps": request.rollout_steps,
            "episode_steps": request.episode_steps,
            "risk_checkpoint": str(risk_path) if risk_path else None,
            "started_at": utcnow(), "command": command,
            "checkpoint": str(checkpoint),
            "rolling_checkpoint": str(checkpoint) + ".latest.pt",
            "progress_file": str(progress_file), "log_file": str(log_file),
        }
        (run_dir / "job.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        try:
            with log_file.open("w", encoding="utf-8") as log:
                PROCESS = subprocess.Popen(
                    command, cwd=ROOT, env=env, stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True,
                )
        except OSError as exc:
            raise HTTPException(500, f"Could not start training process: {exc}") from exc
        metadata["pid"] = PROCESS.pid
        metadata["status"] = "running"
        JOB = metadata
        return job_status()


@app.post("/api/train/pause")
def pause(_: None = Depends(require_auth)):
    with LOCK:
        state = job_status()
        if PROCESS is None or PROCESS.poll() is not None or state["status"] != "running":
            raise HTTPException(409, "No running training job to pause")
        os.killpg(PROCESS.pid, signal.SIGSTOP)
        JOB["status"] = "paused"
        return job_status()


@app.post("/api/train/resume")
def resume(_: None = Depends(require_auth)):
    with LOCK:
        state = job_status()
        if PROCESS is None or PROCESS.poll() is not None or state["status"] != "paused":
            raise HTTPException(409, "No paused training job to resume")
        os.killpg(PROCESS.pid, signal.SIGCONT)
        JOB["status"] = "running"
        return job_status()


@app.post("/api/train/stop")
async def stop(_: None = Depends(require_auth)):
    global PROCESS
    with LOCK:
        state = job_status()
        if PROCESS is None or PROCESS.poll() is not None:
            raise HTTPException(409, "No active training job")
        JOB["status"] = "stopping"
        process = PROCESS
    os.killpg(process.pid, signal.SIGINT)
    try:
        await asyncio.wait_for(asyncio.to_thread(process.wait), timeout=10)
    except asyncio.TimeoutError:
        os.killpg(process.pid, signal.SIGTERM)
        await asyncio.to_thread(process.wait)
    with LOCK:
        JOB["status"] = "stopped"
        JOB["finished_at"] = utcnow()
        JOB["exit_code"] = process.returncode
        return job_status()


@app.get("/api/logs")
def logs(_: None = Depends(require_auth), lines: int = 200):
    state = job_status()
    if not state.get("log_file"):
        return {"lines": []}
    path = Path(state["log_file"])
    if not path.is_file():
        return {"lines": []}
    return {"lines": path.read_text(encoding="utf-8", errors="replace").splitlines()[-max(1, min(lines, 1000)): ]}


@app.get("/api/model")
def model_details(_: None = Depends(require_auth)):
    state = job_status()
    checkpoint = state.get("rolling_checkpoint") or state.get("checkpoint")
    if not checkpoint or not Path(checkpoint).is_file():
        return {"available": False, "message": "A training checkpoint will appear after the first completed rollout."}
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        weights = payload.get("model_state_dict", {})
        metadata = payload.get("metadata", {})
        parameters = sum(int(value.numel()) for value in weights.values() if torch.is_tensor(value))
        return {
            "available": True,
            "algorithm": metadata.get("algorithm", state.get("algorithm")),
            "seed": metadata.get("seed", state.get("seed")),
            "environment_steps": payload.get("environment_steps"),
            "total_steps": payload.get("total_steps"),
            "parameter_count": parameters,
            "agents": metadata.get("agents", []),
            "observation_dim": metadata.get("observation_dim"),
            "action_dims": metadata.get("action_dims"),
            "global_state_dim": metadata.get("global_state_dim"),
            "config": metadata.get("config", {}),
            "device": metadata.get("device"),
            "checkpoint": checkpoint,
            "saved_at_utc": payload.get("saved_at_utc"),
        }
    except (OSError, RuntimeError, ValueError, TypeError) as exc:
        return {"available": False, "message": f"Checkpoint metadata unavailable: {exc}"}


@app.get("/api/checkpoints")
def checkpoints(_: None = Depends(require_auth)):
    if not RUNS.exists():
        return {"items": []}
    items = []
    for path in sorted(RUNS.glob("*/*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)[:100]:
        items.append({
            "name": path.name, "path": str(path.relative_to(ROOT)),
            "size_bytes": path.stat().st_size,
            "modified_at": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
            "rolling": path.name.endswith(".latest.pt"),
        })
    return {"items": items}


@app.get("/api/checkpoints/download")
def download_checkpoint(path: str, _: None = Depends(require_auth)):
    candidate = (ROOT / path).resolve()
    allowed = (ROOT / "artifacts" / "dashboard_runs").resolve()
    if allowed not in candidate.parents or not candidate.is_file() or candidate.suffix != ".pt":
        raise HTTPException(404, "Checkpoint not found")
    return FileResponse(candidate, filename=candidate.name, media_type="application/octet-stream")
