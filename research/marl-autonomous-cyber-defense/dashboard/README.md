# Sentinel MARL GPU Training Console

A responsive web control plane for launching and monitoring single-seed CC4 policy training jobs on the GPU server. The console controls the real training scripts; it does not generate mock results.

## 1. Server setup

Use Python 3.10 in a dedicated virtual environment. First install a PyTorch build compatible with the installed NVIDIA driver and CUDA runtime using the official PyTorch installation selector. Verify that `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"` reports the expected GPU.

From the research project root:

```bash
python -m venv .venv
source .venv/bin/activate
# Install the CUDA-enabled PyTorch build appropriate for this server first.
python -m pip install -r requirements-cc4-validation.txt
python -m pip install -e .
python -m pip install -r dashboard/requirements-server.txt
git clone https://github.com/cage-challenge/cage-challenge-4 vendor/CybORG
git -C vendor/CybORG checkout 8c3c50ca54b176c2de199847944e8dcc035497e3
```

The checkout must match `configs/cc4.lock.yaml`. Do not silently update the simulator revision. If the repository already exists, verify its commit rather than cloning over it.

Validate the environment before training:

```bash
python scripts/validate_cyborg.py --one-step --verify-seed
python -m pytest -q
```

## 2. Start the dashboard

```bash
bash dashboard/start_dashboard.sh
```

The server binds to `127.0.0.1:8787` by default and generates a high-entropy bearer token in `.dashboard_token` with owner-only file permissions. The token is printed once when generated; keep it private. The file is ignored by Git. To use a different port, set `PORT=8788`.

### Remote access (recommended)

Keep the service bound to loopback on the GPU server. From your workstation, create an SSH tunnel:

```bash
ssh -L 8787:127.0.0.1:8787 your-user@your-gpu-server
```

Then open `http://127.0.0.1:8787` locally and paste the token from the server's `.dashboard_token` file. Do not expose the dashboard port directly to the public Internet. For shared access, put it behind a properly configured HTTPS reverse proxy, firewall and identity-aware access control.

The API requires the bearer token on every control and data endpoint. The browser stores it in session storage only.

## 3. Launch a model

The dashboard supports one active policy-training job at a time:

- **RA-HMAPPO** — hierarchical manager, task-conditioned worker, local risk input and centralized team critic. Requires a trained risk checkpoint. The default path is `artifacts/standard_cc4/cc4_risk.pt`.
- **MAPPO** — decentralized actors and centralized team critic.
- **IPPO** — independent actor-critic policies per Blue agent.

Choose the algorithm, GPU, seed, total environment steps, rollout horizon and episode horizon, then click **Start training**. The backend invokes the corresponding Python training script with argument arrays (not a shell command string). Runs are stored under `artifacts/dashboard_runs/<job-id>/`.

For the RA-HMAPPO option, first generate the four disjoint risk datasets and train the risk estimator using the standard research pipeline, or point the dashboard to an existing calibrated risk checkpoint. The dashboard does not silently substitute a dummy risk score.

## 4. Live controls and progress

The dashboard polls authenticated endpoints for job state and telemetry. Every completed rollout/update records:

- Environment steps completed and total, progress percentage, elapsed time, estimated time remaining and throughput.
- Actor/policy loss, critic/value loss, entropy and available episode returns.
- CUDA device name, allocated and reserved GPU memory.
- Model metadata, agent count, observation/action dimensions, global-state dimension, parameter count and hyperparameters.
- Process logs and checkpoint inventory.

A rolling checkpoint containing model weights, optimizer state, update count and configuration is atomically replaced after every update. The final training script also writes the final checkpoint and training history.

**Pause** and **Resume** suspend and continue the same process on Linux; they do not reset the simulator. **Stop** sends an interrupt and preserves the latest rolling checkpoint. It is a process-control checkpoint, not a guaranteed bit-exact training-resume feature: the current trainers do not yet restore simulator state and continue from an optimizer checkpoint after a server restart.

## 5. Security and limitations

- Bind to loopback by default; use SSH tunneling for remote access.
- Use a unique secret token and protect the server account and SSH credentials.
- Do not run the dashboard as root.
- The service currently supports one job per server process and is intended for a trusted research machine.
- GPU utilization percentage is not reported unless a supported telemetry provider is added; GPU memory is reported through PyTorch.
- Training progress is measured in environment steps, not wall-clock percent.
- A rolling checkpoint is written after each PPO update; stopping during an update preserves the preceding completed update.
- The dashboard starts individual policy training runs. The full 10-seed/100-evaluation benchmark matrix remains a separate workflow in `scripts/run_standard_cc4_matrix.sh`.
- This dashboard does not claim research performance, convergence or publication readiness. Results must come from completed experiments and held-out evaluation.

## Troubleshooting

- **CUDA not detected:** check `nvidia-smi`, install the correct CUDA-enabled PyTorch wheel for the host, and restart the dashboard process.
- **Pinned CC4 source missing:** clone/check out the exact commit above into `vendor/CybORG`.
- **RA-HMAPPO refuses to start:** train the calibrated risk model or select an existing checkpoint inside the project directory.
- **401 Unauthorized:** reconnect with the token in `.dashboard_token`; tokens are kept only in the browser session.
- **No progress bar movement:** progress updates arrive after each rollout/update. Check the process log panel and ensure the simulator is stepping.
