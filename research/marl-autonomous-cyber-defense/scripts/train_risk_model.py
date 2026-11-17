"""Train, calibrate, and independently test the CC4 local-observation risk estimator."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn

from autocd.risk_model import RiskEstimatorNet, binary_metrics


def load_dataset(path, agent_names=None, observation_dim=None):
    data = np.load(path, allow_pickle=False)
    names = data["agent_names"].tolist()
    if agent_names is not None and names != agent_names:
        raise ValueError(f"Agent order mismatch in {path}")
    if data["observations"].ndim != 2 or data["observations"].shape[1] == 0:
        raise ValueError(f"Observations in {path} must be a non-empty 2-D matrix")
    if observation_dim is not None and data["observations"].shape[1] != observation_dim:
        raise ValueError(f"Observation dimension mismatch in {path}")
    if len(data["labels"]) == 0 or len(data["labels"]) != len(data["observations"]):
        raise ValueError(f"Labels in {path} are empty or misaligned")
    if not np.isfinite(data["observations"]).all():
        raise ValueError(f"Observations in {path} contain NaN or infinity")
    if not np.isin(data["labels"], [0, 1]).all():
        raise ValueError(f"Labels in {path} must be binary")
    if np.any((data["agent_ids"] < 0) | (data["agent_ids"] >= len(names))):
        raise ValueError(f"agent_ids outside configured range in {path}")
    seeds = set(map(int, data["seeds"].tolist()))
    episode_ids = set(map(int, data["episode_ids"].tolist()))
    if not seeds or not episode_ids or not episode_ids.issubset(seeds):
        raise ValueError(f"Missing or inconsistent seed provenance in {path}")
    return data, names, seeds


def assert_disjoint_splits(named_splits):
    for i, (left_name, left) in enumerate(named_splits):
        for right_name, right in named_splits[i + 1:]:
            overlap = left & right
            if overlap:
                raise ValueError(
                    f"{left_name} and {right_name} episode seeds overlap: {sorted(overlap)}"
                )


def fit_temperature(logits, labels):
    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if logits.size != labels.size or logits.size == 0 or not np.isfinite(logits).all():
        raise ValueError("Calibration logits and labels must be finite, non-empty and aligned")
    if not np.isin(labels, [0, 1]).all() or np.unique(labels).size != 2:
        raise ValueError("Calibration split must contain both binary classes")
    temperatures = np.geomspace(0.25, 4.0, 161)
    losses = [
        float(np.mean(np.logaddexp(0.0, logits / temp) - labels * (logits / temp)))
        for temp in temperatures
    ]
    return float(temperatures[int(np.argmin(losses))])


def probabilities(logits, temperature):
    z = np.asarray(logits, dtype=np.float64) / float(temperature)
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))


def episode_cluster_bootstrap(labels, probs, episode_ids, *, n_boot=1000, seed=1234):
    """Percentile intervals resampling whole episodes, never individual timesteps."""
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    p = np.asarray(probs, dtype=np.float64).reshape(-1)
    groups = np.asarray(episode_ids).reshape(-1)
    if y.size == 0 or y.size != p.size or y.size != groups.size:
        raise ValueError("labels, probabilities and episode_ids must be non-empty and aligned")
    if n_boot < 100:
        raise ValueError("n_boot must be at least 100")
    unique = np.unique(groups)
    if unique.size < 2:
        raise ValueError("at least two independent episodes are required for cluster bootstrap")
    indices = [np.flatnonzero(groups == episode) for episode in unique]
    rng = np.random.default_rng(seed)
    samples = {key: [] for key in ("auroc", "auprc", "brier", "ece")}
    for _ in range(n_boot):
        selected = rng.integers(0, len(indices), size=len(indices))
        row_idx = np.concatenate([indices[i] for i in selected])
        if np.unique(y[row_idx]).size < 2:
            continue
        metrics = binary_metrics(y[row_idx], p[row_idx])
        for key in samples:
            value = float(metrics[key])
            if np.isfinite(value):
                samples[key].append(value)
    intervals = {}
    for key, values in samples.items():
        if len(values) < max(50, n_boot // 2):
            raise ValueError(f"Insufficient valid cluster bootstrap replicates for {key}")
        intervals[key] = [float(v) for v in np.quantile(values, [0.025, 0.975])]
    return {"confidence": 0.95, "unit": "episode", "episodes": int(unique.size),
            "requested_replicates": int(n_boot),
            "valid_replicates": min(len(v) for v in samples.values()),
            "intervals": intervals}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-data", required=True)
    p.add_argument("--validation-data", required=True,
                   help="Model-selection split; not used for temperature fitting or final metrics.")
    p.add_argument("--calibration-data", required=True,
                   help="Disjoint split used only to fit scalar temperature.")
    p.add_argument("--test-data", required=True,
                   help="Disjoint final risk-estimator evaluation split.")
    p.add_argument("--output", required=True)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive")
    torch.manual_seed(args.seed)

    train, names, train_seeds = load_dataset(args.train_data)
    val, _, val_seeds = load_dataset(args.validation_data, names, train["observations"].shape[1])
    calibration, _, calibration_seeds = load_dataset(
        args.calibration_data, names, train["observations"].shape[1]
    )
    test, _, test_seeds = load_dataset(args.test_data, names, train["observations"].shape[1])
    assert_disjoint_splits([
        ("train", train_seeds), ("validation", val_seeds),
        ("calibration", calibration_seeds), ("test", test_seeds),
    ])
    if np.unique(train["labels"]).size != 2 or np.unique(val["labels"]).size != 2:
        raise ValueError("Train and validation data must each contain both classes")
    if np.unique(calibration["labels"]).size != 2 or np.unique(test["labels"]).size != 2:
        raise ValueError("Calibration and final test data must each contain both classes")

    X = torch.tensor(train["observations"], dtype=torch.float32, device=args.device)
    A = torch.tensor(train["agent_ids"], dtype=torch.long, device=args.device)
    Y = torch.tensor(train["labels"], dtype=torch.float32, device=args.device)
    Xv = torch.tensor(val["observations"], dtype=torch.float32, device=args.device)
    Av = torch.tensor(val["agent_ids"], dtype=torch.long, device=args.device)
    Yv = torch.tensor(val["labels"], dtype=torch.float32, device=args.device)
    Xc = torch.tensor(calibration["observations"], dtype=torch.float32, device=args.device)
    Ac = torch.tensor(calibration["agent_ids"], dtype=torch.long, device=args.device)
    Xt = torch.tensor(test["observations"], dtype=torch.float32, device=args.device)
    At = torch.tensor(test["agent_ids"], dtype=torch.long, device=args.device)
    Yt = test["labels"].astype(np.int64)

    positives = float(Y.sum().item())
    negatives = float(Y.numel() - positives)
    if positives == 0 or negatives == 0:
        raise ValueError("Training data must contain both positive and negative labels")
    model = RiskEstimatorNet(X.shape[1], len(names)).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([negatives / positives], device=args.device)
    )
    best_loss, best_state, stale = float("inf"), None, 0
    for _ in range(args.epochs):
        model.train()
        order = torch.randperm(X.shape[0], device=args.device)
        for start in range(0, X.shape[0], args.batch_size):
            idx = order[start:start + args.batch_size]
            loss = loss_fn(model(X[idx], A[idx]), Y[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(Xv, Av), Yv).item())
        if val_loss < best_loss:
            best_loss, stale = val_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= 8:
            break
    if best_state is None:
        raise RuntimeError("No valid model checkpoint selected")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        calibration_logits = model(Xc, Ac).cpu().numpy()
        test_logits = model(Xt, At).cpu().numpy()
    temperature = fit_temperature(calibration_logits, calibration["labels"])
    test_probs = probabilities(test_logits, temperature)
    test_metrics = binary_metrics(Yt, test_probs)
    test_cluster_ci = episode_cluster_bootstrap(
        Yt, test_probs, test["episode_ids"], n_boot=args.bootstrap, seed=args.seed
    )

    checkpoint = {
        "state_dict": model.state_dict(),
        "observation_dim": int(X.shape[1]),
        "hidden_dim": 128,
        "agent_names": names,
        "temperature": temperature,
        "validation_bce": best_loss,
        "calibration_temperature_fit": "disjoint calibration split",
        "test_metrics": test_metrics,
        "test_cluster_bootstrap_95ci": test_cluster_ci,
        "train_episode_ids": sorted(train_seeds),
        "validation_episode_ids": sorted(val_seeds),
        "calibration_episode_ids": sorted(calibration_seeds),
        "test_episode_ids": sorted(test_seeds),
        "seed": args.seed,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    report = {
        "output": args.output, "temperature": temperature,
        "best_validation_bce": best_loss, "calibration_samples": int(len(calibration["labels"])),
        "test_samples": int(len(Yt)), "test_positive_rate": float(np.mean(Yt)),
        "test_metrics": test_metrics, "test_cluster_bootstrap_95ci": test_cluster_ci,
        "split_seeds": {"train": sorted(train_seeds), "validation": sorted(val_seeds),
                        "calibration": sorted(calibration_seeds), "test": sorted(test_seeds)},
    }
    Path(args.output).with_suffix(".metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
