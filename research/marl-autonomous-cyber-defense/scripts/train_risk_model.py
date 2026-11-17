"""Train and calibrate the CC4 local-observation risk estimator."""
import argparse
import json
import numpy as np
import torch
from torch import nn

from autocd.risk_model import RiskEstimatorNet, binary_metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-data", required=True)
    p.add_argument("--validation-data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    train = np.load(args.train_data, allow_pickle=False)
    val = np.load(args.validation_data, allow_pickle=False)
    names = train["agent_names"].tolist()
    if names != val["agent_names"].tolist():
        raise ValueError("Train and validation agent order mismatch")
    if set(train["episode_ids"].tolist()) & set(val["episode_ids"].tolist()):
        raise ValueError("Train and validation episode seeds overlap")

    X = torch.tensor(train["observations"], dtype=torch.float32, device=args.device)
    A = torch.tensor(train["agent_ids"], dtype=torch.long, device=args.device)
    Y = torch.tensor(train["labels"], dtype=torch.float32, device=args.device)
    Xv = torch.tensor(val["observations"], dtype=torch.float32, device=args.device)
    Av = torch.tensor(val["agent_ids"], dtype=torch.long, device=args.device)
    Yv = torch.tensor(val["labels"], dtype=torch.float32, device=args.device)
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
    for epoch in range(args.epochs):
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

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(Xv, Av).cpu().numpy()
    yv = val["labels"].astype(np.int64)
    temperatures = np.geomspace(0.25, 4.0, 81)
    def bce(temp):
        z = logits / temp
        return float(np.mean(np.logaddexp(0.0, z) - yv * z))
    temperature = float(min(temperatures, key=bce))
    probs = 1.0 / (1.0 + np.exp(-logits / temperature))
    metrics = binary_metrics(yv, probs)
    checkpoint = {
        "state_dict": model.state_dict(),
        "observation_dim": int(X.shape[1]),
        "hidden_dim": 128,
        "agent_names": names,
        "temperature": temperature,
        "validation_metrics": metrics,
        "train_episode_ids": train["episode_ids"].tolist(),
        "validation_episode_ids": val["episode_ids"].tolist(),
        "seed": args.seed,
    }
    torch.save(checkpoint, args.output)
    print(json.dumps({"output": args.output, "temperature": temperature,
                      "best_validation_bce": best_loss, "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
