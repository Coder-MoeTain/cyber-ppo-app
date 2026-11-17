"""Run a deterministic *mock-only* MARL training smoke test."""
import argparse
import json
from autocd.algorithms.trainer import TrainConfig, train_mock


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    _, returns = train_mock(TrainConfig(episodes=args.episodes, seed=args.seed))
    print(json.dumps({
        "environment": "mock_enterprise_NOT_CybORG",
        "episodes": len(returns),
        "seed": args.seed,
        "mean_episode_return": sum(returns) / len(returns),
        "episode_returns": returns,
    }, indent=2))


if __name__ == "__main__":
    main()
