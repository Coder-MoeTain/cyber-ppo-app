# Independent PPO baseline

`IndependentPPOTrainer` is a decentralized baseline with a separate local
actor and local value function for every agent. Each agent optimizes its own
reward stream; there is no centralized critic, shared policy, or learned
communication. It supports action masks and GAE with separate terminal and
episode-boundary flags.

Run the interface smoke test through pytest. The real CC4 baseline should be
trained and evaluated with the pinned simulator, matched scenario seeds,
interaction budgets and held-out evaluation protocol. Mock tests validate
software behavior only.
