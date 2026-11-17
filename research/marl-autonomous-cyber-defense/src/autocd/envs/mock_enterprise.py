import numpy as np

class MockEnterpriseEnv:
    """Synthetic interface-test environment; not a CybORG benchmark."""
    possible_agents = ["detection", "risk", "response"]
    action_names = ["observe", "investigate", "contain", "verify", "wait"]

    def __init__(self, n_hosts=8, episode_length=40, seed=42):
        if n_hosts < 2 or episode_length < 1:
            raise ValueError("n_hosts >= 2 and episode_length >= 1 required")
        self.n_hosts, self.episode_length = n_hosts, episode_length
        self.rng = np.random.default_rng(seed)
        self.agents = self.possible_agents.copy()
        self.t = 0
        self.compromised = np.zeros(n_hosts, dtype=np.float32)
        self.alerts = np.zeros(n_hosts, dtype=np.float32)
        self.criticality = np.linspace(.2, 1., n_hosts, dtype=np.float32)
        self.available = np.ones(n_hosts, dtype=np.float32)

    @property
    def observation_dim(self): return 16
    @property
    def action_dim(self): return len(self.action_names)

    def _obs(self, agent):
        summary = np.array([self.t/self.episode_length, self.alerts.mean(),
            self.compromised.mean(), self.available.mean(), self.alerts.max(),
            (self.criticality*self.compromised).max()], dtype=np.float32)
        features = np.concatenate([self.alerts[:5], self.available[:5],
                                   self.criticality[:5]]).astype(np.float32)
        return np.concatenate([summary, features])[:self.observation_dim]

    def reset(self, seed=None):
        if seed is not None: self.rng = np.random.default_rng(seed)
        self.t = 0
        self.compromised.fill(0); self.alerts.fill(0); self.available.fill(1)
        self.agents = self.possible_agents.copy()
        return ({a: self._obs(a) for a in self.agents},
                {a: {"seed": seed} for a in self.agents})

    def step(self, actions):
        if set(actions) != set(self.agents):
            raise ValueError("Provide one action for every active agent")
        self.t += 1
        if self.rng.random() < .22:
            h = int(self.rng.integers(0, self.n_hosts))
            self.alerts[h] = min(1., self.alerts[h] + .35)
        if self.rng.random() < .08:
            self.compromised[int(self.rng.integers(0, self.n_hosts))] = 1.
        disruption = 0.
        for agent, action in actions.items():
            action = int(action)
            if not 0 <= action < self.action_dim: raise ValueError("Invalid action")
            if agent == "detection" and action == 1:
                self.alerts = np.clip(self.alerts + .03, 0, 1)
            elif agent == "response" and action == 2:
                candidates = np.where((self.alerts > .25) | (self.compromised > 0))[0]
                if len(candidates):
                    h = int(candidates[np.argmax(self.criticality[candidates])])
                    self.compromised[h] = 0.; self.alerts[h] = 0.
                    self.available[h] = .95; disruption += .05
        residual = float(np.sum(self.compromised*self.criticality))
        reward = 1. - residual - disruption - .25*float(self.alerts.mean())
        availability = float(self.available.mean())
        obs = {a: self._obs(a) for a in self.agents}
        rewards = {a: float(reward) for a in self.agents}
        terminated = self.t >= self.episode_length
        terms = {a: terminated for a in self.agents}
        truncs = {a: False for a in self.agents}
        infos = {a: {"availability": availability, "weighted_compromise": residual}
                 for a in self.agents}
        if terminated: self.agents = []
        return obs, rewards, terms, truncs, infos

    def close(self): self.agents = []
