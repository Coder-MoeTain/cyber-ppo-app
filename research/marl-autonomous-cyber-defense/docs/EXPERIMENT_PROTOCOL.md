# Preregistered Experimental Protocol (draft; freeze before benchmark runs)

## Benchmark and source control

Use the exact CAGE Challenge 4 source revision in `configs/cc4.lock.yaml`.
Record the commit, CybORG version, Python/dependency lock, OS, hardware,
scenario parameters and all random seeds with every run. Do not silently
upgrade the simulator during an experiment series.

## Research questions

1. Does RA-HMAPPO improve team utility and security/availability outcomes
   relative to a strong heuristic, Independent PPO and MAPPO?
2. Does it generalize to held-out network sizes and host configurations?
3. Does risk conditioning improve prioritization without excessive
   containment or service disruption?

## Baselines

- Strong mask-aware rule-based heuristic; document it as a reimplementation
  unless original competition source is available.
- Independent PPO with decentralized actor/critic per agent.
- MAPPO with decentralized actors and centralized critic.
- Proposed RA-HMAPPO with learned manager and worker policies.

Match environment interactions, training scenario distributions, action masks,
model-selection budget and evaluation seeds. All methods currently send the
wrapper's default empty messages; this study does not evaluate learned
inter-agent communication. Report compute and wall-clock budgets for every method.

## Scenario matrix

- Standard CC4 scenario distribution.
- Constant network size.
- Improved decoy detection.
- Increased phishing.
- Stealthy Red.
- Aggressive Red.
- Additional held-out host-count/topology configurations, if supported by the
  pinned simulator and defined before training.

The challenge perturbations above are based on the published CC4
generalization analysis. Preserve exact parameter changes in versioned
config files rather than relying on prose descriptions.

## Splits and sample sizes

Create disjoint training, validation and final-test scenario seeds/configs.
Use at least 10 independent training seeds where feasible. For final
evaluation target 100 episodes per method and scenario, with episode-level
records and paired seed/scenario comparisons. The official CC4 evaluation
protocol also reports 100 randomized episodes of 500 steps; match it when
comparing challenge-style aggregate reward, and separately report any
different episode horizon used in research experiments.

Do not tune hyperparameters or choose checkpoints on final-test scenarios.
Freeze the primary endpoint, secondary endpoints, statistical tests and
exclusion rules before final evaluation.

## Outcomes

Primary endpoint: cumulative team reward, with the team reward aggregation
rule explicitly defined from the five per-agent reward streams.

Secondary outcomes:
- per-agent cumulative reward and variance;
- critical-zone and mission-phase reward;
- service availability / green-user disruption;
- compromised-host or red-impact outcomes where observable without leaking
  privileged state to the acting policy;
- valid/invalid action rate;
- detection, analysis, removal and restoration action counts and latency;
- unnecessary containment/restoration and false-positive response cost;
- inference time, training time and model size.

Separate training-only privileged labels from actor-visible inputs. If a
risk estimator is supervised using simulator ground truth, disclose that
privilege boundary and ensure evaluation actors receive only permitted
observations.

## Statistical analysis

Report means, standard deviations, medians where distributions are skewed,
95% confidence intervals, paired effect sizes, and paired tests where the
design is genuinely paired. Use bootstrap intervals over independent
evaluation units and Holm-Bonferroni correction for planned multiple
comparisons. Do not treat correlated timesteps within an episode as
independent samples. Publish raw per-episode outcomes and analysis scripts.

## Ablations

- Remove the learned manager (flat MAPPO).
- Remove risk input.
- Replace learned hierarchy with deterministic task allocation.
- Remove risk-aware reward shaping.
- Replace action masks with unmasked policy only as a diagnostic, not as a
  deployable baseline.
- Vary risk estimator calibration and uncertainty handling.

## Reproducibility artifacts

Archive all configs, exact source commits, seeds, episode-level JSONL/CSV,
checkpoints, training curves, test reports, statistical outputs and scripts.
A final paper must distinguish mock tests, simulator smoke tests, and
full benchmark experiments.
