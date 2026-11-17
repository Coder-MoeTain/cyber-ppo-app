# Published H-MARL baseline integration

This project now includes an optional runner for the published CAGE-4 H-MARL checkpoint. It is intentionally isolated from the main benchmark environment because the upstream repository specifies Ray/RLlib 2.10, while this project's pinned CC4 validation environment uses Ray 2.9.3.

## Pinned upstream source

- Repository: https://github.com/adityavs14/Hierarchical-MARL
- Commit: `6fe960f5931f2d7dc5ef522f09ca4a179f377ee4`
- License: MIT (retain the upstream copyright/license when redistributing any source)
- Paper: Singh et al., “Hierarchical Multi-agent Reinforcement Learning for Cyber Network Defense,” arXiv:2410.17351.
- Variants: `meta` (learned master policy) and `expert` (expert rule for master task selection).

The upstream README states that the published saved policies are provided under `h-marl-3policy/saved_policies`. The runner verifies the exact upstream commit before loading those policies.

## Separate environment

Do not install Ray 2.10 into the main research environment. Create an isolated environment and install the upstream H-MARL requirements, its compatible CAGE-4 CybORG checkout, and Ray/RLlib 2.10 as required by the upstream README.

Example checkout:

```bash
mkdir -p vendor
git clone https://github.com/adityavs14/Hierarchical-MARL vendor/Hierarchical-MARL
git -C vendor/Hierarchical-MARL checkout 6fe960f5931f2d7dc5ef522f09ca4a179f377ee4
```

Use an isolated interpreter in `HMARL_PYTHON` that can import the upstream CybORG and Ray dependencies. Run from this research directory:

```bash
HMARL_PYTHON=/path/to/hmarl-venv/bin/python
"$HMARL_PYTHON" scripts/run_hmarl_baseline.py \
  --hmarl-root vendor/Hierarchical-MARL \
  --variant meta \
  --seeds $(seq 501 600) \
  --output results/standard_cc4/hmarl_meta.jsonl
```

The runner evaluates one deterministic first episode per seed, writes a JSONL record and a captured run log per seed, and records the exact upstream commit in its metadata. The one-episode standard-deviation edge case in the upstream evaluation function is handled in memory; no upstream source files are edited.

## Reward conversion and fairness caveat

The upstream evaluation sums the per-step mean of the Blue-agent rewards. This runner multiplies that episode return by five to match this project's primary endpoint (the sum of five Blue-agent reward streams). The converted value and original upstream value are both retained in each record.

**This is not an information-matched comparison.** The published H-MARL wrapper adds malicious-file indicators, messages/communication and custom task-specific observation/action handling. It should be reported as a prior published hierarchical baseline with a feature/interface advantage, not as a strictly controlled ablation. The main matrix's IPPO/MAPPO/RA-HMAPPO variants remain on this project's common BlueFlatWrapper interface.

## Statistical comparison

After producing `hmarl_meta.jsonl`, rerun the paired analysis with the optional H-MARL input:

```bash
python scripts/analyze_cc4_results.py \
  --ra-glob "results/standard_cc4/ra_hmappo_seed*.jsonl" \
  --ippo-glob "results/standard_cc4/ippo_seed*.jsonl" \
  --mappo-glob "results/standard_cc4/mappo_seed*.jsonl" \
  --heuristic-file results/standard_cc4/heuristic.jsonl \
  --alert-only-file results/standard_cc4/alert_only.jsonl \
  --hmarl-file results/standard_cc4/hmarl_meta.jsonl \
  --output results/standard_cc4/paired_analysis_with_hmarl.json
```

The analysis requires the same 100 evaluation seeds and applies Holm correction to the additional comparison. Do not pool the H-MARL result into the main matched-information baseline table without clearly marking the wrapper difference.
