# Tabular Q-Learning Training Report

## Status

The shared tabular Q-learning agent is implemented, trained, saved, and formally evaluated.

## Training Configuration

| Setting | Value | Rationale |
|---|---:|---|
| Episodes | `50,000` | Repeated coverage within the laptop runtime limit. |
| Learning rate (`alpha`) | `0.10` | Gradual updates under stochastic demand. |
| Discount factor (`gamma`) | `0.99` | Values delayed stockout, holding, and terminal effects. |
| Epsilon | `1.00` to `0.05` | Broad early exploration with limited retained exploration. |
| Epsilon decay | Linear over first `80%` | Leaves the final `20%` at minimum exploration. |
| Agent seed | `697` | Reproducible exploration and tie-breaking. |
| Environment seeds | `100000` to `149999` | Training-only range, disjoint from evaluation. |
| Initial Q-value | `0.0` | Neutral and transparent initialization. |
| Rolling window | `500` episodes | Smooths stochastic training metrics. |

## Implementation Checks

- Q-table shape: `5 x 12 x 31 x 4 x 3 x 4`, or `89,280` cells.
- Exploration, greedy selection, and bootstrapping consider only feasible actions.
- Terminal transitions use no future bootstrap value.
- Tests verify exact temporal-difference arithmetic, epsilon decay, deterministic reruns,
  and Q-table save/load.
- The artifact verifier checks row counts, hashes, epsilon endpoints, monotonic visitation,
  and zero visits to infeasible actions.

## Training Results

The primary run completed in `534.19` seconds, approximately `8.9` minutes. A fresh-clone
reproduction completed training in `700.17` seconds, approximately `11.7` minutes.

| Metric | Initial 500-Episode Mean | Final 500-Episode Mean |
|---|---:|---:|
| Episode reward | `599.05` | `2,117.84` |
| Stockout units | `64.30` | `23.57` |
| Holding units | `616.47` | `416.48` |

The agent visited `39,880` of `46,080` feasible state-action cells. No infeasible action
cell was visited. Representative terminal-week states generally select zero, which is
consistent with the terminal cost structure.

Training metrics establish learning progress, not generalization. The formal evaluation in
`EVALUATION_REPORT.md` freezes the policy and uses unseen seeds.

## Generated Outputs

- `artifacts/q_table.csv`
- `artifacts/training_run_config.json`
- `artifacts/training_manifest.json`
- `evidence/training_history.csv`
- `evidence/reward_learning_curve.png`
- `evidence/learned_policy_snapshot.csv`

Run `make verify-training` to validate the saved training artifact contract.
