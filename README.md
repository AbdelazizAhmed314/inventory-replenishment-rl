# Inventory Replenishment RL

This project trains and evaluates a shared tabular Q-learning policy for weekly
replenishment of five simulated retail products. The policy decides whether to order
`0`, `5`, `10`, or `15` units for each product while balancing sales, procurement,
holding, stockout, and terminal-excess costs.

## Decision Model

Each 12-week episode contains 60 product-level transitions. A state contains product ID,
week, exact on-hand inventory, incoming order, and observable demand regime. Products have
different demand profiles, but share one policy and do not compete for a shared budget.

Hard feasibility is enforced by the simulator:

```text
on_hand + incoming_order + order_quantity <= 30
```

The scalar reward is:

```text
10 * sales - 2 * orders - 1 * holding - 15 * stockouts - 5 * terminal excess
```

Raw operational metrics are always reported separately from reward.

## Main Workflows

Prerequisites: Python `3.12`, `uv`, Git, and Make.

```bash
make sync
make smoke
make run
make verify
make final-audit
```

- `make smoke` runs fast tests and regenerates pre-training diagnostics.
- `make run` trains the Q-table, evaluates the frozen policy, and runs a seeded CLI demo.
- `make verify` runs all quality gates and validates the generated artifact contracts.
- `make final-audit` additionally checks repository hygiene and final deliverables.
- `make demo` compares the saved learned policy with the rule baseline on seed `220000`.

Training takes approximately `8.9` minutes on the development laptop. Evaluation and
verification take under one minute after training. Generated outputs are intentionally
ignored by Git and are recreated by the commands above.

## Policies

- `q_learning_frozen`: greedy decisions from the saved Q-table, with no learning during
  evaluation.
- `rule_order_up_to`: orders toward two weeks of expected demand using only the current
  observable state.
- `random_feasible`: uniformly samples from the feasible-action mask.

The dense Q-table has `89,280` nominal cells. Training runs for `50,000` episodes with
`alpha=0.10`, `gamma=0.99`, and linear epsilon decay from `1.00` to `0.05`.

## Main Results

Formal evaluation uses 200 unseen standard seeds and identical demand realizations for all
policies.

| Policy | Mean Reward | Reward Std. Dev. | Stockout Units | Service Level | Avg. Holding | Terminal Excess |
|---|---:|---:|---:|---:|---:|---:|
| Frozen Q-learning | `2,221.11` | `275.16` | `21.13` | `94.65%` | `6.72` | `17.67` |
| Rule order-up-to | `1,840.95` | `268.25` | `35.70` | `90.84%` | `4.05` | `46.94` |
| Random feasible | `576.17` | `516.29` | `65.32` | `83.47%` | `10.30` | `76.79` |

The learned policy beats the rule baseline on 196 of 200 standard seeds, raises mean reward
by `380.16`, and records zero capacity violations. It also carries `65.7%` more average
inventory than the rule. Under a demand-drop scenario it loses to the rule on all 50 seeds;
under high holding costs it wins only 21 of 50 seeds.

The recommendation is **shadow mode**, not autonomous deployment. See
`FAILURE_AND_SAFETY_ANALYSIS.md` and `BUSINESS_MEMO.md`.

## Evidence and Reports

Generated locally by `make run`:

- `artifacts/q_table.csv`
- `artifacts/training_manifest.json`
- `artifacts/evaluation_manifest.json`
- `evidence/training_history.csv`
- `evidence/policy_evaluation_results.csv`
- `evidence/scenario_evaluation_results.csv`
- `evidence/reward_learning_curve.png`
- `evidence/policy_comparison.png`
- `evidence/scenario_comparison.png`

Tracked reports:

- `TRAINING_REPORT.md`
- `EVALUATION_REPORT.md`
- `REWARD_HACKING_AUDIT.md`
- `FAILURE_AND_SAFETY_ANALYSIS.md`
- `BUSINESS_MEMO.md`
- `SUBMISSION_CHECKLIST.md`

## Repository Layout

```text
config/       MDP and Q-learning configuration
src/          Simulator, policies, reward, training, and evaluation code
scripts/      Reproducible generation, demo, verification, and audit commands
tests/        Unit and integration tests
artifacts/    Generated trained-policy files
evidence/     Generated tables, reports, and plots
```
