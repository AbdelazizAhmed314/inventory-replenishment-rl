# Inventory Replenishment RL

A course project for training and evaluating a shared tabular Q-learning policy on a small, simulated five-product retail inventory replenishment problem.

## Current Status

The repository scaffold, MDP, simulator, baselines, and reward design are approved and complete. The tabular Q-learning agent has been trained and is ready for review before formal evaluation.

## Setup

```bash
make sync
```

## Current Quality Gates

```bash
make test
make check
make validate-design
make simulator-evidence
make baseline-evidence
make reward-audit
make train
make verify-training
make smoke
```

`make simulator-evidence` generates deterministic simulator diagnostics. `make baseline-evidence` evaluates the non-learning baselines. `make reward-audit` checks the finalized reward against controlled edge cases. `make train` trains the Q-learning agent and saves its local artifacts and evidence. `make verify-training` validates the saved training outputs. `make smoke` runs the fast pre-training checks without retraining.

## Step 4 Baselines

- `random_feasible` uniformly samples from the environment's feasible-action mask.
- `rule_order_up_to` orders toward two weeks of expected demand using only the current product, demand regime, on-hand inventory, and incoming order.

The order-up-to rule is a credible operational benchmark because it responds to product demand profiles and inventory position without seeing future demand. Both policies must pass the same environment-enforced action and capacity constraints. Products do not compete for a shared budget in the approved MDP.

## Step 5 Reward

The transparent scalar reward, in simulated monetary units, is:

```text
10 * sales - 2 * orders - 1 * holding - 15 * stockouts - 5 * terminal excess
```

Raw business metrics remain separate from the scalar training signal. See `REWARD_HACKING_AUDIT.md` for the pre-training audit, rejected alternatives, and residual risks.

## Step 6 Training

The dense Q-table has `89,280` cells. Training runs for `50,000` episodes with `alpha=0.10`, `gamma=0.99`, and linear epsilon decay from `1.00` to `0.05`. The final run completed in approximately `8.9` minutes.

See `TRAINING_REPORT.md` for the course basis, implementation checks, results, representative learned decisions, and remaining risks. Formal comparison on unseen seeds has not been performed.
