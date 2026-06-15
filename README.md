# Inventory Replenishment RL

A course project for training and evaluating a shared tabular Q-learning policy on a small, simulated five-product retail inventory replenishment problem.

## Current Status

The repository scaffold, five-product MDP design, and deterministic simulator are approved and complete. Random and rule-based baseline policies are implemented and ready for review. The finalized reward and Q-learning agent have not yet been implemented.

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
make smoke
```

`make simulator-evidence` generates deterministic simulator diagnostics. `make baseline-evidence` evaluates the random-feasible and rule-based order-up-to policies over identical environment seeds. `make smoke` runs the current checks and regenerates both evidence sets. No Q-learning agent has been implemented.

## Step 4 Baselines

- `random_feasible` uniformly samples from the environment's feasible-action mask.
- `rule_order_up_to` orders toward two weeks of expected demand using only the current product, demand regime, on-hand inventory, and incoming order.

The order-up-to rule is a credible operational benchmark because it responds to product demand profiles and inventory position without seeing future demand. Both policies must pass the same environment-enforced action and capacity constraints. Products do not compete for a shared budget in the approved MDP.
