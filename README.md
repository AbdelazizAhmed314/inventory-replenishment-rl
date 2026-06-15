# Inventory Replenishment RL

A course project for training and evaluating a shared tabular Q-learning policy on a small, simulated five-product retail inventory replenishment problem.

## Current Status

The repository scaffold, five-product MDP design, deterministic simulator, and baseline policies are approved and complete. The scalar reward and pre-training reward-hacking audit are implemented and ready for review. No Q-learning agent has been implemented.

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
make smoke
```

`make simulator-evidence` generates deterministic simulator diagnostics. `make baseline-evidence` evaluates the random-feasible and rule-based order-up-to policies over identical environment seeds. `make reward-audit` checks the finalized reward against controlled edge cases and extreme policies. `make smoke` runs all current checks and regenerates the evidence. No Q-learning agent has been implemented.

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
