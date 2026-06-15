# Inventory Replenishment RL

A course project for training and evaluating a shared tabular Q-learning policy on a small, simulated five-product retail inventory replenishment problem.

## Current Status

The repository scaffold and five-product MDP configuration are complete. The simulator, finalized reward, policies, and evaluation pipeline have not yet been implemented.

## Setup

```bash
make sync
```

## Current Quality Gates

```bash
make test
make check
make validate-design
make smoke
```

These commands currently validate the repository scaffold and Step 2 MDP configuration. Later roadmap steps will extend them to cover the simulator, policies, training, evaluation, and artifact verification.
