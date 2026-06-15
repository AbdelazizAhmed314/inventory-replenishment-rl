# Inventory Replenishment RL

A course project for training and evaluating a shared tabular Q-learning policy on a small, simulated five-product retail inventory replenishment problem.

## Current Status

The repository scaffold and approved five-product MDP design are complete. The deterministic simulator is implemented and ready for review. The finalized reward, policies, and evaluation pipeline have not yet been implemented.

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
make smoke
```

`make simulator-evidence` generates deterministic sample episodes and a validation report using explicit diagnostic schedules. `make smoke` runs the current checks and regenerates this evidence. Neither command implements a baseline or learning policy. Later roadmap steps will extend the project to cover policies, training, evaluation, and artifact verification.
