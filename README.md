# Inventory Replenishment RL

A course project for training and evaluating a shared tabular Q-learning policy on a small, simulated five-product retail inventory replenishment problem.

## Current Status

Step 1 is approved and complete. The repository scaffold and five-product scope are defined, but the MDP, simulator, reward, policies, and evaluation pipeline have not yet been implemented.

Read:

- [`PROJECT_BRIEF.md`](PROJECT_BRIEF.md) for the proposed business problem and boundaries.
- [`IMPLEMENTATION_ROADMAP.md`](IMPLEMENTATION_ROADMAP.md) for the reviewed step-by-step workflow.
- [`Instructions.md`](Instructions.md) for the assignment requirements.

## Setup

```bash
make sync
```

## Current Quality Gates

```bash
make test
make check
make smoke
```

These commands currently validate only the Step 1 scaffold. Later roadmap steps will extend them to cover the simulator, policies, training, evaluation, and artifact verification.
