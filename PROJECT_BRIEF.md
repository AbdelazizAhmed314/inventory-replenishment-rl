# Project Brief: Single-Product Inventory Replenishment Agent

## Decision Status

This brief defines the proposed Step 1 scope for review. The detailed state, action, transition, horizon, and reward specifications remain intentionally open until Step 2 and Step 5.

## Problem Statement

A retail inventory manager must repeatedly decide how much of one product to reorder without knowing future demand exactly. Ordering too little can create stockouts and lost sales. Ordering too much ties up cash and creates holding or waste costs.

We will build a small local simulator and train a tabular Q-learning agent to learn a replenishment policy from repeated simulated experience. The learned policy will be compared with random and rule-based baselines before any deployment recommendation is made.

## Why This Is a Sequential Decision Problem

Each ordering decision changes the inventory available in later periods. That means an action can look expensive today but prevent a future stockout, or look profitable today but create future excess inventory. The policy must therefore consider long-term consequences rather than optimize one isolated decision.

The project is deliberately limited to one product and discrete decisions so the learning problem remains understandable, testable, and laptop-friendly.

## Business Decision-Maker

The primary decision-maker is a retail inventory or supply-chain manager responsible for setting replenishment policy and deciding whether a learned policy is safe enough to test in shadow mode.

## Affected Stakeholders

- Customers, who may experience unavailable products when the policy under-orders.
- Store and warehouse staff, who receive, store, and manage ordered inventory.
- Finance and operations teams, which are affected by tied-up cash, lost sales, and waste.
- Business leadership, which owns the final deployment decision and risk tolerance.

## Goals

- Frame one bounded retail replenishment problem as a sequential decision problem.
- Build a deterministic and seeded local simulator for safe experimentation.
- Implement a transparent tabular Q-learning policy.
- Compare the learned policy fairly against random and rule-based baselines.
- Evaluate both financial performance and operational failure modes.
- Produce reproducible artifacts, tests, plots, and a governance recommendation.
- Keep core training under the assignment's 10–15 minute laptop limit.

## Non-Goals

- Managing multiple products, stores, warehouses, suppliers, or interacting agents.
- Training or fine-tuning an LLM.
- Using real customers, production inventory systems, or live business data.
- Claiming that simulator performance proves production readiness.
- Optimizing a continuous order quantity in the core tabular implementation.
- Making DQN or PPO part of the required core submission.
- Finalizing the MDP or reward equation during Step 1.

## Initial Assumptions for Later Review

These assumptions keep the initial problem bounded. They may be revised when the MDP is specified.

- One product is replenished over repeated fixed-length episodes.
- Decisions occur at regular simulated intervals.
- Order choices will be discrete and small enough for a Q-table.
- Demand will be generated locally using seeded, bounded randomness.
- The simulator will represent delayed consequences such as replenishment timing and future stock availability.
- Hard operational constraints will be enforced by the simulator, not left for the agent to discover through failure.
- Training, validation, and evaluation scenarios will use separate random seeds.

## Proposed Success Criteria

The final success thresholds will be approved after the MDP, baselines, and metrics are defined. At minimum, the project will be considered technically successful only if:

- The full core workflow is reproducible from a clean checkout.
- Simulator behavior and policy logic are covered by automated tests.
- The learned policy is evaluated on unseen seeded episodes.
- The learned policy is compared against both random and credible rule-based baselines.
- The evaluation reports reward, stockouts, excess inventory, and other approved business metrics.
- The project identifies edge scenarios where the learned policy performs poorly.
- Training completes within 10–15 minutes on a normal laptop.
- The final recommendation does not overstate what offline simulator results prove.

Outperforming the rule-based baseline is a target, not a result that will be assumed in advance. A well-supported rejection of the learned policy remains an acceptable business conclusion.

## Repository and Reproducibility Strategy

- Python version: 3.12
- Dependency management: `uv` with committed `uv.lock`
- Package layout: `src/inventory_replenishment_rl/`
- Tests: `tests/`
- Documentation: `docs/`
- Generated run outputs: `artifacts/`
- Supporting review evidence: `evidence/`
- Validation helpers: `scripts/`
- Standard commands: `make sync`, `make test`, `make check`, and `make smoke`
- Version control: standalone Git repository in `Assignments/Assignment2`
- GitHub remote: deferred until the local scaffold is approved

## Course Sources Used

### `Lec-4-5-deep-reinforcement-learning-agentic-ai-deckset.pdf`

The lecture deck supports inventory as a useful sequential domain:

> "Inventory is where delayed reward becomes concrete. An over-order today becomes a markdown tomorrow."

It also supports keeping the core method simple:

> "Default to the simplest algorithm that matches the action space and the operational risk you can absorb."

### `learning-agents-showcase`

The showcase describes its reproducible core as:

> "The deterministic core is laptop- and CI-friendly."

It also provides the project-structure and clean-checkout pattern adopted by this repository.

### `adaptive-course-assistant-rl-showcase`

The showcase establishes a useful scope boundary:

> "There is one learned controller in the simulator, not multiple learning agents with coordination or competition."

It also warns against overclaiming:

> "Better simulator reward is not deployment readiness."

## Step 1 Approval Questions

1. Should the project proceed with single-product retail inventory replenishment?
2. Should tabular Q-learning remain the approved core method?
3. Are the stakeholders, goals, non-goals, and assumptions appropriately bounded?
4. Are the proposed success criteria sufficient before Step 2 defines the MDP?
5. Is the repository and reproducibility strategy acceptable?
