# Project Brief: Five-Product Inventory Replenishment Agent

## Decision Status

Step 1 was approved with one scope modification: the simulator will manage five products instead of one. The detailed state, action, transition, horizon, and reward specifications remain intentionally open until Step 2 and Step 5.

## Problem Statement

A retail inventory manager must repeatedly decide how much of five products to reorder without knowing future demand exactly. Ordering too little can create stockouts and lost sales. Ordering too much ties up cash and creates holding or waste costs.

We will build a small local simulator and train one shared tabular Q-learning policy from repeated simulated experience. At each decision point, the policy will make a replenishment decision for one product using that product's current situation and profile. The learned policy will be compared with random and rule-based baselines before any deployment recommendation is made.

## Why This Is a Sequential Decision Problem

Each ordering decision changes the inventory available for that product in later periods. That means an action can look expensive today but prevent a future stockout, or look profitable today but create future excess inventory. The policy must therefore consider long-term consequences rather than optimize one isolated decision.

The project is deliberately limited to five products and discrete product-level decisions. It will not use a joint action that chooses all five order quantities simultaneously. This keeps the Q-table understandable, testable, and laptop-friendly while allowing evaluation across products with different demand and cost profiles.

## Business Decision-Maker

The primary decision-maker is a retail inventory or supply-chain manager responsible for setting replenishment policy and deciding whether a learned policy is safe enough to test in shadow mode.

## Affected Stakeholders

- Customers, who may experience unavailable products when the policy under-orders.
- Store and warehouse staff, who receive, store, and manage ordered inventory.
- Finance and operations teams, which are affected by tied-up cash, lost sales, and waste.
- Business leadership, which owns the final deployment decision and risk tolerance.

## Goals

- Frame one bounded five-product retail replenishment problem as a sequential decision problem.
- Build a deterministic and seeded local simulator for safe experimentation.
- Implement one transparent tabular Q-learning policy that can make product-level replenishment decisions across five product profiles.
- Compare the learned policy fairly against random and rule-based baselines.
- Evaluate aggregate, per-product, financial, and operational failure metrics.
- Produce reproducible artifacts, tests, plots, and a governance recommendation.
- Keep core training under the assignment's 10–15 minute laptop limit.

## Non-Goals

- Managing more than five products, multiple stores, warehouses, suppliers, or interacting agents.
- Learning a joint five-product action or optimizing cross-product budget allocation.
- Modeling product substitution, complementary demand, or shared-capacity competition.
- Training or fine-tuning an LLM.
- Using real customers, production inventory systems, or live business data.
- Claiming that simulator performance proves production readiness.
- Optimizing a continuous order quantity in the core tabular implementation.
- Making DQN or PPO part of the required core submission.
- Finalizing the MDP or reward equation during Step 1.

## Initial Assumptions for Later Review

These assumptions keep the initial problem bounded. They may be revised when the MDP is specified.

- Five products are replenished over repeated fixed-length episodes.
- Products may have different bounded demand and cost profiles.
- One shared policy makes one product-level replenishment decision at a time.
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
- The evaluation reports aggregate and per-product reward, stockouts, excess inventory, and other approved business metrics.
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
- GitHub remote: `https://github.com/AbdelazizAhmed314/inventory-replenishment-rl`

## Final Delivery Interface

The approved core delivery will be an artifact-driven command-line repository, not a Streamlit application.

- `make sync`: install the locked dependencies.
- `make smoke`: run the fastest honest end-to-end validation path.
- `make run`: run the core simulator, baselines, training, evaluation, and artifact generation.
- `make verify`: mechanically verify required generated artifacts.
- `make demo`: display a readable seeded episode for a selected policy.
- `make test` and `make check`: run automated tests and code-quality checks.

The grader will use the commands above and inspect generated CSV, JSON, Markdown, and plot files under `artifacts/`. A Streamlit interface may be considered only as optional polish after the core submission is complete.

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

## Step 1 Approval Record

1. **Business problem:** Approved with five products instead of one.
2. **Core method:** Tabular Q-learning approved.
3. **Stakeholders, goals, non-goals, and assumptions:** Approved, with the five-product scope clarified above.
4. **Proposed success criteria:** Approved.
5. **Repository and reproducibility strategy:** Approved.
6. **Final delivery interface:** Artifact-driven CLI and Makefile workflow approved; Streamlit is not part of the core.

## Step 2 Design Constraint

Step 2 must confirm that the five-product design remains tractable for tabular Q-learning. The default design will use one shared policy making product-level decisions rather than a joint five-product action. Any proposed state representation must include only enough product context to support the next decision without creating an impractically large Q-table.
