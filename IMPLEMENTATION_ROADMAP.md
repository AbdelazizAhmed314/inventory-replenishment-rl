# Assignment 2 Implementation Roadmap

## Purpose

This document is the working checklist for completing Assignment 2 one reviewed step at a time.

For each step:

1. Prompt Codex to implement the step.
2. Review the outputs and evidence.
3. Request modifications or clarification as needed.
4. Approve the checkpoint before moving to the next step.

Do not begin optional stretch work until all core steps are complete and reviewed.

## Current Project Direction

The proposed core project is a **retail inventory replenishment agent trained with tabular Q-learning**.

The agent will learn how much inventory to order at each decision point while balancing profit, stockouts, holding costs, and other business constraints. Training will happen entirely in a small local simulator.

This direction can be changed during Step 1.

## Repository and Tooling Decision

The assignment will be developed as a standalone Git repository inside `Assignments/Assignment2`.

- Initialize Git during Step 1.
- Use a course-aligned Python project structure rather than a generic Cookiecutter template.
- Use `pyproject.toml` and `uv.lock` for dependencies and reproducibility.
- Keep source code under `src/`, tests under `tests/`, and generated evidence under `artifacts/` or `evidence/`.
- Add a GitHub remote after the initial scaffold has been reviewed.
- Verify the final project from a clean clone before submission.

**Cookiecutter decision:** Do not use Cookiecutter initially. It is not required by the assignment or course sources, and a generic template could add unnecessary structure. We can reconsider only if a specific approved template clearly matches the project.

## Requirements Summary

The completed assignment must include:

- A reproducible working code project and strong `README.md`
- Clear MDP definition: state, action, reward, transition, and horizon
- A random, rule-based, or heuristic baseline
- A learning agent
- Evaluation against the baseline, including edge episodes
- Plots showing learning and policy performance
- Failure, safety, instability, overfitting, and reward-hacking analysis
- A 1–2 page business memo recommending deployment, shadow mode, or rejection
- Training that runs in under 10–15 minutes on a normal laptop

## Requirement Traceability

| Assignment Requirement | Primary Roadmap Step |
|---|---|
| Working code project and strong `README.md` | Steps 3–10 |
| Problem framing: state, action, reward, transition, horizon | Steps 1–2 and 5 |
| Baseline policy | Step 4 |
| Learning agent | Step 6 |
| Evaluation against the baseline and edge episodes | Step 7 |
| Plots | Steps 4, 6, and 7 |
| Failure analysis | Steps 5 and 8 |
| Business memo | Step 9 |
| Reproducibility and compute limit | Steps 3, 6, 9, and 10 |

## Course Context Guiding the Plan

- **`Lec-4-5-deep-reinforcement-learning-agentic-ai-deckset.pdf`** explains that a state should contain the information needed for the next decision, rather than every available variable.
- The same lecture deck defines the central reward-design question as: **“what outcome do we really want?”**
- It also warns: **“Weak rewards produce strong agents with bad habits.”**
- **`Lec6.mp3`** explains tabular RL as a lookup-table approach where every state-action pair has a learned value.
- The **`learning-agents-showcase`** course source emphasizes that the learning component controls decisions and does not retrain the LLM.

---

# Core Implementation Steps

## Step 1: Confirm the Business Problem and Scope

**Status:** Ready for review

### Goal

Choose the final business problem and define a scope small enough to implement, train, evaluate, and explain well.

### Proposed Decisions

- Domain: single-product retail inventory replenishment
- Learning method: tabular Q-learning
- Episode: a fixed number of simulated business days
- Agent decision: how many units to order
- Primary objective: maximize long-term profit while controlling stockouts and excess inventory

### Work

- Confirm or revise the proposed project direction.
- Write a concise problem statement.
- Identify the business decision-maker and users affected by the policy.
- Define explicit project goals and non-goals.
- Record assumptions about demand, replenishment, and business operations.
- Decide what successful performance means.
- Initialize a standalone Git repository.
- Create the course-aligned Python project scaffold.
- Configure `pyproject.toml`, `uv.lock`, `.gitignore`, and standard run commands.
- Create the GitHub repository and connect the remote after the local scaffold is reviewed.

### Expected Outputs

- `PROJECT_BRIEF.md`
- Standalone local Git repository
- Course-aligned project structure
- `pyproject.toml`, `uv.lock`, and `.gitignore`
- Initial dependency and run strategy
- Connected GitHub repository after scaffold approval

### Review Checkpoint

- Is the decision genuinely sequential?
- Do current actions affect future states or rewards?
- Is the scope small enough for laptop-friendly training?
- Is the business objective clear?
- Are the non-goals explicit?
- Is the repository structure minimal and appropriate for the assignment?
- Can dependencies be installed reproducibly?
- Does the initial commit exclude generated files, secrets, and local environments?

### Completion Condition

The business problem, boundaries, and success criteria are approved.

### Prompt to Start

> Implement Step 1 from `IMPLEMENTATION_ROADMAP.md`. Use the Agentic AI course notebook as the course source of truth, name the specific sources used, and stop for my review when the expected outputs are ready.

---

## Step 2: Define the Markov Decision Process

**Status:** Not started

### Goal

Translate the business problem into a precise and implementable Markov Decision Process.

### Work

- Define the state representation.
- Define the discrete action space.
- Define the transition process.
- Define the episode horizon and termination conditions.
- Draft the reward function and its components.
- Explain why the state is sufficient for the next decision.
- Identify information deliberately excluded from the state.

### Proposed Initial MDP

| Component | Proposed Definition |
|---|---|
| State | Inventory level, recent demand category, and incoming-order status |
| Action | Order 0, 5, 10, or 20 units |
| Reward | Sales revenue minus purchasing, holding, stockout, and waste costs |
| Transition | Demand occurs, inventory changes, and placed orders progress |
| Horizon | One fixed-length simulated business period |

### Expected Outputs

- `MDP_SPEC.md`
- Machine-readable environment configuration
- State and action encoding definitions

### Review Checkpoint

- Can every state and action be represented unambiguously in code?
- Is the state small enough for a Q-table?
- Does the state avoid future-information leakage?
- Does the reward reflect the intended business outcome?
- Are delayed consequences represented?

### Completion Condition

The MDP specification is internally consistent and approved before simulator implementation.

### Prompt to Start

> Implement Step 2 from `IMPLEMENTATION_ROADMAP.md`. Explain and document the MDP, but stop before building the full simulator so I can review the design.

---

## Step 3: Build and Validate the Simulator

**Status:** Not started

### Goal

Create a deterministic, testable environment in which policies can safely interact with the business problem.

### Work

- Implement environment reset and step behavior.
- Implement demand generation and inventory transitions.
- Implement ordering, lead-time, capacity, and cost rules.
- Support deterministic seeded runs.
- Log complete episode trajectories.
- Add unit tests for transitions, rewards, termination, and reproducibility.
- Generate sample episodes without a learning agent.

### Expected Outputs

- Simulator source code
- Environment configuration
- Simulator unit tests
- `evidence/sample_episodes.csv`
- `evidence/simulator_validation.md`

### Review Checkpoint

- Do identical seeds produce identical episodes?
- Are inventory and cost calculations correct?
- Are impossible states prevented?
- Does the simulator avoid future-information leakage?
- Are reward components visible in the logs?

### Completion Condition

The simulator passes its tests and sample trajectories behave as expected.

### Prompt to Start

> Implement Step 3 from `IMPLEMENTATION_ROADMAP.md`. Build and test the simulator, generate the requested evidence, and stop for my review before implementing any policy.

---

## Step 4: Implement Baseline Policies and Safety Constraints

**Status:** Not started

### Goal

Establish meaningful comparison policies and enforce hard operational limits before training the RL agent.

### Work

- Implement a random baseline.
- Implement a sensible rule-based reorder policy.
- Define hard constraints such as order limits, inventory capacity, and budget limits.
- Enforce constraints inside the environment.
- Evaluate baseline policies across multiple seeded episodes.
- Document why the rule-based policy is a credible benchmark.

### Expected Outputs

- Baseline policy source code
- Safety-constraint tests
- `evidence/baseline_episode_results.csv`
- `evidence/baseline_summary.csv`
- Baseline comparison plot

### Review Checkpoint

- Is the rule-based baseline reasonable?
- Are all constraints enforced by code?
- Can the random policy expose simulator weaknesses?
- Are results aggregated across enough seeds?
- Are the metrics useful for later RL comparison?

### Completion Condition

Baseline behavior and hard constraints are validated and approved.

### Prompt to Start

> Implement Step 4 from `IMPLEMENTATION_ROADMAP.md`. Add the baseline policies and hard safety constraints, evaluate them, and stop for review before training the Q-learning agent.

---

## Step 5: Finalize the Reward and Conduct a Pre-Training Hacking Audit

**Status:** Not started

### Goal

Ensure the numerical reward represents the actual business objective before allowing the agent to optimize it.

### Work

- Finalize the reward equation and each component’s scale.
- Explain the business meaning of every reward term.
- Test rewards on hand-designed scenarios.
- Identify ways the agent could maximize reward while violating business intent.
- Add safeguards or evaluation metrics for identified failure modes.
- Record reward-design alternatives that were rejected.

### Expected Outputs

- Final reward section in `MDP_SPEC.md`
- Reward calculation tests
- `REWARD_HACKING_AUDIT.md`
- `evidence/reward_scenario_checks.csv`

### Review Checkpoint

- Does the reward balance profit, stockouts, and excess inventory?
- Can the agent benefit by ordering nothing?
- Can the agent benefit by over-ordering?
- Are important business outcomes missing from the reward?
- Are safety controls separate from reward penalties where appropriate?

### Completion Condition

The reward specification and hacking audit are approved.

### Prompt to Start

> Implement Step 5 from `IMPLEMENTATION_ROADMAP.md`. Finalize and test the reward design, conduct the pre-training reward-hacking audit, and stop for my review.

---

## Step 6: Implement and Train the Tabular Q-Learning Agent

**Status:** Not started

### Goal

Train a reproducible Q-learning agent that learns a policy from simulator experience.

### Work

- Implement the Q-table.
- Implement epsilon-greedy action selection.
- Implement the Q-learning update.
- Set and document learning rate, discount factor, epsilon schedule, episode count, and seed strategy.
- Log episode rewards and behavior metrics.
- Save the trained Q-table and run configuration.
- Add tests for action selection and Q-value updates.

### Expected Outputs

- Q-learning agent and training code
- Q-learning unit tests
- Saved run configuration
- `artifacts/q_table.csv`
- `evidence/training_history.csv`
- Reward-learning curve

### Review Checkpoint

- Is the update rule implemented correctly?
- Does exploration decrease appropriately?
- Is training reproducible?
- Does the reward curve show learning rather than only noise?
- Do learned Q-values produce understandable decisions?
- Does training remain under the compute limit?

### Completion Condition

The training implementation is correct, reproducible, and produces a plausible learned policy.

### Prompt to Start

> Implement Step 6 from `IMPLEMENTATION_ROADMAP.md`. Train the tabular Q-learning agent, save its evidence and artifacts, and stop for review before formal evaluation.

---

## Step 7: Evaluate the Learned Policy Against Baselines

**Status:** Not started

### Goal

Determine whether the learned policy reliably improves on the baselines and understand when it fails.

### Work

- Freeze the trained policy before evaluation.
- Evaluate Q-learning, rule-based, and random policies on identical unseen seeds.
- Compare average reward and business metrics.
- Evaluate edge scenarios such as demand spikes, demand drops, and adverse cost conditions.
- Report variability, not only averages.
- Inspect representative successful and failed episodes.
- Create clear comparison plots.

### Expected Outputs

- Evaluation code and tests
- `evidence/policy_evaluation_results.csv`
- `evidence/scenario_evaluation_results.csv`
- `EVALUATION_REPORT.md`
- Reward curve and policy-comparison plots
- Edge-case behavior plots or tables

### Review Checkpoint

- Was evaluation performed on unseen seeds?
- Did all policies face identical evaluation scenarios?
- Does Q-learning outperform the meaningful baseline?
- Are gains consistent or dependent on a few episodes?
- What business tradeoffs changed?
- Which edge scenarios expose weaknesses?

### Completion Condition

The agent’s performance and limitations are supported by reproducible evidence.

### Prompt to Start

> Implement Step 7 from `IMPLEMENTATION_ROADMAP.md`. Evaluate the frozen learned policy against both baselines, include edge scenarios and plots, and stop for my review.

---

## Step 8: Complete Failure, Safety, and Governance Analysis

**Status:** Not started

### Goal

Explain how the learned policy could fail in production and define a responsible rollout decision.

### Work

- Perform a post-training reward-hacking review using observed behavior.
- Analyze unsafe behavior, instability, overfitting, and simulator mismatch.
- Identify monitoring metrics and alert thresholds.
- Define rollback and human-override conditions.
- Recommend deploy, shadow, or reject.
- Explain what evidence would be required before broader deployment.

### Expected Outputs

- Updated `REWARD_HACKING_AUDIT.md`
- `FAILURE_AND_SAFETY_ANALYSIS.md`
- Draft business recommendation
- Monitoring and rollback plan

### Review Checkpoint

- Does the analysis discuss observed failures rather than only hypothetical ones?
- Are simulator limitations explicit?
- Is the rollout recommendation supported by evidence?
- Are monitoring and rollback conditions concrete?
- Does the recommendation avoid overstating offline results?

### Completion Condition

The project has a defensible governance position and clear production boundaries.

### Prompt to Start

> Implement Step 8 from `IMPLEMENTATION_ROADMAP.md`. Complete the failure, safety, reward-hacking, and rollout analysis, then stop for my review.

---

## Step 9: Finish the README and Business Memo

**Status:** Not started

### Goal

Turn the implementation and evidence into a clear, reproducible submission.

### Work

- Write a complete `README.md`.
- Explain installation, training, evaluation, artifact locations, and expected runtime.
- Document the MDP, algorithm choice, baseline, and major findings.
- Write the required 1–2 page business memo.
- State whether the policy should be deployed, shadowed, or rejected.
- Ensure all claims link to generated evidence.

### Expected Outputs

- `README.md`
- `BUSINESS_MEMO.md`
- Finalized project documentation

### Review Checkpoint

- Can a grader reproduce the project from the README?
- Does the README explain the business problem and technical design clearly?
- Is the memo understandable to a business audience?
- Is the rollout recommendation evidence-based?
- Does the documentation identify limitations honestly?

### Completion Condition

The written deliverables are complete, clear, and aligned with the implementation.

### Prompt to Start

> Implement Step 9 from `IMPLEMENTATION_ROADMAP.md`. Finish the README and 1–2 page business memo using the existing evidence, then stop for my review.

---

## Step 10: Run the Final Submission Audit

**Status:** Not started

### Goal

Verify that the complete project satisfies every assignment requirement and runs reliably from a clean setup.

### Work

- Run all tests.
- Run training and evaluation from documented commands.
- Confirm runtime stays within the assignment limit.
- Verify generated artifacts and plots.
- Check all assignment requirements against the final files.
- Review documentation for consistency.
- Remove temporary files and clearly identify submission files.
- Clone the GitHub repository into a clean temporary folder.
- Reproduce setup, tests, training, evaluation, and artifact verification from the clean clone.
- Confirm the final reviewed changes are committed and pushed.

### Expected Outputs

- `SUBMISSION_CHECKLIST.md`
- Final test and reproducibility evidence
- Clean final assignment folder

### Review Checkpoint

- Does every assignment requirement map to a file or artifact?
- Do the documented commands work?
- Are all results reproducible?
- Are plots readable and correctly labeled?
- Is the project ready for grading without unstated setup?
- Does a clean GitHub clone reproduce the documented workflow?

### Completion Condition

The complete core submission passes its final audit.

### Prompt to Start

> Implement Step 10 from `IMPLEMENTATION_ROADMAP.md`. Run the final submission audit, fix any issues discovered, and report whether the core assignment is ready.

---

# Optional Stretch Steps

Optional steps begin only after Step 10 is complete.

## Stretch Step A: Sensitivity and Robustness Analysis

**Status:** Not started

### Goal

Test whether the conclusions remain stable when assumptions and Q-learning settings change.

### Work

- Compare multiple learning rates, discount factors, and exploration schedules.
- Test alternative demand patterns and cost structures.
- Evaluate multiple training seeds.
- Summarize stability and sensitivity.

### Expected Outputs

- `evidence/sensitivity_results.csv`
- Sensitivity plots
- Robustness section in `EVALUATION_REPORT.md`

### Review Checkpoint

- Are conclusions stable across reasonable assumptions and training seeds?
- Do any results materially weaken the core recommendation?

### Completion Condition

The sensitivity findings are documented without replacing or obscuring the approved core results.

### Prompt to Start

> Implement Stretch Step A from `IMPLEMENTATION_ROADMAP.md`. Run the sensitivity and robustness analysis without changing the approved core results, then stop for my review.

---

## Stretch Step B: Add a DQN Agent

**Status:** Not started

### Goal

Compare tabular Q-learning with a neural-network approximation of the Q-function.

### Work

- Implement DQN using Gymnasium and Stable-Baselines3 or another approved framework.
- Use the same simulator, action space, reward, and evaluation scenarios.
- Save the trained model and training history.
- Compare performance, stability, runtime, and complexity against tabular Q-learning.

### Expected Outputs

- DQN implementation and configuration
- Saved DQN model
- `evidence/dqn_training_history.csv`
- `evidence/rl_family_comparison.csv`
- DQN comparison plots and analysis

### Review Checkpoint

- Does DQN improve performance meaningfully?
- Is the comparison fair?
- Is the additional complexity justified?
- Is training still within the assignment compute limit?

### Completion Condition

DQN is evaluated fairly and retained only as an optional comparison.

### Prompt to Start

> Implement Stretch Step B from `IMPLEMENTATION_ROADMAP.md`. Add and evaluate DQN as an optional comparison while preserving the completed tabular core, then stop for my review.

---

## Stretch Step C: Add a PPO Agent

**Status:** Not started

### Goal

Evaluate a policy-gradient approach on the same business environment.

### Work

- Implement PPO using Gymnasium and Stable-Baselines3.
- Use the same simulator and evaluation protocol.
- Save the trained policy and training evidence.
- Compare PPO with tabular Q-learning, DQN if available, and the baselines.

### Expected Outputs

- PPO implementation and configuration
- Saved PPO model
- `evidence/ppo_training_history.csv`
- Updated `evidence/rl_family_comparison.csv`
- PPO comparison plots and analysis

### Review Checkpoint

- Does PPO provide a meaningful advantage?
- Is training stable across seeds?
- Does it remain within the compute limit?
- Does it improve business outcomes or only simulator reward?

### Completion Condition

PPO is evaluated fairly and retained only as an optional comparison.

### Prompt to Start

> Implement Stretch Step C from `IMPLEMENTATION_ROADMAP.md`. Add and evaluate PPO as an optional comparison while preserving the completed core submission, then stop for my review.

---

## Stretch Step D: Package the Environment as a Gymnasium Environment

**Status:** Not started

### Goal

Create a reusable standard interface for the simulator and validate it against Gymnasium expectations.

### Work

- Implement Gymnasium observation and action spaces.
- Add `reset()` and `step()` interfaces.
- Run Gymnasium environment validation.
- Confirm the core Q-learning workflow still behaves consistently.

### Expected Outputs

- Gymnasium-compatible environment
- Environment validation tests
- Updated README usage instructions

### Review Checkpoint

- Does the environment pass Gymnasium validation?
- Does packaging preserve the approved core behavior and results?

### Completion Condition

The Gymnasium interface is validated without changing the approved core environment semantics.

### Prompt to Start

> Implement Stretch Step D from `IMPLEMENTATION_ROADMAP.md`. Package the approved simulator as a validated Gymnasium environment without changing the core project behavior, then stop for my review.

---

# Progress Tracker

| Step | Description | Status | Approved |
|---|---|---|---|
| 1 | Confirm business problem and scope | Ready for review | No |
| 2 | Define the MDP | Not started | No |
| 3 | Build and validate simulator | Not started | No |
| 4 | Implement baselines and safety constraints | Not started | No |
| 5 | Finalize reward and pre-training hacking audit | Not started | No |
| 6 | Implement and train Q-learning | Not started | No |
| 7 | Evaluate learned policy | Not started | No |
| 8 | Complete failure, safety, and governance analysis | Not started | No |
| 9 | Finish README and business memo | Not started | No |
| 10 | Run final submission audit | Not started | No |
| A | Sensitivity and robustness analysis | Optional | No |
| B | Add DQN agent | Optional | No |
| C | Add PPO agent | Optional | No |
| D | Package as Gymnasium environment | Optional | No |
