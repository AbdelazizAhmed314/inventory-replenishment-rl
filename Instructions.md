# Instructions

Train and evaluate a sequential decision agent on a small business simulator. The default domain is retail inventory replenishment or dynamic pricing; a similar small environment of your choosing is fine if you check with us first.

This is not a compute contest. Most of the grade is in how cleanly you frame the decision, design the reward, evaluate the result, and reason about what could go wrong in production.

## Two Paths, Equal Value

| Path | What it looks like |
|---|---|
| **Core** | Contextual bandit, tabular Q-learning, or a small custom MDP |
| **Stretch** | DQN or PPO using Gymnasium / Stable-Baselines3 |

***Either path can earn full marks. Pick the one that lets you do the thinking well — not the one that looks more impressive on a slide.***

## What to Submit

- **A working code project with a great README.md** — same standard as Assignment 1. See *What we expect in your code and README*.
- **Problem framing** — state, action, reward, transition, horizon.
- **A baseline** — random, rule-based, or a simple heuristic policy.
- **A learning agent** — Q-learning, bandit, DQN, PPO, or equivalent.
- **Evaluation against the baseline** — not just final score, behavior on edge episodes too.
- **Plots** — reward curve, performance comparison, or policy behavior.
- **Failure analysis** — reward hacking, unsafe behavior, instability, overfitting.
- **Business memo (1–2 pages)** — should this be deployed, shadowed, or rejected? Argue it.

## How You Will Be Graded — 25 Points

| Category | Points |
|---|---:|
| MDP and reward design | 5 |
| Implementation and reproducibility | 5 |
| Evaluation against baseline | 5 |
| Correct use of RL / DRL concepts | 4 |
| Safety, reward hacking, governance | 3 |
| Business communication | 3 |

## Compute Limit

Training should run in under 10–15 minutes on a normal laptop. If you reach for something heavier, submit a saved run artifact so the grader does not have to retrain it.

## Due Date

**Due on Jun 14, 2026 11:59 PM**
