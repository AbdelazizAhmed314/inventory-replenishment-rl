# Tabular Q-Learning Training Report

## Status

Step 6 is ready for review. The shared tabular Q-learning agent has been implemented and
trained. Formal evaluation on unseen seeds has not been performed and remains Step 7 work.

## Course Basis

The implementation follows `Lec-4-5-6-deep-reinforcement-learning-agentic-ai-deckset.pdf`.
The deck gives the tabular update as:

```text
td_target = reward + gamma * best_next
td_error = td_target - q[state][action]
q[state][action] += alpha * td_error
```

For exploration, it defines epsilon-greedy behavior as:

> "with probability epsilon: choose a random action"

and recommends:

> "high epsilon early"

and:

> "lower epsilon later"

`Lec6.mp3` describes the exploration pattern as:

> "explore a bit, find the best-so-far, then increasingly commit to it."

The action mask remains active during both random exploration and next-state maximization,
consistent with the deck's rule:

> "RL is allowed to optimize within those boundaries."

The saved Q-table, run configuration, history, learning curve, and mechanical verifier follow
the artifact-oriented pattern in
`mcgill-showcases/projects/learning-agents-showcase`.

## Training Configuration

The numerical hyperparameters are educational inferences, not direct course prescriptions.

| Setting | Value | Rationale |
|---|---:|---|
| Episodes | `50,000` | Provides repeated coverage while remaining within the laptop limit. |
| Learning rate (`alpha`) | `0.10` | Updates gradually under stochastic demand. |
| Discount factor (`gamma`) | `0.99` | Values delayed stockout, holding, and terminal effects. |
| Epsilon | `1.00` to `0.05` | Starts with broad exploration and retains limited exploration. |
| Epsilon decay | Linear over first `80%` | Leaves the final `20%` at minimum exploration. |
| Agent seed | `697` | Makes action exploration and tie-breaking reproducible. |
| Environment seeds | `100000` to `149999` | Training-only seed range; Step 7 must use unseen seeds. |
| Initial Q-value | `0.0` | Neutral, transparent initialization. |
| Rolling window | `500` episodes | Smooths the stochastic learning curve. |

## Implementation Checks

- The Q-table shape is `5 x 12 x 31 x 4 x 3 x 4`, or `89,280` cells.
- Random exploration samples only feasible actions.
- Greedy selection and next-state maximization ignore infeasible actions.
- Terminal transitions use no future bootstrap value.
- Tests verify exact TD arithmetic, epsilon decay, deterministic reruns, and Q-table save/load.
- The generated artifact verifier checks row counts, hashes, epsilon endpoints, monotonic
  visitation, and zero visits to infeasible actions.

## Training Results

Training completed in `534.19` seconds, or approximately `8.9` minutes.

| Metric | Initial 500-Episode Mean | Final 500-Episode Mean |
|---|---:|---:|
| Episode reward | `599.05` | `2,117.84` |
| Stockout units | `64.30` | `23.57` |
| Holding units | `616.47` | `416.48` |

- The rolling reward rises steadily and plateaus near `2,100` after epsilon reaches `0.05`.
- The agent visited `39,880` of `46,080` feasible state-action cells.
- No infeasible action cell was visited.
- The saved table contains all `89,280` nominal cells for transparent inspection.

## Representative Learned Decisions

The learned-policy snapshot selects the most-visited state for every product-week.

- In the common initial state, the learned greedy action is `10` units for specialty coffee
  beans, packaged snacks, umbrellas, and bottled water.
- In the common initial state, premium olive oil selects `0`, consistent with its slow-moving
  demand profile.
- All five representative terminal-week states select `0`, showing that the learned policy
  generally recognizes that final-week orders cannot generate future sales.
- Faster-demand products typically choose `10` in frequently visited middle-week states, while
  coffee beans and olive oil more often choose `0` or `5`.

These observations establish plausibility, not generalization.

## Review Risks

- The final training behavior still carries substantial inventory. The lower stockout count may
  partly reflect aggressive ordering, so Step 7 must compare holding and terminal excess against
  the rule-based baseline.
- Unvisited state-action cells retain zero values. Step 7 must inspect whether the frozen policy
  encounters poorly learned states under unseen seeds and edge scenarios.
- One training run demonstrates reproducibility of the implementation, not sensitivity across
  training seeds. Seed sensitivity remains optional robustness work and a Step 8 limitation.
- Training reward is not proof of business performance. Step 7 must evaluate the frozen policy
  on unseen seeds and continue reporting raw business metrics.

## Generated Outputs

- `artifacts/q_table.csv`
- `artifacts/training_run_config.json`
- `artifacts/training_manifest.json`
- `evidence/training_history.csv`
- `evidence/reward_learning_curve.png`
- `evidence/learned_policy_snapshot.csv`

Run `make verify-training` to validate the saved artifact contract.
