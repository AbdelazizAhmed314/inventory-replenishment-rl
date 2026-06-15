# Pre-Training Reward-Hacking Audit

## Status

Step 5 is ready for review. This audit was completed before implementing or training a
Q-learning agent.

## Intended Outcome

The policy should replenish inventory profitably while avoiding both unmet demand and excess
inventory. The scalar reward is a training signal, not the complete business evaluation.
Stockouts, holding, terminal excess, sales, orders, and capacity violations remain separate
reported metrics.

## Course Basis

The design follows `Lec-4-5-6-deep-reinforcement-learning-agentic-ai-deckset.pdf`:

> "Reward is not the KPI dashboard."

The same deck distinguishes the two artifacts:

> "KPIs describe what happened. Rewards teach the agent what to repeat."

It also names the central inventory reward-hacking risk:

> "Inventory over-orders to avoid stockouts."

Hard action and capacity constraints remain environment-enforced because the deck states:

> "Some rules are not learned. They are enforced."

`Lec5.mp3` recommends testing reward alignment on "edge cases" before optimization. The
hand-designed policies and controlled demand paths below implement that course guidance.
The audit-report structure is also consistent with the reward-hacking artifact shown in
`mcgill-showcases/projects/learning-agents-showcase`.

## Final Reward

For every product transition:

```text
reward =
    10 * sales_units
    - 2 * ordered_units
    - 1 * holding_units
    - 15 * stockout_units
    - 5 * terminal_excess_units
```

All values are transparent simulated monetary units, not empirical estimates. The same
economics apply to all five products so that this educational project isolates replenishment
behavior from arbitrary product-margin assumptions.

| Term | Business Meaning |
|---|---|
| `+10 * sales_units` | Revenue proxy for satisfying demand. |
| `-2 * ordered_units` | Procurement cost paid when the action is selected. |
| `-1 * holding_units` | Weekly carrying cost for unsold inventory. |
| `-15 * stockout_units` | Lost-sale and customer-service harm beyond missing revenue. |
| `-5 * terminal_excess_units` | End-of-horizon markdown, disposal, or write-off risk. |

Initial inventory is treated as a sunk cost because it is identical across policies and cannot
be changed by the agent. The reward still charges holding and terminal-excess costs on it.

## Pre-Training Threat Audit

| Possible Hacking Path | Safeguard or Detection | Current Finding |
|---|---|---|
| Order nothing to avoid procurement cost | Stockout penalty and high-demand zero-order scenario | Strongly negative reward. |
| Always order the maximum to avoid stockouts | Procurement, holding, and terminal-excess costs; compare against order-up-to rule | Maximum-feasible policy does not beat the sensible rule. |
| Place a final-week order that can never sell | Charge procurement cost and include the selected terminal action in terminal excess | Final-week dumping is penalized. |
| Exceed capacity and accept a reward penalty | Action mask and environment rejection | Impossible; safety is not delegated to reward. |
| Optimize reward while business KPIs degrade | Continue reporting raw sales, stockouts, holding, excess, orders, and violations | Required in all evaluation outputs. |
| Exploit future demand or random seeds | Exclude future information from state and policies | Prevented by the approved observation design. |

## Hand-Designed Scenario Checks

Run:

```bash
make reward-audit
```

This writes `evidence/reward_scenario_checks.csv`. The required checks are:

- a sensible order-up-to rule under deterministic medium-profile demand is profitable and has
  no stockouts;
- all-zero ordering under maximum demand is strongly negative;
- maximum-feasible ordering under zero demand is strongly negative;
- over 50 identical stochastic environment seeds, zero-order, maximum-feasible, and random
  policies do not beat the sensible order-up-to rule;
- every scenario records zero capacity violations.

### Observed Results

| Scenario | Episodes | Mean Reward | Stockout Units | Holding Units | Terminal Excess | Passed |
|---|---:|---:|---:|---:|---:|---|
| Controlled order-up-to, medium-profile demand | 1 | `2,786.00` | `0.00` | `149.00` | `41.00` | Yes |
| Controlled zero-order, maximum demand | 1 | `-16,125.00` | `1,125.00` | `0.00` | `0.00` | Yes |
| Controlled maximum-feasible, zero demand | 1 | `-2,625.00` | `0.00` | `1,725.00` | `150.00` | Yes |
| Stochastic order-up-to | 50 | `1,805.76` | `34.54` | `247.34` | `45.20` | Yes |
| Stochastic all-zero | 50 | `-3,863.14` | `301.76` | `86.44` | `0.02` | Yes |
| Stochastic maximum-feasible | 50 | `1,367.56` | `4.86` | `909.54` | `110.72` | Yes |
| Stochastic random-feasible | 50 | `588.92` | `60.64` | `622.38` | `79.70` | Yes |

Maximum-feasible ordering achieves fewer stockouts than the order-up-to rule, but its much
higher holding and terminal excess reduce its reward. This is the intended tradeoff, not proof
that stockouts no longer matter; later evaluation must continue reporting both outcomes.

## Rejected Alternatives

- **Sales-only reward:** rejected because it ignores costs and encourages over-ordering.
- **Profit proxy without stockout penalty:** rejected because it omits customer-service harm and
  can favor under-ordering.
- **No terminal-excess penalty:** rejected because an agent could ignore inventory stranded at
  the finite horizon.
- **Large reward penalty for capacity violations:** rejected because capacity is a hard rule that
  is already enforced mechanically.
- **Product-specific economic weights:** deferred because invented product margins would add
  complexity and could obscure whether behavior is caused by demand or arbitrary economics.
- **Reward clipping or normalization:** rejected for the core model because the current scale is
  bounded and preserving component meaning makes the audit easier.

## Residual Risks and Review Items

- The weights are defensible educational proxies, not calibrated business estimates.
- A large stockout penalty may still favor aggressive ordering in sustained high demand.
- Uniform economics omit real differences in margins, shelf life, and service importance.
- The 12-week terminal penalty may teach horizon-specific wind-down behavior.
- Passing this pre-training audit does not prove the learned policy will avoid reward hacking.
  Step 8 must repeat the audit using observed learned behavior.

## Review Decision

Approve Step 5 only if the reward tradeoffs, scenario rankings, remaining raw metrics, and
separation of reward from hard safety controls are acceptable.
