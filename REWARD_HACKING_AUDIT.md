# Reward-Hacking Audit

## Intended Outcome

The policy should replenish inventory profitably while avoiding both unmet demand and excess
inventory. Reward is the optimization signal; sales, stockouts, holding, terminal excess,
orders, service level, and violations remain separate evaluation metrics.

## Reward

```text
reward =
    10 * sales_units
    - 2 * ordered_units
    - 1 * holding_units
    - 15 * stockout_units
    - 5 * terminal_excess_units
```

Hard action and capacity constraints are enforced by the simulator rather than represented
only as penalties.

## Pre-Training Checks

| Possible Hacking Path | Safeguard or Detection | Result |
|---|---|---|
| Order nothing to avoid procurement cost | Stockout penalty and maximum-demand check | Strongly negative reward. |
| Always order the maximum to avoid stockouts | Procurement, holding, and terminal-excess costs | Does not beat the rule. |
| Place unusable final-week orders | Procurement and terminal-excess charges | Penalized. |
| Exceed capacity | Action mask and environment rejection | Mechanically impossible. |
| Improve reward while degrading operations | Report raw operational metrics | Required in every evaluation. |
| Exploit future demand | Future information excluded from observations | Prevented by interface design. |

## Post-Training Findings

The learned policy is not exploiting infeasible actions: standard and stress evaluations
record zero capacity violations. It also avoids final-week dumping, ordering only `0.075`
units on average in the final week under standard evaluation.

The remaining observed reward-alignment risk is aggressive inventory:

- Standard evaluation: learned average holding is `6.72` units versus `4.05` for the rule.
- Zero-demand stress: learned reward is `-2,287.40` versus `-1,774.70` for the rule because
  the learned policy continues ordering after demand disappears.
- High-holding-cost stress: learned mean reward falls below the rule (`1,026.64` versus
  `1,106.86`).
- Specialty coffee beans has learned average holding of `10.36` units versus `4.08` for
  the rule.

This behavior is not a hidden simulator exploit. It is a real objective tradeoff created by
the selected reward weights and training distribution: lower stockout risk is purchased
with additional inventory.

## Residual Risks

- Reward weights are transparent simulated values, not calibrated business estimates.
- A demand collapse is outside the standard training distribution.
- Uniform economics omit real product differences in margin, shelf life, and service value.
- The finite horizon can teach end-of-period behavior that differs from continuous operation.
- A single training run does not establish robustness across training seeds.

## Controls

- Keep capacity and feasible actions as hard environment constraints.
- Shadow the learned policy before any automated ordering.
- Compare every recommendation with the rule fallback and actual demand.
- Alert on inventory, service, action rejection, and demand-drift thresholds.
- Revert to the rule policy when thresholds in `FAILURE_AND_SAFETY_ANALYSIS.md` are breached.

The reward is suitable for continued controlled testing, but the observed demand-drop and
holding-cost weaknesses prevent an autonomous rollout recommendation.
