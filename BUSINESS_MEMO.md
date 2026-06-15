# Business Memo: Inventory Replenishment Policy Recommendation

**To:** Retail inventory decision-makers

**Subject:** Recommendation for the five-product replenishment policy

**Decision:** Proceed with shadow mode; do not automate orders yet

## Executive Summary

We developed a tabular Q-learning policy that recommends weekly order quantities for five
retail products. In a controlled simulator, the learned policy materially improves the
main objective and customer service compared with a sensible order-up-to rule. However, it
also carries more inventory and performs poorly when demand suddenly disappears or holding
cost rises. We recommend an 8-to-12-week shadow evaluation with the rule policy retained as
the operational fallback.

## Business Result

Formal evaluation used 200 unseen demand seeds, with all policies facing identical demand.
The learned policy achieved mean reward of `2,221.11`, compared with `1,840.95` for the rule,
a `20.7%` improvement. It won on 196 of 200 paired seeds.

The improvement is operationally meaningful:

| Metric | Learned Policy | Rule Policy | Change |
|---|---:|---:|---:|
| Service level | `94.65%` | `90.84%` | `+3.81` points |
| Stockout units | `21.13` | `35.70` | `-14.57` |
| Terminal excess | `17.67` | `46.94` | `-29.27` |
| Average holding | `6.72` | `4.05` | `+2.66` |

The policy improves reward and service for all five products. Bottled water, umbrellas, and
packaged snacks show the largest absolute reward gains. The main concern is specialty coffee
beans, where learned average holding is `10.36` units versus `4.08` under the rule.

## Why We Should Not Automate Yet

Stress testing shows that the learned policy depends on the environment it was trained in.
When demand is forced to zero, it continues ordering and performs worse than the rule on all
50 tested seeds. When holding cost is raised from `1` to `4`, the learned policy also falls
below the rule on average. Under maximum demand, it reduces losses compared with the rule
but service level falls to `42.33%`, which is not operationally acceptable.

The tests record zero capacity violations, and the simulator blocks infeasible actions.
These are important safeguards, but they do not resolve the gap between simulated and real
operations. The model does not yet represent supplier failures, promotions, spoilage,
shared budgets, or changing lead times. Its economic weights are transparent assumptions,
not values calibrated from real financial data.

## Recommended Next Step

Run the learned policy in shadow mode for at least 8 to 12 weeks. During this period:

- Continue using the rule policy or human-approved actions for actual orders.
- Record learned recommendations, approved orders, actual demand, service, inventory, and
  costs.
- Require review when four-week service falls below `90%`, average holding exceeds `8`
  units, product holding exceeds `12` units for two weeks, or modeled costs change by more
  than `20%`.
- Stop learned recommendations immediately after any capacity or action-validation failure.

Before considering automation, retrain across multiple seeds, calibrate reward weights to
real economics, represent missing operational constraints, and demonstrate stable shadow
performance. The current evidence justifies continued controlled testing, not autonomous
ordering.
