# Frozen-Policy Evaluation Report

## Protocol

The learned Q-table is loaded as a deterministic greedy policy. Evaluation performs no
exploration and no Q-value updates. The learned, rule-based, and random policies face
identical demand realizations for each environment seed.

- Standard evaluation: 200 unseen seeds, `200000` through `200199`.
- Stress evaluation: 50 seeds, `210000` through `210049`, for each scenario.
- Training seeds: `100000` through `149999`; there is no overlap.
- Hard capacity violations: zero in all standard and stress runs.

## Standard Unseen-Seed Results

| Policy | Mean Reward | Reward Std. Dev. | Stockout Units | Service Level | Avg. Holding | Terminal Excess |
|---|---:|---:|---:|---:|---:|---:|
| Frozen Q-learning | `2,221.11` | `275.16` | `21.13` | `94.65%` | `6.72` | `17.67` |
| Rule order-up-to | `1,840.95` | `268.25` | `35.70` | `90.84%` | `4.05` | `46.94` |
| Random feasible | `576.17` | `516.29` | `65.32` | `83.47%` | `10.30` | `76.79` |

The learned policy improves mean reward over the rule by `380.16` (`20.7%`) and wins on
196 of 200 paired seeds. It reduces mean stockouts by `14.57` units and improves service
level by `3.81` percentage points. Its main tradeoff is higher average holding: `6.72`
units versus `4.05`, a `65.7%` increase.

All learned-policy decisions during standard evaluation occurred in states with at least
one previously visited feasible state-action pair.

## Product-Level Findings

The learned policy raises mean reward and service level for every product. Its largest
absolute reward gains over the rule occur for bottled water, umbrellas, and packaged
snacks. Specialty coffee beans has the highest learned average holding at `10.36` units,
compared with `4.08` under the rule, making it the clearest target for shadow monitoring.

## Stress Scenarios

| Scenario | Frozen Q-learning Reward | Rule Reward | Learned Wins | Main Finding |
|---|---:|---:|---:|---|
| Maximum demand every week | `-6,168.30` | `-7,379.00` | `49/50` | Learned policy reduces losses, but service is only `42.33%`. |
| Zero demand every week | `-2,287.40` | `-1,774.70` | `0/50` | Learned policy over-orders after demand disappears. |
| Procurement cost raised to `6` | `996.86` | `554.64` | `50/50` | Learned policy remains stronger under this tested cost. |
| Holding cost raised to `4` | `1,026.64` | `1,106.86` | `21/50` | Higher learned inventory reverses the average ranking. |

The demand-spike scenario is intentionally severe: all products demand the configured
maximum every week. The result demonstrates graceful loss reduction, not adequate service.
The demand-drop and high-holding-cost results demonstrate that the policy is sensitive to
economic and demand-distribution changes.

## Representative Episodes

The three highest learned-policy rewards range from `2,822` to `2,921`, with service levels
from `93.50%` to `97.89%`. The three lowest range from `1,352` to `1,598`, with service
levels from `91.67%` to `97.59%`. Even in these low-reward episodes the learned policy
slightly beats the rule, but it consistently carries more inventory.

Detailed paired rows are saved in `evidence/representative_episode_results.csv`.

## Conclusion

The frozen learned policy reliably improves the approved standard-simulator objective and
meaningful service metrics. It should not be treated as production-ready because it fails
to adapt when demand collapses and can become inferior when holding economics change. The
supported decision is a controlled shadow evaluation with the rule policy retained as the
fallback.
