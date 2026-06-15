# Failure, Safety, and Rollout Analysis

## Decision

Use the learned policy in **shadow mode only**. It may generate recommendations beside the
existing rule policy, but a human or the rule policy must remain responsible for orders.

The learned policy performs strongly under standard unseen conditions and records zero
capacity violations. It is not ready for autonomous use because it over-orders after demand
collapses, loses its advantage when holding cost rises, and was trained entirely in a
simulator.

## Observed Failure Modes

| Risk | Evidence | Operational Effect | Control |
|---|---|---|---|
| Demand-collapse over-ordering | Zero-demand reward `-2,287.40`, below rule `-1,774.70` | Excess stock and carrying cost | Drift alert and rule fallback |
| Holding-cost sensitivity | Learned reward `1,026.64`, below rule `1,106.86` | Economic advantage can reverse | Re-evaluate after cost changes |
| Extreme-demand service failure | Maximum-demand service `42.33%` | Severe unmet demand | Escalate supply shortage; do not trust normal policy |
| Product-specific excess | Coffee holding `10.36` versus rule `4.08` | Slow-moving inventory risk | Product-level inventory alert |
| Simulator mismatch | No live lead-time, supplier, promotion, spoilage, or budget data | Offline ranking may not transfer | Shadow comparison and human review |
| Training-seed sensitivity | One trained table | Performance may depend on one run | Multi-seed retraining before automation |

## Hard Safety Controls

- The simulator rejects unsupported or capacity-infeasible actions.
- The action mask remains active for learned and baseline policies.
- Capacity is `30` units per product and the maximum order is `15`.
- Any action rejection or capacity violation outside the simulator is an immediate stop.
- The rule order-up-to policy is the deterministic rollback policy.

## Shadow Monitoring Plan

Run shadow mode for at least 8 to 12 operating weeks. Log the learned recommendation, rule
recommendation, approved action, realized demand, sales, inventory, and costs.

| Metric | Alert Threshold | Response |
|---|---|---|
| Capacity or action validation failure | Any occurrence | Stop learned recommendations and investigate |
| Four-week service level | Below `90%` or more than 2 points below rule | Revert to rule and review demand shift |
| Four-week average holding | Above `8` units or more than 2x rule | Require human approval for positive orders |
| Product average holding | Above `12` units for two consecutive weeks | Freeze replenishment for that product pending review |
| Demand drift | More than 10% of observations outside modeled range, or four-week mean changes by over 30% | Re-evaluate and retrain before resuming |
| Economic drift | Procurement or holding cost changes by over 20% | Re-run stress evaluation before resuming |
| Missing or stale input | Any decision with incomplete current state | Use rule fallback; do not call learned policy |

## Override, Rollback, and Recovery

Authorized inventory managers may override any learned recommendation. Overrides must record
a reason code so repeated disagreement can be analyzed.

Rollback is immediate when a hard safety control fails or when a monitoring threshold is
breached. Recovery requires:

1. Identify whether the cause is data quality, demand drift, changed economics, or policy
   behavior.
2. Correct the simulator or data pipeline where needed.
3. Retrain and rerun unseen-seed plus stress evaluation.
4. Confirm zero violations and acceptable product-level tradeoffs.
5. Restart in shadow mode; do not resume directly into automated ordering.

## Evidence Required Before Automation

- Successful shadow operation across normal and changing demand conditions.
- Multi-seed training robustness with similar business tradeoffs.
- Reward weights calibrated to real product margins, carrying costs, and service targets.
- Real lead-time, supplier, promotion, spoilage, and budget constraints represented.
- Approved monitoring ownership, incident response, and rollback procedures.

Offline evidence supports further testing, not a claim of production safety.
