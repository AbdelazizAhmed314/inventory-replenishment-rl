# Markov Decision Process Specification

## Status

This is the proposed Step 2 design for review. It defines the MDP precisely enough to implement, but it does not implement the simulator or finalize the numerical reward parameters.

## Decision Summary

The project will use one tabular Q-learning controller to make product-level replenishment decisions across five products.

- One episode represents 12 simulated weeks.
- Each week contains one decision transition for each of the five products.
- Products do not compete for a shared budget or shared storage capacity.
- The controller chooses one order quantity for one product at a time.
- Each product's next state depends on its own current state, action, and stochastic demand.
- The same Q-table stores all product-level state-action values.

This avoids a joint action such as `(order_product_1, ..., order_product_5)`, which would create an unnecessarily large action space.

## Course Basis

### `Lec-4-5-deep-reinforcement-learning-agentic-ai-deckset.pdf`

The state is intentionally limited to variables needed for the next replenishment decision:

> "State is not 'all available data.' State is the decision-relevant view."

The deck describes the Markov property as:

> "The future depends on the present state and action, not the entire history that led here."

It also warns:

> "Do not pretend the state is complete when it is not."

For inventory decisions, the deck identifies the relevant concepts as stock, forecast, lead time, and capacity. It also warns that future-information leakage can make offline results look better than real behavior.

### `Lec6.mp3`

The lecture explains Q-table scaling using one value per state-action combination:

> "If there are 10 states and four actions the table has 10 multiplied by four 40 cells."

### `adaptive-course-assistant-rl-showcase`

The showcase provides the system-boundary pattern used here:

> "There is one learned controller in the simulator, not multiple learning agents with coordination or competition."

## Unit of Decision

A decision transition concerns one product during one simulated week.

During each week, the controller is called once for each product. The five product transitions are independent except that they:

- use the same policy and Q-table;
- share the same week number;
- contribute to the episode's aggregate business results.

The Q-learning update for a product uses that same product's next-week state and reward.

## State

The state is the tuple:

```text
state = (
    product_id,
    week,
    on_hand,
    incoming_order,
    demand_regime,
)
```

| Variable | Values | Why It Is Needed |
|---|---|---|
| `product_id` | `0`–`4` | Identifies the product's fixed demand profile. Without it, two products with different demand distributions could appear identical. |
| `week` | `0`–`11` | Represents time remaining in the finite 12-week horizon. It is needed because excess inventory near the end of the episode has different consequences. |
| `on_hand` | `0`–`30` units | Exact sellable inventory remaining from the previous week. |
| `incoming_order` | `0`, `5`, `10`, or `15` units | Previously ordered inventory due to arrive before this week's demand. It prevents the controller from ignoring the pipeline and double-ordering. |
| `demand_regime` | `low`, `medium`, or `high` | The observable demand forecast category for the current week. Actual demand remains unknown. |

### Why Exact Inventory Is Used

Exact inventory from `0` to `30` is retained instead of broad low/medium/high buckets because:

- the resulting Q-table is still small;
- exact inventory supports clear capacity checks;
- broad buckets could hide materially different stockout risk;
- it avoids adding unnecessary state aliasing before evidence shows bucketing is needed.

### Information Deliberately Excluded

The state excludes:

- actual current-week demand before it occurs;
- future demand realizations or future random seeds;
- future demand regimes;
- previous rewards and complete episode history;
- other products' states, because there are no cross-product constraints;
- shared budget or shared-capacity information, because those interactions are out of scope;
- learned Q-values, which belong to the policy rather than the environment state.

Actual demand must never be included in the observation before the action. The controller receives only the demand regime available at decision time.

## Product Profiles

The five product IDs represent distinct bounded demand profiles:

| ID | Profile | Low Mean | Medium Mean | High Mean |
|---:|---|---:|---:|---:|
| `0` | Steady light demand | 2 | 4 | 6 |
| `1` | Steady standard demand | 5 | 8 | 11 |
| `2` | Regime-sensitive demand | 2 | 7 | 13 |
| `3` | Premium slow mover | 1 | 3 | 5 |
| `4` | Fast mover | 6 | 10 | 14 |

These are simulator design values, not real business estimates. Step 3 will implement and validate demand generation. Step 5 will finalize product economics and reward weights.

## Actions

The nominal action set is:

```text
actions = [0, 5, 10, 15]
```

Each action is the number of units ordered for the current product. The order becomes `incoming_order` in that product's next-week state.

### Feasible-Action Constraint

The policy may select only actions satisfying:

```text
on_hand + incoming_order + order_quantity <= capacity
```

where capacity is `30` units per product.

This conservative inventory-position constraint guarantees that a requested order cannot cause a capacity violation even if no units sell before it arrives. Step 4 will implement and test action masking as a hard safety control.

## Transition Process

For each product in week `t`, the intended transition order is:

1. Observe the current state.
2. Determine the feasible actions using the capacity constraint.
3. Choose one feasible order quantity.
4. Receive the existing `incoming_order`, making it available for current-week demand.
5. Sample actual demand from the current product profile and observed demand regime.
6. Sell the minimum of available inventory and actual demand.
7. Record unmet demand as stockout units.
8. Set ending on-hand inventory after sales.
9. Move the selected action into the next state's `incoming_order`.
10. Sample the next demand regime from the configured Markov transition matrix.
11. Increment the week.
12. Calculate the per-product reward components for the transition.

The next state is:

```text
next_state = (
    same_product_id,
    week + 1,
    ending_on_hand,
    selected_order_quantity,
    next_demand_regime,
)
```

At the terminal week, no next decision state is used.

## Demand Process

Actual demand is sampled from a Poisson distribution using the product profile and current demand regime. Demand is capped at `20` units per product-week to keep edge cases bounded and testable.

The demand regime evolves using this shared transition matrix:

| Current Regime | Next Low | Next Medium | Next High |
|---|---:|---:|---:|
| Low | 0.65 | 0.30 | 0.05 |
| Medium | 0.15 | 0.70 | 0.15 |
| High | 0.05 | 0.30 | 0.65 |

The regime is observable before the decision; actual sampled demand is not.

## Horizon and Termination

- Horizon: 12 simulated weeks.
- Decisions per episode: `12 weeks × 5 products = 60` product-level transitions.
- Normal termination: after all five product decisions and transitions for week 11.
- Stockout does not terminate an episode.
- Excess inventory does not terminate an episode.
- There is no early-success termination.

Keeping stockout and excess inventory inside the episode allows the controller to experience their delayed consequences.

## Draft Reward Decomposition

Step 2 defines the reward components but deliberately does not finalize their numerical values.

For product `p` in week `t`:

```text
reward =
    sales_revenue
    - procurement_cost
    - holding_cost
    - stockout_penalty
    - terminal_excess_penalty
```

| Component | Draft Definition | Timing |
|---|---|---|
| `sales_revenue` | Units sold multiplied by product selling price | Every transition |
| `procurement_cost` | Selected order quantity multiplied by product unit cost | Every transition |
| `holding_cost` | Ending on-hand inventory multiplied by a holding-cost parameter | Every transition |
| `stockout_penalty` | Unmet demand multiplied by a stockout-cost parameter | Every transition |
| `terminal_excess_penalty` | Remaining and already ordered inventory penalized at episode end | Terminal transition only |

Step 5 must finalize the product economics, penalty weights, scale, and reward-hacking analysis before training.

## Markov Property Assessment

The proposed state is intended to be sufficient for the scoped simulator:

- `product_id` determines the static demand profile.
- `week` captures the finite-horizon position.
- `on_hand` captures currently held inventory.
- `incoming_order` captures the relevant pipeline.
- `demand_regime` determines the current demand distribution and the distribution of the next regime.
- The action determines the next incoming order.

Given these variables and the action, the next-state distribution does not depend on earlier history.

This Markov claim depends on the approved scope. It would no longer hold if later work introduced unobserved seasonality, multi-week pipelines, shared budgets, cross-product substitution, or other hidden dependencies.

## Q-Table Tractability

Nominal state count:

```text
5 product IDs
× 12 weeks
× 31 on-hand inventory values
× 4 incoming-order values
× 3 demand regimes
= 22,320 states
```

Nominal Q-table cells:

```text
22,320 states × 4 actions = 89,280 cells
```

At eight bytes per `float64` value, the dense Q-values require approximately `0.68 MiB`, excluding minor container overhead. The reachable state set will be smaller because capacity constraints make some nominal combinations impossible.

This is comfortably tractable for tabular Q-learning and the assignment's laptop compute limit.

## Open Review Decisions

1. Approve exact inventory rather than bucketed inventory.
2. Approve inclusion of `product_id` and `week` in the state.
3. Approve the four order quantities: `0`, `5`, `10`, and `15`.
4. Approve a capacity of `30` units per product.
5. Approve the conservative feasible-action constraint based on inventory position.
6. Approve a 12-week episode with 60 product-level transitions.
7. Approve the five demand profiles and shared regime-transition matrix.
8. Approve the draft reward components while leaving numerical values for Step 5.
