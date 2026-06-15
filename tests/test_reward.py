from __future__ import annotations

from inventory_replenishment_rl.reward import calculate_reward
from inventory_replenishment_rl.simulator import InventorySimulator, InventoryState


def zero_actions(simulator: InventorySimulator) -> dict[int, int]:
    return {product_id: 0 for product_id in simulator.states}


def test_reward_configuration_is_finalized_and_transparent() -> None:
    reward = InventorySimulator(seed=0).config.reward

    assert reward.units == "simulated_monetary_units"
    assert reward.sales_revenue_per_unit == 10.0
    assert reward.procurement_cost_per_unit == 2.0
    assert reward.holding_cost_per_unit_week == 1.0
    assert reward.stockout_penalty_per_unit == 15.0
    assert reward.terminal_excess_penalty_per_unit == 5.0
    assert reward.initial_inventory_cost_treatment == "sunk_cost_excluded"


def test_reward_breakdown_matches_approved_equation() -> None:
    config = InventorySimulator(seed=0).config.reward

    reward = calculate_reward(
        config=config,
        sales_units=12,
        ordered_units=10,
        holding_units=3,
        stockout_units=2,
        terminal_excess_units=4,
    )

    assert reward.sales_revenue == 120.0
    assert reward.procurement_cost == 20.0
    assert reward.holding_cost == 3.0
    assert reward.stockout_penalty == 30.0
    assert reward.terminal_excess_penalty == 20.0
    assert reward.scalar_reward == 47.0


def test_simulator_logs_scalar_reward_and_components() -> None:
    simulator = InventorySimulator(seed=42)
    initial_states = {
        product_id: InventoryState(product_id, 0, 15, 0, "medium")
        for product_id in simulator.states
    }
    initial_states[0] = InventoryState(0, 0, 10, 5, "medium")
    simulator.reset(seed=42, initial_states=initial_states)
    actions = zero_actions(simulator)
    actions[0] = 10

    transition = simulator.step_week(
        actions,
        actual_demand_overrides={product_id: 0 for product_id in simulator.states} | {0: 12},
        next_regime_overrides={product_id: "medium" for product_id in simulator.states},
    )[0]
    row = transition.to_flat_dict()

    assert transition.scalar_reward == 97.0
    assert row["sales_revenue"] == 120.0
    assert row["procurement_cost"] == 20.0
    assert row["holding_cost"] == 3.0
    assert row["scalar_reward"] == 97.0
    assert row["reward_status"] == "finalized_step_5"


def test_ordering_nothing_does_not_escape_high_demand_cost() -> None:
    simulator = InventorySimulator(seed=42)
    transitions = []
    while not simulator.terminated:
        transitions.extend(
            simulator.step_week(
                zero_actions(simulator),
                actual_demand_overrides={product_id: 20 for product_id in simulator.states},
                next_regime_overrides={product_id: "medium" for product_id in simulator.states},
            )
        )

    assert sum(record.scalar_reward for record in transitions) < 0
    assert sum(record.reward_drivers.stockout_units for record in transitions) > 0


def test_over_ordering_does_not_escape_holding_and_terminal_costs() -> None:
    simulator = InventorySimulator(seed=42)
    transitions = []
    while not simulator.terminated:
        actions = {
            product_id: max(simulator.feasible_actions(product_id))
            for product_id in simulator.states
        }
        transitions.extend(
            simulator.step_week(
                actions,
                actual_demand_overrides={product_id: 0 for product_id in simulator.states},
                next_regime_overrides={product_id: "medium" for product_id in simulator.states},
            )
        )

    assert sum(record.scalar_reward for record in transitions) < 0
    assert sum(record.reward_drivers.holding_units for record in transitions) > 0
    assert sum(record.reward_drivers.terminal_excess_units for record in transitions) > 0


def test_final_week_order_is_charged_as_terminal_excess() -> None:
    simulator = InventorySimulator(seed=42)
    high_demand = {product_id: 20 for product_id in simulator.states}
    medium_regimes = {product_id: "medium" for product_id in simulator.states}
    for _ in range(11):
        simulator.step_week(
            zero_actions(simulator),
            actual_demand_overrides=high_demand,
            next_regime_overrides=medium_regimes,
        )

    actions = zero_actions(simulator)
    actions[0] = 15
    final_transition = simulator.step_week(
        actions,
        actual_demand_overrides={product_id: 0 for product_id in simulator.states},
    )[0]

    assert final_transition.reward_drivers.terminal_excess_units == 15
    assert final_transition.reward_breakdown.procurement_cost == 30.0
    assert final_transition.reward_breakdown.terminal_excess_penalty == 75.0
    assert final_transition.scalar_reward == -105.0
