from __future__ import annotations

import pytest

from inventory_replenishment_rl.simulator import InventorySimulator, InventoryState


def zero_actions(simulator: InventorySimulator) -> dict[int, int]:
    return {product_id: 0 for product_id in simulator.states}


def medium_regimes(simulator: InventorySimulator) -> dict[int, str]:
    return {product_id: "medium" for product_id in simulator.states}


def test_reset_produces_five_approved_initial_states() -> None:
    simulator = InventorySimulator(seed=42)

    assert simulator.current_week == 0
    assert simulator.terminated is False
    assert len(simulator.states) == 5
    assert all(state.week == 0 for state in simulator.states.values())
    assert all(state.on_hand == 15 for state in simulator.states.values())
    assert all(state.incoming_order == 0 for state in simulator.states.values())
    assert all(state.demand_regime == "medium" for state in simulator.states.values())


def test_observation_state_excludes_future_information_and_policy_values() -> None:
    assert tuple(InventoryState.__dataclass_fields__) == (
        "product_id",
        "week",
        "on_hand",
        "incoming_order",
        "demand_regime",
    )


def test_known_transition_receives_pipeline_before_demand_and_logs_reward_drivers() -> None:
    simulator = InventorySimulator(seed=42)
    initial_states = {
        product_id: InventoryState(product_id, 0, 15, 0, "medium")
        for product_id in simulator.states
    }
    initial_states[0] = InventoryState(0, 0, 10, 5, "medium")
    simulator.reset(seed=42, initial_states=initial_states)

    actions = zero_actions(simulator)
    actions[0] = 10
    demands = {product_id: 0 for product_id in simulator.states}
    demands[0] = 12
    transition = simulator.step_week(
        actions,
        actual_demand_overrides=demands,
        next_regime_overrides=medium_regimes(simulator),
    )[0]

    assert transition.available_inventory == 15
    assert transition.actual_demand == 12
    assert transition.reward_drivers.sales_units == 12
    assert transition.reward_drivers.stockout_units == 0
    assert transition.reward_drivers.holding_units == 3
    assert transition.reward_drivers.ordered_units == 10
    assert transition.reward_drivers.terminal_excess_units == 0
    assert transition.next_state == InventoryState(0, 1, 3, 10, "medium")


def test_infeasible_action_is_rejected_before_any_state_changes() -> None:
    simulator = InventorySimulator(seed=42)
    initial_states = {
        product_id: InventoryState(product_id, 0, 15, 0, "medium")
        for product_id in simulator.states
    }
    initial_states[0] = InventoryState(0, 0, 20, 10, "medium")
    simulator.reset(seed=42, initial_states=initial_states)
    before = simulator.states
    actions = zero_actions(simulator)
    actions[0] = 5

    with pytest.raises(ValueError, match="Infeasible order quantity"):
        simulator.step_week(actions)

    assert simulator.current_week == 0
    assert simulator.states == before


def test_unsupported_action_is_rejected_before_any_state_changes() -> None:
    simulator = InventorySimulator(seed=42)
    before = simulator.states
    actions = zero_actions(simulator)
    actions[0] = 20

    with pytest.raises(ValueError, match="Unsupported order quantity"):
        simulator.step_week(actions)

    assert simulator.current_week == 0
    assert simulator.states == before


def test_episode_terminates_after_twelve_weeks_without_early_stockout_termination() -> None:
    simulator = InventorySimulator(seed=42)
    transitions = []
    for _ in range(12):
        transitions.extend(
            simulator.step_week(
                zero_actions(simulator),
                actual_demand_overrides={product_id: 20 for product_id in simulator.states},
                next_regime_overrides=medium_regimes(simulator),
            )
        )

    assert len(transitions) == 60
    assert simulator.current_week == 12
    assert simulator.terminated is True
    assert sum(record.terminated for record in transitions) == 5
    assert all(record.next_state is None for record in transitions[-5:])

    with pytest.raises(RuntimeError, match="terminated episode"):
        simulator.step_week(zero_actions(simulator))


def test_front_loaded_max_order_produces_holding_and_terminal_excess_drivers() -> None:
    simulator = InventorySimulator(seed=42)
    maximum_actions = {product_id: 15 for product_id in simulator.states}
    transitions = simulator.step_week(
        maximum_actions,
        actual_demand_overrides={product_id: 0 for product_id in simulator.states},
        next_regime_overrides=medium_regimes(simulator),
    )
    assert all(record.reward_drivers.holding_units == 15 for record in transitions)
    assert all(record.reward_drivers.ordered_units == 15 for record in transitions)

    for _ in range(11):
        transitions = simulator.step_week(
            zero_actions(simulator),
            actual_demand_overrides={product_id: 0 for product_id in simulator.states},
            next_regime_overrides=medium_regimes(simulator),
        )

    assert all(record.reward_drivers.holding_units == 30 for record in transitions)
    assert all(record.reward_drivers.terminal_excess_units == 30 for record in transitions)


def test_seeded_episodes_are_reproducible_and_different_seeds_differ() -> None:
    first = InventorySimulator(seed=123)
    second = InventorySimulator(seed=123)
    different = InventorySimulator(seed=456)

    first_records = first.run_action_schedule([zero_actions(first) for _ in range(12)])
    second_records = second.run_action_schedule([zero_actions(second) for _ in range(12)])
    different_records = different.run_action_schedule([zero_actions(different) for _ in range(12)])

    assert first_records == second_records
    assert first_records != different_records
    assert all(0 <= record.actual_demand <= 20 for record in first_records)


def test_scenario_overrides_do_not_consume_random_draws() -> None:
    controlled = InventorySimulator(seed=123)
    reference = InventorySimulator(seed=123)
    actions = zero_actions(controlled)
    demands = {product_id: 0 for product_id in controlled.states}
    regimes = medium_regimes(controlled)

    controlled.step_week(
        actions,
        actual_demand_overrides=demands,
        next_regime_overrides=regimes,
    )

    assert controlled._rng.bit_generator.state == reference._rng.bit_generator.state
