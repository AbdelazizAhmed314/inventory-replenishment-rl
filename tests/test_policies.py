from __future__ import annotations

from inventory_replenishment_rl.baseline_evaluation import (
    evaluate_baselines,
    evaluate_policy_episode,
)
from inventory_replenishment_rl.policies import OrderUpToPolicy, RandomFeasiblePolicy
from inventory_replenishment_rl.simulator import InventorySimulator, InventoryState


def test_action_mask_matches_feasible_actions() -> None:
    simulator = InventorySimulator(seed=42)
    states = simulator.states
    states[0] = InventoryState(0, 0, 20, 5, "medium")
    simulator.reset(seed=42, initial_states=states)

    assert simulator.feasible_actions(0) == (0, 5)
    assert simulator.action_mask(0) == (True, True, False, False)


def test_random_policy_is_reproducible_and_always_feasible() -> None:
    first_simulator = InventorySimulator(seed=42)
    second_simulator = InventorySimulator(seed=42)
    first_policy = RandomFeasiblePolicy(seed=99)
    second_policy = RandomFeasiblePolicy(seed=99)

    first_actions = first_policy.select_actions(first_simulator)
    second_actions = second_policy.select_actions(second_simulator)

    assert first_actions == second_actions
    assert all(
        action in first_simulator.feasible_actions(product_id)
        for product_id, action in first_actions.items()
    )


def test_rule_policy_uses_product_and_regime_demand_profile() -> None:
    simulator = InventorySimulator(seed=42)
    states = {
        product_id: InventoryState(product_id, 0, 0, 0, "medium") for product_id in simulator.states
    }
    simulator.reset(seed=42, initial_states=states)

    actions = OrderUpToPolicy().select_actions(simulator)

    assert actions == {
        0: 10,
        1: 15,
        2: 15,
        3: 10,
        4: 15,
    }


def test_rule_policy_respects_capacity_action_mask() -> None:
    simulator = InventorySimulator(seed=42)
    states = simulator.states
    states[4] = InventoryState(4, 0, 20, 10, "high")
    simulator.reset(seed=42, initial_states=states)

    actions = OrderUpToPolicy().select_actions(simulator)

    assert actions[4] == 0
    assert actions[4] in simulator.feasible_actions(4)


def test_baseline_episode_contains_no_capacity_violations() -> None:
    simulator_config = InventorySimulator(seed=0).config
    transitions = evaluate_policy_episode(
        policy=OrderUpToPolicy(),
        environment_seed=123,
    )

    assert len(transitions) == 60
    assert all(
        record.state.on_hand + record.state.incoming_order + record.action
        <= simulator_config.capacity_per_product
        for record in transitions
    )
    assert all(record.action in simulator_config.order_quantities for record in transitions)


def test_baseline_evaluation_uses_identical_environment_seeds() -> None:
    results = evaluate_baselines(
        {
            "random_feasible": lambda seed: RandomFeasiblePolicy(seed=seed),
            "rule_order_up_to": lambda _seed: OrderUpToPolicy(),
        },
        environment_seeds=(100, 101, 102),
    )

    seed_counts = results.groupby("environment_seed")["policy"].nunique()
    assert len(results) == 6
    assert seed_counts.eq(2).all()
    assert results["capacity_violations"].eq(0).all()


def test_same_environment_seed_produces_same_demand_and_regime_path() -> None:
    random_transitions = evaluate_policy_episode(
        policy=RandomFeasiblePolicy(seed=999),
        environment_seed=123,
    )
    rule_transitions = evaluate_policy_episode(
        policy=OrderUpToPolicy(),
        environment_seed=123,
    )

    def stochastic_path(transitions):
        return [
            (
                record.state.product_id,
                record.state.week,
                record.state.demand_regime,
                record.actual_demand,
                record.next_state.demand_regime if record.next_state else None,
            )
            for record in transitions
        ]

    assert stochastic_path(random_transitions) == stochastic_path(rule_transitions)
