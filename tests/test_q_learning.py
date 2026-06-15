from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from inventory_replenishment_rl.q_learning import (
    QLearningAgent,
    StateActionEncoder,
    TrainingConfig,
    epsilon_for_episode,
    q_table_frame,
    train_q_learning,
)
from inventory_replenishment_rl.simulator import InventoryState, SimulatorConfig


def small_training_config() -> TrainingConfig:
    return TrainingConfig(
        algorithm="tabular_q_learning",
        episodes=5,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon_schedule="linear",
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_fraction=0.8,
        agent_seed=697,
        environment_seed_start=100000,
        q_initial_value=0.0,
        rolling_window=2,
    )


def test_encoder_matches_approved_q_table_shape() -> None:
    encoder = StateActionEncoder(SimulatorConfig.from_json())
    state = InventoryState(4, 11, 30, 0, "high")

    assert encoder.q_shape == (5, 12, 31, 4, 3, 4)
    assert int(np.prod(encoder.q_shape)) == 89_280
    assert encoder.state_index(state) == (4, 11, 30, 0, 2)
    assert encoder.action_index(15) == 3


def test_epsilon_schedule_decays_and_retains_exploration() -> None:
    config = small_training_config()

    assert epsilon_for_episode(0, config) == 1.0
    assert epsilon_for_episode(2, config) == 0.525
    assert epsilon_for_episode(4, config) == 0.05
    assert epsilon_for_episode(10, config) == 0.05


def test_epsilon_greedy_never_selects_infeasible_action() -> None:
    simulator_config = SimulatorConfig.from_json()
    agent = QLearningAgent(simulator_config=simulator_config, seed=42)
    state = InventoryState(0, 0, 20, 5, "medium")
    state_index = agent.encoder.state_index(state)
    agent.q_values[(*state_index, agent.encoder.action_index(15))] = 9999

    exploratory = {
        agent.choose_action(state=state, feasible_actions=(0, 5), epsilon=1.0).action
        for _ in range(100)
    }
    greedy = agent.greedy_action(state=state, feasible_actions=(0, 5))

    assert exploratory == {0, 5}
    assert greedy == 0


def test_q_learning_update_matches_td_equation_and_feasible_next_maximum() -> None:
    config = SimulatorConfig.from_json()
    agent = QLearningAgent(simulator_config=config, seed=42)
    state = InventoryState(0, 0, 10, 0, "medium")
    next_state = InventoryState(0, 1, 5, 10, "high")
    next_index = agent.encoder.state_index(next_state)
    agent.q_values[(*next_index, agent.encoder.action_index(0))] = 4.0
    agent.q_values[(*next_index, agent.encoder.action_index(5))] = 8.0
    agent.q_values[(*next_index, agent.encoder.action_index(15))] = 1000.0

    update = agent.update(
        state=state,
        action=10,
        reward=20.0,
        next_state=next_state,
        next_feasible_actions=(0, 5),
        terminated=False,
        learning_rate=0.1,
        discount_factor=0.9,
    )

    assert update.td_target == 27.2
    assert update.td_error == 27.2
    assert update.new_value == 2.72
    assert agent.visited_state_action_pairs == 1


def test_terminal_update_does_not_bootstrap() -> None:
    config = SimulatorConfig.from_json()
    agent = QLearningAgent(simulator_config=config, seed=42)
    state = InventoryState(0, 11, 5, 0, "medium")

    update = agent.update(
        state=state,
        action=0,
        reward=-25.0,
        next_state=None,
        next_feasible_actions=(),
        terminated=True,
        learning_rate=0.2,
        discount_factor=0.99,
    )

    assert update.td_target == -25.0
    assert update.new_value == -5.0


def test_tiny_training_run_is_reproducible_and_logs_business_metrics() -> None:
    config = small_training_config()

    first_agent, first_history = train_q_learning(config)
    second_agent, second_history = train_q_learning(config)

    np.testing.assert_array_equal(first_agent.q_values, second_agent.q_values)
    np.testing.assert_array_equal(first_agent.visit_counts, second_agent.visit_counts)
    pd.testing.assert_frame_equal(first_history, second_history)
    assert {
        "episode_reward",
        "stockout_units",
        "holding_units",
        "terminal_excess_units",
        "mean_absolute_td_error",
    }.issubset(first_history.columns)


def test_saved_q_table_round_trip(tmp_path) -> None:
    config = small_training_config()
    first_agent, _ = train_q_learning(config)

    path = tmp_path / "q_table.csv"
    q_table_frame(first_agent).sample(frac=1, random_state=42).to_csv(path, index=False)
    loaded_agent = QLearningAgent.from_q_table_csv(
        simulator_config=SimulatorConfig.from_json(),
        path=path,
        seed=999,
    )

    np.testing.assert_allclose(first_agent.q_values, loaded_agent.q_values, rtol=1e-15, atol=1e-15)
    np.testing.assert_array_equal(first_agent.visit_counts, loaded_agent.visit_counts)
    assert loaded_agent.visited_state_action_pairs == first_agent.visited_state_action_pairs


def test_non_terminal_update_requires_next_state() -> None:
    agent = QLearningAgent(simulator_config=SimulatorConfig.from_json(), seed=42)

    with pytest.raises(ValueError, match="Non-terminal updates require"):
        agent.update(
            state=InventoryState(0, 0, 10, 0, "medium"),
            action=0,
            reward=0.0,
            next_state=None,
            next_feasible_actions=(),
            terminated=False,
            learning_rate=0.1,
            discount_factor=0.99,
        )
