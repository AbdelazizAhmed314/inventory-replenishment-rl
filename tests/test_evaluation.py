from __future__ import annotations

from dataclasses import replace

import pandas as pd

from inventory_replenishment_rl.evaluation import (
    EvaluationScenario,
    FrozenQLearningPolicy,
    config_for_scenario,
    evaluate_episode,
    evaluate_policies,
    summarize_results,
)
from inventory_replenishment_rl.policies import OrderUpToPolicy, RandomFeasiblePolicy
from inventory_replenishment_rl.q_learning import QLearningAgent
from inventory_replenishment_rl.simulator import SimulatorConfig


def test_frozen_policy_does_not_change_q_table() -> None:
    config = SimulatorConfig.from_json()
    agent = QLearningAgent(simulator_config=config, seed=42)
    before = agent.q_values.copy()

    transitions = evaluate_episode(
        policy=FrozenQLearningPolicy(agent),
        environment_seed=200000,
        scenario=EvaluationScenario(name="standard"),
        base_config=config,
    )

    assert len(transitions) == 60
    assert (agent.q_values == before).all()
    assert all(record.action in config.order_quantities for record in transitions)


def test_scenario_cost_override_changes_reward_without_changing_base_config() -> None:
    config = SimulatorConfig.from_json()
    scenario = EvaluationScenario(name="high_cost", procurement_cost_per_unit=6.0)

    changed = config_for_scenario(config, scenario)

    assert changed.reward.procurement_cost_per_unit == 6.0
    assert config.reward.procurement_cost_per_unit == 2.0
    assert replace(changed.reward, procurement_cost_per_unit=2.0) == config.reward


def test_identical_environment_seed_produces_identical_demand_across_policies() -> None:
    scenario = EvaluationScenario(name="standard")
    results = evaluate_policies(
        policy_factories={
            "rule_order_up_to": lambda _seed: OrderUpToPolicy(),
            "random_feasible": lambda seed: RandomFeasiblePolicy(seed=seed),
        },
        environment_seeds=(200000, 200001),
        scenario=scenario,
    )

    assert results.groupby("environment_seed")["total_demand"].nunique().eq(1).all()
    assert results["capacity_violations"].eq(0).all()


def test_maximum_demand_scenario_is_controlled_and_summary_reports_variability() -> None:
    scenario = EvaluationScenario(name="spike", demand_mode="maximum")
    results = evaluate_policies(
        policy_factories={"rule_order_up_to": lambda _seed: OrderUpToPolicy()},
        environment_seeds=(210000, 210001),
        scenario=scenario,
    )
    summary = summarize_results(results, ["scenario", "policy"])

    assert results["total_demand"].eq(1200).all()
    assert isinstance(summary, pd.DataFrame)
    assert "total_scalar_reward_std" in summary.columns


def test_summary_accepts_product_metrics_without_episode_only_columns() -> None:
    results = pd.DataFrame([{"policy": "rule", "product_id": 0, "total_scalar_reward": 10.0}])

    summary = summarize_results(results, ["policy", "product_id"])

    assert summary.loc[0, "total_scalar_reward_mean"] == 10.0
    assert "unvisited_state_decisions_mean" not in summary.columns
