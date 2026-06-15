"""Frozen-policy evaluation utilities for the inventory replenishment simulator."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from typing import Protocol

import pandas as pd

from inventory_replenishment_rl.q_learning import QLearningAgent
from inventory_replenishment_rl.simulator import (
    InventorySimulator,
    SimulatorConfig,
    TransitionRecord,
)


class EvaluationPolicy(Protocol):
    name: str

    def select_actions(self, simulator: InventorySimulator) -> dict[int, int]:
        """Select one feasible action for each product without learning."""


PolicyFactory = Callable[[int], EvaluationPolicy]


@dataclass(frozen=True)
class EvaluationScenario:
    name: str
    demand_mode: str = "stochastic"
    procurement_cost_per_unit: float | None = None
    holding_cost_per_unit_week: float | None = None

    def validate(self) -> None:
        if self.demand_mode not in {"stochastic", "zero", "maximum"}:
            raise ValueError(f"Unsupported demand mode: {self.demand_mode}")


class FrozenQLearningPolicy:
    """Deterministic greedy policy backed by a trained Q-table."""

    name = "q_learning_frozen"

    def __init__(self, agent: QLearningAgent) -> None:
        self.agent = agent

    def select_actions(self, simulator: InventorySimulator) -> dict[int, int]:
        return self.agent.greedy_actions(simulator)


def config_for_scenario(
    base_config: SimulatorConfig,
    scenario: EvaluationScenario,
) -> SimulatorConfig:
    scenario.validate()
    reward = base_config.reward
    reward_updates: dict[str, float] = {}
    if scenario.procurement_cost_per_unit is not None:
        reward_updates["procurement_cost_per_unit"] = scenario.procurement_cost_per_unit
    if scenario.holding_cost_per_unit_week is not None:
        reward_updates["holding_cost_per_unit_week"] = scenario.holding_cost_per_unit_week
    if reward_updates:
        reward = replace(reward, **reward_updates)
    return replace(base_config, reward=reward)


def demand_overrides(
    simulator: InventorySimulator,
    scenario: EvaluationScenario,
) -> dict[int, int] | None:
    if scenario.demand_mode == "stochastic":
        return None
    value = 0 if scenario.demand_mode == "zero" else simulator.config.maximum_realized_demand
    return {product_id: value for product_id in simulator.states}


def evaluate_episode(
    *,
    policy: EvaluationPolicy,
    environment_seed: int,
    scenario: EvaluationScenario,
    base_config: SimulatorConfig | None = None,
) -> list[TransitionRecord]:
    config = config_for_scenario(base_config or SimulatorConfig.from_json(), scenario)
    simulator = InventorySimulator(config=config, seed=environment_seed)
    transitions: list[TransitionRecord] = []
    while not simulator.terminated:
        transitions.extend(
            simulator.step_week(
                policy.select_actions(simulator),
                actual_demand_overrides=demand_overrides(simulator, scenario),
            )
        )
    return transitions


def transition_metrics(
    transitions: list[TransitionRecord],
    *,
    capacity_per_product: int,
    terminal_week: int,
) -> dict[str, float | int]:
    total_demand = sum(record.actual_demand for record in transitions)
    total_sales = sum(record.reward_drivers.sales_units for record in transitions)
    total_stockouts = sum(record.reward_drivers.stockout_units for record in transitions)
    total_holding = sum(record.reward_drivers.holding_units for record in transitions)
    return {
        "transitions": len(transitions),
        "total_scalar_reward": sum(record.scalar_reward for record in transitions),
        "total_demand": total_demand,
        "total_sales": total_sales,
        "total_stockout_units": total_stockouts,
        "stockout_events": sum(record.reward_drivers.stockout_units > 0 for record in transitions),
        "service_level": total_sales / total_demand if total_demand else 1.0,
        "total_ordered_units": sum(record.reward_drivers.ordered_units for record in transitions),
        "average_holding_units": total_holding / len(transitions),
        "terminal_excess_units": sum(
            record.reward_drivers.terminal_excess_units for record in transitions
        ),
        "final_week_ordered_units": sum(
            record.reward_drivers.ordered_units
            for record in transitions
            if record.state.week == terminal_week
        ),
        "capacity_violations": sum(
            record.state.on_hand + record.state.incoming_order + record.action
            > capacity_per_product
            for record in transitions
        ),
    }


def unvisited_state_decisions(
    agent: QLearningAgent,
    transitions: list[TransitionRecord],
) -> int:
    count = 0
    for record in transitions:
        state_index = agent.encoder.state_index(record.state)
        feasible_indices = agent.encoder.feasible_action_indices(
            tuple(
                action
                for action in agent.encoder.actions
                if record.state.on_hand + record.state.incoming_order + action
                <= agent.encoder.capacity
            )
        )
        count += int(agent.visit_counts[state_index][list(feasible_indices)].sum() == 0)
    return count


def evaluate_policies(
    *,
    policy_factories: dict[str, PolicyFactory],
    environment_seeds: Iterable[int],
    scenario: EvaluationScenario,
    learned_agent: QLearningAgent | None = None,
    base_config: SimulatorConfig | None = None,
) -> pd.DataFrame:
    config = base_config or SimulatorConfig.from_json()
    rows: list[dict[str, float | int | str]] = []
    for environment_seed in environment_seeds:
        for policy_offset, (policy_name, factory) in enumerate(policy_factories.items()):
            policy_seed = environment_seed * 10 + policy_offset
            transitions = evaluate_episode(
                policy=factory(policy_seed),
                environment_seed=environment_seed,
                scenario=scenario,
                base_config=config,
            )
            row: dict[str, float | int | str] = {
                "scenario": scenario.name,
                "policy": policy_name,
                "environment_seed": environment_seed,
                "policy_seed": policy_seed,
                **transition_metrics(
                    transitions,
                    capacity_per_product=config.capacity_per_product,
                    terminal_week=config.episode_weeks - 1,
                ),
                "unvisited_state_decisions": 0,
            }
            if policy_name == FrozenQLearningPolicy.name and learned_agent is not None:
                row["unvisited_state_decisions"] = unvisited_state_decisions(
                    learned_agent,
                    transitions,
                )
            rows.append(row)
    return pd.DataFrame(rows)


def product_episode_metrics(
    *,
    policy_name: str,
    environment_seed: int,
    transitions: list[TransitionRecord],
    capacity_per_product: int,
    terminal_week: int,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    product_ids = sorted({record.state.product_id for record in transitions})
    for product_id in product_ids:
        product_records = [
            record for record in transitions if record.state.product_id == product_id
        ]
        rows.append(
            {
                "policy": policy_name,
                "environment_seed": environment_seed,
                "product_id": product_id,
                "product_name": product_records[0].product_name,
                **transition_metrics(
                    product_records,
                    capacity_per_product=capacity_per_product,
                    terminal_week=terminal_week,
                ),
            }
        )
    return rows


def summarize_results(results: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    metrics = [
        "total_scalar_reward",
        "total_sales",
        "total_stockout_units",
        "stockout_events",
        "service_level",
        "total_ordered_units",
        "average_holding_units",
        "terminal_excess_units",
        "final_week_ordered_units",
        "unvisited_state_decisions",
        "capacity_violations",
    ]
    available_metrics = [metric for metric in metrics if metric in results.columns]
    summary = results.groupby(group_columns)[available_metrics].agg(["mean", "std"]).reset_index()
    summary.columns = [
        column if isinstance(column, str) else "_".join(part for part in column if part)
        for column in summary.columns
    ]
    return summary
