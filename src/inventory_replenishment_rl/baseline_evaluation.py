"""Fair seeded evaluation for non-learning baseline policies."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import pandas as pd

from inventory_replenishment_rl.policies import BaselinePolicy
from inventory_replenishment_rl.simulator import (
    InventorySimulator,
    SimulatorConfig,
    TransitionRecord,
)

PolicyFactory = Callable[[int], BaselinePolicy]


def evaluate_policy_episode(
    *,
    policy: BaselinePolicy,
    environment_seed: int,
) -> list[TransitionRecord]:
    simulator = InventorySimulator(seed=environment_seed)
    transitions: list[TransitionRecord] = []

    while not simulator.terminated:
        actions = policy.select_actions(simulator)
        transitions.extend(simulator.step_week(actions))
    return transitions


def episode_metrics(
    *,
    policy_name: str,
    environment_seed: int,
    policy_seed: int,
    transitions: list[TransitionRecord],
    capacity_per_product: int,
) -> dict[str, float | int | str]:
    total_demand = sum(record.actual_demand for record in transitions)
    total_sales = sum(record.reward_drivers.sales_units for record in transitions)
    total_stockouts = sum(record.reward_drivers.stockout_units for record in transitions)
    total_ordered = sum(record.reward_drivers.ordered_units for record in transitions)
    total_holding = sum(record.reward_drivers.holding_units for record in transitions)
    total_terminal_excess = sum(
        record.reward_drivers.terminal_excess_units for record in transitions
    )
    stockout_events = sum(record.reward_drivers.stockout_units > 0 for record in transitions)

    return {
        "policy": policy_name,
        "environment_seed": environment_seed,
        "policy_seed": policy_seed,
        "transitions": len(transitions),
        "total_demand": total_demand,
        "total_sales": total_sales,
        "total_stockout_units": total_stockouts,
        "stockout_events": stockout_events,
        "service_level": total_sales / total_demand if total_demand else 1.0,
        "total_ordered_units": total_ordered,
        "average_holding_units": total_holding / len(transitions),
        "terminal_excess_units": total_terminal_excess,
        "capacity_violations": sum(
            record.state.on_hand + record.state.incoming_order + record.action
            > capacity_per_product
            for record in transitions
        ),
    }


def evaluate_baselines(
    policy_factories: dict[str, PolicyFactory],
    environment_seeds: Iterable[int],
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    capacity_per_product = SimulatorConfig.from_json().capacity_per_product
    for environment_seed in environment_seeds:
        for policy_offset, (policy_name, factory) in enumerate(policy_factories.items()):
            policy_seed = environment_seed * 10 + policy_offset
            policy = factory(policy_seed)
            transitions = evaluate_policy_episode(
                policy=policy,
                environment_seed=environment_seed,
            )
            rows.append(
                episode_metrics(
                    policy_name=policy_name,
                    environment_seed=environment_seed,
                    policy_seed=policy_seed,
                    transitions=transitions,
                    capacity_per_product=capacity_per_product,
                )
            )
    return pd.DataFrame(rows)


def summarize_baselines(episode_results: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "total_sales",
        "total_stockout_units",
        "stockout_events",
        "service_level",
        "total_ordered_units",
        "average_holding_units",
        "terminal_excess_units",
        "capacity_violations",
    ]
    summary = episode_results.groupby("policy")[metrics].agg(["mean", "std"]).reset_index()
    summary.columns = [
        column if isinstance(column, str) else "_".join(part for part in column if part)
        for column in summary.columns
    ]
    return summary
