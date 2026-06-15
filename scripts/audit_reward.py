"""Generate deterministic pre-training reward-hacking audit evidence."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd

from inventory_replenishment_rl.policies import OrderUpToPolicy, RandomFeasiblePolicy
from inventory_replenishment_rl.simulator import InventorySimulator, TransitionRecord

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_DIR = ROOT / "evidence"
OUTPUT_PATH = EVIDENCE_DIR / "reward_scenario_checks.csv"
STOCHASTIC_SEEDS = tuple(range(1000, 1050))
ActionSelector = Callable[[InventorySimulator], dict[int, int]]


def zero_actions(simulator: InventorySimulator) -> dict[int, int]:
    return {product_id: 0 for product_id in simulator.states}


def maximum_feasible_actions(simulator: InventorySimulator) -> dict[int, int]:
    return {
        product_id: max(simulator.feasible_actions(product_id)) for product_id in simulator.states
    }


def run_episode(
    *,
    seed: int,
    action_selector: ActionSelector,
    demand_by_product: dict[int, int] | None = None,
) -> list[TransitionRecord]:
    simulator = InventorySimulator(seed=seed)
    transitions: list[TransitionRecord] = []
    while not simulator.terminated:
        product_ids = simulator.states
        demand_overrides = demand_by_product if demand_by_product is not None else None
        regime_overrides = (
            {product_id: "medium" for product_id in product_ids}
            if demand_by_product is not None
            else None
        )
        transitions.extend(
            simulator.step_week(
                action_selector(simulator),
                actual_demand_overrides=demand_overrides,
                next_regime_overrides=regime_overrides,
            )
        )
    return transitions


def episode_totals(transitions: list[TransitionRecord]) -> dict[str, float | int]:
    capacity = InventorySimulator(seed=0).config.capacity_per_product
    return {
        "scalar_reward": sum(record.scalar_reward for record in transitions),
        "sales_units": sum(record.reward_drivers.sales_units for record in transitions),
        "stockout_units": sum(record.reward_drivers.stockout_units for record in transitions),
        "holding_units": sum(record.reward_drivers.holding_units for record in transitions),
        "terminal_excess_units": sum(
            record.reward_drivers.terminal_excess_units for record in transitions
        ),
        "ordered_units": sum(record.reward_drivers.ordered_units for record in transitions),
        "capacity_violations": sum(
            record.state.on_hand + record.state.incoming_order + record.action > capacity
            for record in transitions
        ),
    }


def scenario_row(
    *,
    scenario_name: str,
    scenario_type: str,
    episode_results: list[dict[str, float | int]],
    review_expectation: str,
) -> dict[str, float | int | str | bool]:
    frame = pd.DataFrame(episode_results)
    return {
        "scenario_name": scenario_name,
        "scenario_type": scenario_type,
        "episodes": len(frame),
        "mean_scalar_reward": frame["scalar_reward"].mean(),
        "mean_sales_units": frame["sales_units"].mean(),
        "mean_stockout_units": frame["stockout_units"].mean(),
        "mean_holding_units": frame["holding_units"].mean(),
        "mean_terminal_excess_units": frame["terminal_excess_units"].mean(),
        "mean_ordered_units": frame["ordered_units"].mean(),
        "capacity_violations": int(frame["capacity_violations"].sum()),
        "review_expectation": review_expectation,
        "passed": False,
    }


def build_audit_frame() -> pd.DataFrame:
    reference_simulator = InventorySimulator(seed=0)
    medium_profile_demand = {
        product.product_id: int(product.demand_means["medium"])
        for product in reference_simulator.config.products
    }
    high_demand = {
        product_id: reference_simulator.config.maximum_realized_demand
        for product_id in reference_simulator.states
    }
    zero_demand = {product_id: 0 for product_id in reference_simulator.states}

    controlled_rule = episode_totals(
        run_episode(
            seed=501,
            action_selector=OrderUpToPolicy().select_actions,
            demand_by_product=medium_profile_demand,
        )
    )
    controlled_zero = episode_totals(
        run_episode(seed=502, action_selector=zero_actions, demand_by_product=high_demand)
    )
    controlled_maximum = episode_totals(
        run_episode(
            seed=503,
            action_selector=maximum_feasible_actions,
            demand_by_product=zero_demand,
        )
    )

    stochastic_results: dict[str, list[dict[str, float | int]]] = {
        "stochastic_rule_order_up_to": [],
        "stochastic_all_zero_orders": [],
        "stochastic_maximum_feasible_orders": [],
        "stochastic_random_feasible": [],
    }
    for seed in STOCHASTIC_SEEDS:
        stochastic_results["stochastic_rule_order_up_to"].append(
            episode_totals(run_episode(seed=seed, action_selector=OrderUpToPolicy().select_actions))
        )
        stochastic_results["stochastic_all_zero_orders"].append(
            episode_totals(run_episode(seed=seed, action_selector=zero_actions))
        )
        stochastic_results["stochastic_maximum_feasible_orders"].append(
            episode_totals(run_episode(seed=seed, action_selector=maximum_feasible_actions))
        )
        stochastic_results["stochastic_random_feasible"].append(
            episode_totals(
                run_episode(
                    seed=seed,
                    action_selector=RandomFeasiblePolicy(seed=seed * 10).select_actions,
                )
            )
        )

    rows = [
        scenario_row(
            scenario_name="controlled_rule_medium_profile",
            scenario_type="controlled_edge_case",
            episode_results=[controlled_rule],
            review_expectation="sensible policy is profitable with no stockouts",
        ),
        scenario_row(
            scenario_name="controlled_all_zero_high_demand",
            scenario_type="controlled_edge_case",
            episode_results=[controlled_zero],
            review_expectation="ordering nothing under high demand is strongly negative",
        ),
        scenario_row(
            scenario_name="controlled_maximum_feasible_zero_demand",
            scenario_type="controlled_edge_case",
            episode_results=[controlled_maximum],
            review_expectation="over-ordering under zero demand is strongly negative",
        ),
    ]
    for scenario_name, results in stochastic_results.items():
        rows.append(
            scenario_row(
                scenario_name=scenario_name,
                scenario_type="paired_stochastic_policy",
                episode_results=results,
                review_expectation="extreme or random policy does not beat sensible rule",
            )
        )

    frame = pd.DataFrame(rows)
    rule_mean = float(
        frame.loc[
            frame["scenario_name"] == "stochastic_rule_order_up_to", "mean_scalar_reward"
        ].iloc[0]
    )
    frame.loc[frame["scenario_name"] == "controlled_rule_medium_profile", "passed"] = (
        (frame["mean_scalar_reward"] > 0)
        & (frame["mean_stockout_units"] == 0)
        & (frame["capacity_violations"] == 0)
    )
    frame.loc[
        frame["scenario_name"].isin(
            [
                "controlled_all_zero_high_demand",
                "controlled_maximum_feasible_zero_demand",
            ]
        ),
        "passed",
    ] = frame["mean_scalar_reward"] < 0
    frame.loc[frame["scenario_name"] == "stochastic_rule_order_up_to", "passed"] = (
        frame["capacity_violations"] == 0
    )
    frame.loc[
        frame["scenario_name"].isin(
            [
                "stochastic_all_zero_orders",
                "stochastic_maximum_feasible_orders",
                "stochastic_random_feasible",
            ]
        ),
        "passed",
    ] = frame["mean_scalar_reward"] < rule_mean
    frame["passed"] = frame["passed"] & frame["capacity_violations"].eq(0)
    return frame


def main() -> None:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    frame = build_audit_frame()
    frame.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(frame)} reward audit rows to {OUTPUT_PATH}")
    print(frame.to_string(index=False))
    if not frame["passed"].all():
        raise SystemExit("At least one reward audit scenario failed.")


if __name__ == "__main__":
    main()
