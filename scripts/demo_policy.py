"""Run a concise seeded comparison of the frozen learned policy and rule baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

from inventory_replenishment_rl.evaluation import (
    EvaluationScenario,
    FrozenQLearningPolicy,
    evaluate_episode,
    transition_metrics,
)
from inventory_replenishment_rl.policies import OrderUpToPolicy
from inventory_replenishment_rl.q_learning import QLearningAgent
from inventory_replenishment_rl.simulator import SimulatorConfig

ROOT = Path(__file__).resolve().parents[1]
Q_TABLE_PATH = ROOT / "artifacts" / "q_table.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=220000, help="Unseen environment seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not Q_TABLE_PATH.exists():
        raise SystemExit("Missing artifacts/q_table.csv. Run `make train` first.")

    config = SimulatorConfig.from_json()
    agent = QLearningAgent.from_q_table_csv(simulator_config=config, path=Q_TABLE_PATH)
    policies = {
        FrozenQLearningPolicy.name: FrozenQLearningPolicy(agent),
        "rule_order_up_to": OrderUpToPolicy(),
    }
    scenario = EvaluationScenario(name="seeded_demo")

    print(f"Seeded inventory-policy demo: environment seed {args.seed}")
    print("policy             reward  service  stockouts  avg_holding  terminal_excess  violations")
    for policy_name, policy in policies.items():
        transitions = evaluate_episode(
            policy=policy,
            environment_seed=args.seed,
            scenario=scenario,
            base_config=config,
        )
        metrics = transition_metrics(
            transitions,
            capacity_per_product=config.capacity_per_product,
            terminal_week=config.episode_weeks - 1,
        )
        print(
            f"{policy_name:<18} "
            f"{metrics['total_scalar_reward']:>7.1f}  "
            f"{metrics['service_level']:>7.1%}  "
            f"{metrics['total_stockout_units']:>9}  "
            f"{metrics['average_holding_units']:>11.2f}  "
            f"{metrics['terminal_excess_units']:>15}  "
            f"{metrics['capacity_violations']:>10}"
        )


if __name__ == "__main__":
    main()
