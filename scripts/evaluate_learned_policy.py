"""Evaluate the frozen Q-learning policy against baselines and stress scenarios."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from inventory_replenishment_rl.evaluation import (
    EvaluationScenario,
    FrozenQLearningPolicy,
    evaluate_episode,
    evaluate_policies,
    product_episode_metrics,
    summarize_results,
)
from inventory_replenishment_rl.policies import OrderUpToPolicy, RandomFeasiblePolicy
from inventory_replenishment_rl.q_learning import QLearningAgent
from inventory_replenishment_rl.simulator import SimulatorConfig

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT / "artifacts"
EVIDENCE_DIR = ROOT / "evidence"
Q_TABLE_PATH = ARTIFACTS_DIR / "q_table.csv"
POLICY_RESULTS_PATH = EVIDENCE_DIR / "policy_evaluation_results.csv"
POLICY_SUMMARY_PATH = EVIDENCE_DIR / "policy_evaluation_summary.csv"
PRODUCT_SUMMARY_PATH = EVIDENCE_DIR / "product_evaluation_summary.csv"
SCENARIO_RESULTS_PATH = EVIDENCE_DIR / "scenario_evaluation_results.csv"
SCENARIO_SUMMARY_PATH = EVIDENCE_DIR / "scenario_evaluation_summary.csv"
REPRESENTATIVE_PATH = EVIDENCE_DIR / "representative_episode_results.csv"
POLICY_PLOT_PATH = EVIDENCE_DIR / "policy_comparison.png"
SCENARIO_PLOT_PATH = EVIDENCE_DIR / "scenario_comparison.png"
EVALUATION_MANIFEST_PATH = ARTIFACTS_DIR / "evaluation_manifest.json"

UNSEEN_SEEDS = tuple(range(200000, 200200))
SCENARIO_SEEDS = tuple(range(210000, 210050))
STANDARD_SCENARIO = EvaluationScenario(name="standard_unseen")
STRESS_SCENARIOS = (
    EvaluationScenario(name="demand_spike", demand_mode="maximum"),
    EvaluationScenario(name="demand_drop", demand_mode="zero"),
    EvaluationScenario(name="high_procurement_cost", procurement_cost_per_unit=6.0),
    EvaluationScenario(name="high_holding_cost", holding_cost_per_unit_week=4.0),
)


def policy_factories(agent: QLearningAgent):
    return {
        FrozenQLearningPolicy.name: lambda _seed: FrozenQLearningPolicy(agent),
        "rule_order_up_to": lambda _seed: OrderUpToPolicy(),
        "random_feasible": lambda seed: RandomFeasiblePolicy(seed=seed),
    }


def save_policy_plot(summary: pd.DataFrame) -> None:
    summary = summary.set_index("policy").loc[
        [FrozenQLearningPolicy.name, "rule_order_up_to", "random_feasible"]
    ]
    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    metrics = [
        ("total_scalar_reward_mean", "Mean Reward"),
        ("total_stockout_units_mean", "Mean Stockout Units"),
        ("average_holding_units_mean", "Mean Holding Units"),
        ("terminal_excess_units_mean", "Mean Terminal Excess"),
    ]
    colors = ["#1f77b4", "#2ca02c", "#7f8c8d"]
    for axis, (column, title) in zip(axes.flat, metrics, strict=True):
        axis.bar(summary.index, summary[column], color=colors)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=15)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Frozen Policy Evaluation on Identical Unseen Seeds")
    figure.tight_layout()
    figure.savefig(POLICY_PLOT_PATH, dpi=160)
    plt.close(figure)


def save_scenario_plot(summary: pd.DataFrame) -> None:
    reward = summary.pivot(
        index="scenario",
        columns="policy",
        values="total_scalar_reward_mean",
    )
    reward = reward[[FrozenQLearningPolicy.name, "rule_order_up_to", "random_feasible"]]
    axis = reward.plot(kind="bar", figsize=(11, 5), color=["#1f77b4", "#2ca02c", "#7f8c8d"])
    axis.set_ylabel("Mean scalar reward")
    axis.set_title("Frozen Policy Stress-Scenario Comparison")
    axis.tick_params(axis="x", rotation=15)
    axis.grid(axis="y", alpha=0.25)
    axis.figure.tight_layout()
    axis.figure.savefig(SCENARIO_PLOT_PATH, dpi=160)
    plt.close(axis.figure)


def build_product_summary(
    agent: QLearningAgent,
    simulator_config: SimulatorConfig,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for seed in UNSEEN_SEEDS:
        for policy_name, factory in policy_factories(agent).items():
            transitions = evaluate_episode(
                policy=factory(seed * 10),
                environment_seed=seed,
                scenario=STANDARD_SCENARIO,
                base_config=simulator_config,
            )
            rows.extend(
                product_episode_metrics(
                    policy_name=policy_name,
                    environment_seed=seed,
                    transitions=transitions,
                    capacity_per_product=simulator_config.capacity_per_product,
                    terminal_week=simulator_config.episode_weeks - 1,
                )
            )
    frame = pd.DataFrame(rows)
    return summarize_results(frame, ["policy", "product_id", "product_name"])


def representative_results(results: pd.DataFrame) -> pd.DataFrame:
    learned = results[results["policy"] == FrozenQLearningPolicy.name]
    seeds = set(learned.nsmallest(3, "total_scalar_reward")["environment_seed"])
    seeds.update(learned.nlargest(3, "total_scalar_reward")["environment_seed"])
    selected = results[results["environment_seed"].isin(seeds)].copy()
    selected["learned_episode_label"] = selected["environment_seed"].map(
        {
            **{
                seed: "learned_low_reward"
                for seed in learned.nsmallest(3, "total_scalar_reward")["environment_seed"]
            },
            **{
                seed: "learned_high_reward"
                for seed in learned.nlargest(3, "total_scalar_reward")["environment_seed"]
            },
        }
    )
    return selected.sort_values(["learned_episode_label", "environment_seed", "policy"])


def main() -> None:
    if not Q_TABLE_PATH.exists():
        raise SystemExit("Missing artifacts/q_table.csv. Run `make train` first.")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    simulator_config = SimulatorConfig.from_json()
    agent = QLearningAgent.from_q_table_csv(
        simulator_config=simulator_config,
        path=Q_TABLE_PATH,
    )
    factories = policy_factories(agent)

    policy_results = evaluate_policies(
        policy_factories=factories,
        environment_seeds=UNSEEN_SEEDS,
        scenario=STANDARD_SCENARIO,
        learned_agent=agent,
        base_config=simulator_config,
    )
    policy_summary = summarize_results(policy_results, ["policy"])
    product_summary = build_product_summary(agent, simulator_config)

    scenario_frames = [
        evaluate_policies(
            policy_factories=factories,
            environment_seeds=SCENARIO_SEEDS,
            scenario=scenario,
            learned_agent=agent,
            base_config=simulator_config,
        )
        for scenario in STRESS_SCENARIOS
    ]
    scenario_results = pd.concat(scenario_frames, ignore_index=True)
    scenario_summary = summarize_results(scenario_results, ["scenario", "policy"])

    policy_results.to_csv(POLICY_RESULTS_PATH, index=False)
    policy_summary.to_csv(POLICY_SUMMARY_PATH, index=False)
    product_summary.to_csv(PRODUCT_SUMMARY_PATH, index=False)
    scenario_results.to_csv(SCENARIO_RESULTS_PATH, index=False)
    scenario_summary.to_csv(SCENARIO_SUMMARY_PATH, index=False)
    representative_results(policy_results).to_csv(REPRESENTATIVE_PATH, index=False)
    save_policy_plot(policy_summary)
    save_scenario_plot(scenario_summary)

    learned_summary = policy_summary.set_index("policy").loc[FrozenQLearningPolicy.name]
    rule_summary = policy_summary.set_index("policy").loc["rule_order_up_to"]
    manifest = {
        "unseen_seed_start": min(UNSEEN_SEEDS),
        "unseen_seed_end": max(UNSEEN_SEEDS),
        "unseen_seed_count": len(UNSEEN_SEEDS),
        "scenario_seed_start": min(SCENARIO_SEEDS),
        "scenario_seed_end": max(SCENARIO_SEEDS),
        "scenario_seed_count": len(SCENARIO_SEEDS),
        "policies": list(factories),
        "scenarios": [scenario.name for scenario in STRESS_SCENARIOS],
        "learned_mean_reward": float(learned_summary["total_scalar_reward_mean"]),
        "rule_mean_reward": float(rule_summary["total_scalar_reward_mean"]),
        "learned_reward_lift_vs_rule": float(
            learned_summary["total_scalar_reward_mean"] - rule_summary["total_scalar_reward_mean"]
        ),
        "learned_mean_stockouts": float(learned_summary["total_stockout_units_mean"]),
        "rule_mean_stockouts": float(rule_summary["total_stockout_units_mean"]),
        "learned_mean_holding": float(learned_summary["average_holding_units_mean"]),
        "rule_mean_holding": float(rule_summary["average_holding_units_mean"]),
        "learned_mean_terminal_excess": float(learned_summary["terminal_excess_units_mean"]),
        "rule_mean_terminal_excess": float(rule_summary["terminal_excess_units_mean"]),
        "capacity_violations": int(
            policy_results["capacity_violations"].sum()
            + scenario_results["capacity_violations"].sum()
        ),
        "learned_unvisited_state_decisions": int(
            policy_results.loc[
                policy_results["policy"] == FrozenQLearningPolicy.name,
                "unvisited_state_decisions",
            ].sum()
        ),
    }
    EVALUATION_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {len(policy_results)} standard evaluation rows to {POLICY_RESULTS_PATH}")
    print(f"Wrote {len(scenario_results)} stress-scenario rows to {SCENARIO_RESULTS_PATH}")
    print(f"Wrote evaluation manifest to {EVALUATION_MANIFEST_PATH}")
    print(policy_summary.to_string(index=False))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
