"""Evaluate non-learning baselines and generate local Step 4 evidence."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from inventory_replenishment_rl.baseline_evaluation import (
    evaluate_baselines,
    summarize_baselines,
)
from inventory_replenishment_rl.policies import OrderUpToPolicy, RandomFeasiblePolicy

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_DIR = ROOT / "evidence"
EPISODE_RESULTS_PATH = EVIDENCE_DIR / "baseline_episode_results.csv"
SUMMARY_PATH = EVIDENCE_DIR / "baseline_summary.csv"
PLOT_PATH = EVIDENCE_DIR / "baseline_comparison.png"
ENVIRONMENT_SEEDS = tuple(range(1000, 1050))


def policy_factories():
    return {
        "random_feasible": lambda seed: RandomFeasiblePolicy(seed=seed),
        "rule_order_up_to": lambda _seed: OrderUpToPolicy(),
    }


def save_comparison_plot(summary) -> None:
    policies = summary["policy"]
    service_level = summary["service_level_mean"]
    service_level_std = summary["service_level_std"]
    stockout_units = summary["total_stockout_units_mean"]
    stockout_units_std = summary["total_stockout_units_std"]

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(
        policies,
        service_level,
        yerr=service_level_std,
        capsize=4,
        color=["#7f8c8d", "#2874a6"],
    )
    axes[0].set_title("Mean Service Level (+/- 1 SD)")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Units sold / demand")

    axes[1].bar(
        policies,
        stockout_units,
        yerr=stockout_units_std,
        capsize=4,
        color=["#7f8c8d", "#2874a6"],
    )
    axes[1].set_title("Mean Stockout Units (+/- 1 SD)")
    axes[1].set_ylabel("Units per episode")

    for axis in axes:
        axis.tick_params(axis="x", rotation=15)
        axis.grid(axis="y", alpha=0.25)

    figure.suptitle("Baseline Policy Comparison Across Identical Environment Seeds")
    figure.tight_layout()
    figure.savefig(PLOT_PATH, dpi=160)
    plt.close(figure)


def main() -> None:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    episode_results = evaluate_baselines(policy_factories(), ENVIRONMENT_SEEDS)
    summary = summarize_baselines(episode_results)

    episode_results.to_csv(EPISODE_RESULTS_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)
    save_comparison_plot(summary)

    print(f"Wrote {len(episode_results)} episode rows to {EPISODE_RESULTS_PATH}")
    print(f"Wrote baseline summary to {SUMMARY_PATH}")
    print(f"Wrote comparison plot to {PLOT_PATH}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
