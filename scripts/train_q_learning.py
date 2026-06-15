"""Train the approved tabular Q-learning agent and save Step 6 artifacts."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from inventory_replenishment_rl.q_learning import (
    TrainingConfig,
    q_table_frame,
    train_q_learning,
)
from inventory_replenishment_rl.simulator import InventoryState, SimulatorConfig

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT / "artifacts"
EVIDENCE_DIR = ROOT / "evidence"
Q_TABLE_PATH = ARTIFACTS_DIR / "q_table.csv"
RUN_CONFIG_PATH = ARTIFACTS_DIR / "training_run_config.json"
MANIFEST_PATH = ARTIFACTS_DIR / "training_manifest.json"
HISTORY_PATH = EVIDENCE_DIR / "training_history.csv"
CURVE_PATH = EVIDENCE_DIR / "reward_learning_curve.png"
POLICY_SNAPSHOT_PATH = EVIDENCE_DIR / "learned_policy_snapshot.csv"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_learning_curve(history) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(history["episode"], history["episode_reward"], alpha=0.12, color="#7f8c8d")
    axes[0].plot(
        history["episode"],
        history["rolling_mean_reward"],
        color="#2874a6",
        label="Rolling mean reward",
    )
    axes[0].set_ylabel("Episode reward")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(history["episode"], history["epsilon"], color="#a93226")
    axes[1].set_xlabel("Training episode")
    axes[1].set_ylabel("Epsilon")
    axes[1].grid(alpha=0.25)

    figure.suptitle("Tabular Q-Learning Training History")
    figure.tight_layout()
    figure.savefig(CURVE_PATH, dpi=160)
    plt.close(figure)


def policy_snapshot(agent, simulator_config: SimulatorConfig):
    rows = []
    for product in simulator_config.products:
        for week in range(simulator_config.episode_weeks):
            state_visits = agent.visit_counts[product.product_id, week].sum(axis=-1)
            on_hand, incoming_index, regime_index = (
                int(index) for index in np.unravel_index(state_visits.argmax(), state_visits.shape)
            )
            incoming_order = simulator_config.order_quantities[incoming_index]
            regime = simulator_config.regimes[regime_index]
            state = InventoryState(product.product_id, week, on_hand, incoming_order, regime)
            feasible_actions = tuple(
                action
                for action in simulator_config.order_quantities
                if on_hand + incoming_order + action <= simulator_config.capacity_per_product
            )
            state_index = agent.encoder.state_index(state)
            q_values = {
                action: agent.q_values[(*state_index, agent.encoder.action_index(action))]
                for action in simulator_config.order_quantities
            }
            rows.append(
                {
                    "product_id": product.product_id,
                    "product_name": product.product_name,
                    "week": week,
                    "on_hand": on_hand,
                    "incoming_order": incoming_order,
                    "demand_regime": regime,
                    "state_visit_count": int(state_visits[on_hand, incoming_index, regime_index]),
                    "feasible_actions": "|".join(str(action) for action in feasible_actions),
                    "greedy_action": agent.greedy_action(
                        state=state,
                        feasible_actions=feasible_actions,
                    ),
                    **{f"q_action_{action}": value for action, value in q_values.items()},
                }
            )
    return rows


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    training_config = TrainingConfig.from_json()
    simulator_config = SimulatorConfig.from_json()

    started = time.perf_counter()
    agent, history = train_q_learning(training_config, simulator_config)
    elapsed_seconds = time.perf_counter() - started
    table = q_table_frame(agent)

    history.to_csv(HISTORY_PATH, index=False)
    table.to_csv(Q_TABLE_PATH, index=False)
    RUN_CONFIG_PATH.write_text(
        json.dumps(training_config.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )
    save_learning_curve(history)
    pd.DataFrame(policy_snapshot(agent, simulator_config)).to_csv(
        POLICY_SNAPSHOT_PATH,
        index=False,
    )

    window = training_config.rolling_window
    manifest = {
        "algorithm": training_config.algorithm,
        "episodes": training_config.episodes,
        "q_table_rows": len(table),
        "history_rows": len(history),
        "visited_state_action_pairs": int((table["visit_count"] > 0).sum()),
        "final_epsilon": float(history["epsilon"].iloc[-1]),
        "initial_window_mean_reward": float(history["episode_reward"].head(window).mean()),
        "final_rolling_mean_reward": float(history["rolling_mean_reward"].iloc[-1]),
        "initial_window_mean_stockouts": float(history["stockout_units"].head(window).mean()),
        "final_rolling_mean_stockouts": float(history["rolling_mean_stockouts"].iloc[-1]),
        "initial_window_mean_holding": float(history["holding_units"].head(window).mean()),
        "final_rolling_mean_holding": float(history["rolling_mean_holding"].iloc[-1]),
        "artifacts": {
            "q_table.csv": file_sha256(Q_TABLE_PATH),
            "training_run_config.json": file_sha256(RUN_CONFIG_PATH),
            "training_history.csv": file_sha256(HISTORY_PATH),
        },
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"Training completed in {elapsed_seconds:.2f} seconds")
    print(f"Wrote {len(table):,} Q-table rows to {Q_TABLE_PATH}")
    print(f"Wrote {len(history):,} history rows to {HISTORY_PATH}")
    print(f"Wrote run configuration to {RUN_CONFIG_PATH}")
    print(f"Wrote reward-learning curve to {CURVE_PATH}")
    print(f"Wrote learned-policy snapshot to {POLICY_SNAPSHOT_PATH}")
    print(f"Wrote artifact manifest to {MANIFEST_PATH}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
