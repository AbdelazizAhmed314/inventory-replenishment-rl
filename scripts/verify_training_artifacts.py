"""Mechanically verify the generated Step 6 training artifact contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from inventory_replenishment_rl.q_learning import QLearningAgent, TrainingConfig
from inventory_replenishment_rl.simulator import SimulatorConfig

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


def main() -> None:
    required_paths = [
        Q_TABLE_PATH,
        RUN_CONFIG_PATH,
        MANIFEST_PATH,
        HISTORY_PATH,
        CURVE_PATH,
        POLICY_SNAPSHOT_PATH,
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise SystemExit(f"Missing training artifacts: {missing}")

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    training_config = TrainingConfig.from_json(RUN_CONFIG_PATH)
    simulator_config = SimulatorConfig.from_json()
    q_table = pd.read_csv(Q_TABLE_PATH)
    history = pd.read_csv(HISTORY_PATH)
    snapshot = pd.read_csv(POLICY_SNAPSHOT_PATH)

    expected_q_rows = (
        len(simulator_config.products)
        * simulator_config.episode_weeks
        * (simulator_config.capacity_per_product + 1)
        * len(simulator_config.order_quantities)
        * len(simulator_config.regimes)
        * len(simulator_config.order_quantities)
    )
    assert len(q_table) == expected_q_rows == manifest["q_table_rows"]
    assert len(history) == training_config.episodes == manifest["history_rows"]
    assert history["episode"].iloc[0] == 0
    assert history["episode"].iloc[-1] == training_config.episodes - 1
    assert history["epsilon"].iloc[0] == training_config.epsilon_start
    assert history["epsilon"].iloc[-1] == training_config.epsilon_end
    assert q_table.loc[~q_table["feasible_action"], "visit_count"].eq(0).all()
    assert history["visited_state_action_pairs"].is_monotonic_increasing
    assert len(snapshot) == len(simulator_config.products) * simulator_config.episode_weeks
    assert snapshot["state_visit_count"].gt(0).all()
    assert snapshot["greedy_action"].isin(simulator_config.order_quantities).all()

    loaded_agent = QLearningAgent.from_q_table_csv(
        simulator_config=simulator_config,
        path=Q_TABLE_PATH,
    )
    assert loaded_agent.visited_state_action_pairs == manifest["visited_state_action_pairs"]

    hashes = manifest["artifacts"]
    assert hashes["q_table.csv"] == file_sha256(Q_TABLE_PATH)
    assert hashes["training_run_config.json"] == file_sha256(RUN_CONFIG_PATH)
    assert hashes["training_history.csv"] == file_sha256(HISTORY_PATH)

    print(
        "Training artifacts valid: "
        f"{len(history):,} episodes, {len(q_table):,} Q-table rows, "
        f"{manifest['visited_state_action_pairs']:,} visited state-action pairs"
    )


if __name__ == "__main__":
    main()
