"""Verify the generated formal-evaluation artifact contract."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from inventory_replenishment_rl.evaluation import FrozenQLearningPolicy

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT / "artifacts"
EVIDENCE_DIR = ROOT / "evidence"
MANIFEST_PATH = ARTIFACTS_DIR / "evaluation_manifest.json"


def main() -> None:
    required = [
        MANIFEST_PATH,
        EVIDENCE_DIR / "policy_evaluation_results.csv",
        EVIDENCE_DIR / "policy_evaluation_summary.csv",
        EVIDENCE_DIR / "product_evaluation_summary.csv",
        EVIDENCE_DIR / "scenario_evaluation_results.csv",
        EVIDENCE_DIR / "scenario_evaluation_summary.csv",
        EVIDENCE_DIR / "representative_episode_results.csv",
        EVIDENCE_DIR / "policy_comparison.png",
        EVIDENCE_DIR / "scenario_comparison.png",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"Missing evaluation artifacts: {missing}")

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    policy_results = pd.read_csv(EVIDENCE_DIR / "policy_evaluation_results.csv")
    scenario_results = pd.read_csv(EVIDENCE_DIR / "scenario_evaluation_results.csv")
    policy_summary = pd.read_csv(EVIDENCE_DIR / "policy_evaluation_summary.csv")

    assert len(policy_results) == manifest["unseen_seed_count"] * len(manifest["policies"])
    assert len(scenario_results) == (
        manifest["scenario_seed_count"] * len(manifest["policies"]) * len(manifest["scenarios"])
    )
    assert set(policy_results["environment_seed"]).isdisjoint(range(100000, 150000))
    assert policy_results.groupby("environment_seed")["policy"].nunique().eq(3).all()
    assert (
        scenario_results.groupby(["scenario", "environment_seed"])["policy"].nunique().eq(3).all()
    )
    assert policy_results.groupby("environment_seed")["total_demand"].nunique().eq(1).all()
    assert policy_results["capacity_violations"].sum() == 0
    assert scenario_results["capacity_violations"].sum() == 0
    assert FrozenQLearningPolicy.name in set(policy_summary["policy"])
    assert manifest["capacity_violations"] == 0

    print(
        "Evaluation artifacts valid: "
        f"{len(policy_results)} unseen-seed rows, {len(scenario_results)} scenario rows, "
        "zero capacity violations"
    )


if __name__ == "__main__":
    main()
