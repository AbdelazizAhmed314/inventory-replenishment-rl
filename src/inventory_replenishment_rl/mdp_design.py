"""Validation helpers for the reviewed MDP design."""

from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "mdp_config.json"
ENCODING_PATH = ROOT / "config" / "state_action_encoding.json"


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def validate_mdp_design() -> dict[str, int]:
    config = load_json(CONFIG_PATH)
    encoding = load_json(ENCODING_PATH)

    product_count = len(config["products"])
    assert product_count == config["decision_model"]["products_per_week"]

    regimes = config["demand"]["regimes"]
    assert regimes == ["low", "medium", "high"]
    for regime in regimes:
        probabilities = config["demand"]["regime_transition_matrix"][regime]
        assert set(probabilities) == set(regimes)
        assert math.isclose(sum(probabilities.values()), 1.0)

    variables = encoding["state_variables"]
    state_count = math.prod(variable["count"] for variable in variables.values())
    action_count = encoding["actions"]["count"]
    q_table_cells = state_count * action_count

    assert product_count == variables["product_id"]["count"]
    assert config["decision_model"]["episode_weeks"] == variables["week"]["count"]
    assert config["inventory"]["capacity_per_product"] + 1 == variables["on_hand"]["count"]
    assert len(config["actions"]["order_quantities"]) == action_count
    assert state_count == encoding["nominal_state_count"]
    assert q_table_cells == encoding["nominal_q_table_cells"]
    assert q_table_cells * 8 == encoding["float64_q_value_bytes"]

    return {
        "products": product_count,
        "states": state_count,
        "actions": action_count,
        "q_table_cells": q_table_cells,
    }
