from inventory_replenishment_rl.mdp_design import validate_mdp_design


def test_mdp_design_is_internally_consistent() -> None:
    result = validate_mdp_design()

    assert result == {
        "products": 5,
        "states": 22_320,
        "actions": 4,
        "q_table_cells": 89_280,
    }
