"""Validate the reviewed MDP configuration without running a simulator."""

from inventory_replenishment_rl.mdp_design import validate_mdp_design

if __name__ == "__main__":
    result = validate_mdp_design()
    print(
        "MDP design valid: "
        f"{result['products']} products, "
        f"{result['states']:,} states, "
        f"{result['actions']} actions, "
        f"{result['q_table_cells']:,} Q-table cells"
    )
