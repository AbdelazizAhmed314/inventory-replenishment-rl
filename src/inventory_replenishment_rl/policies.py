"""Non-learning baseline policies for the approved inventory simulator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from inventory_replenishment_rl.simulator import (
    InventorySimulator,
    InventoryState,
    ProductConfig,
)


class BaselinePolicy(Protocol):
    name: str

    def select_actions(self, simulator: InventorySimulator) -> dict[int, int]:
        """Select one feasible order quantity for every product."""


@dataclass
class RandomFeasiblePolicy:
    """Uniformly sample from actions allowed by the hard capacity constraint."""

    seed: int
    name: str = "random_feasible"
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def select_actions(self, simulator: InventorySimulator) -> dict[int, int]:
        return {
            product_id: int(self._rng.choice(simulator.feasible_actions(product_id)))
            for product_id in simulator.states
        }


@dataclass(frozen=True)
class OrderUpToPolicy:
    """Order toward two weeks of expected demand using only observable state."""

    coverage_weeks: float = 2.0
    name: str = "rule_order_up_to"

    def select_actions(self, simulator: InventorySimulator) -> dict[int, int]:
        products = {product.product_id: product for product in simulator.config.products}
        return {
            product_id: self.select_action(
                state=state,
                product=products[product_id],
                feasible_actions=simulator.feasible_actions(product_id),
                capacity=simulator.config.capacity_per_product,
            )
            for product_id, state in simulator.states.items()
        }

    def select_action(
        self,
        *,
        state: InventoryState,
        product: ProductConfig,
        feasible_actions: tuple[int, ...],
        capacity: int,
    ) -> int:
        demand_mean = product.demand_means[state.demand_regime]
        target_inventory_position = min(capacity, round(self.coverage_weeks * demand_mean))
        inventory_position = state.on_hand + state.incoming_order
        gap = max(target_inventory_position - inventory_position, 0)

        actions_meeting_gap = [action for action in feasible_actions if action >= gap]
        if actions_meeting_gap:
            return min(actions_meeting_gap)
        return max(feasible_actions)
