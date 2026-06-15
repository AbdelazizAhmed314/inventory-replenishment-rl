"""Deterministic, seeded simulator for the approved inventory MDP."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np

DemandRegime = Literal["low", "medium", "high"]
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "mdp_config.json"


@dataclass(frozen=True)
class ProductConfig:
    product_id: int
    product_name: str
    profile: str
    demand_means: Mapping[DemandRegime, float]


@dataclass(frozen=True)
class SimulatorConfig:
    products: tuple[ProductConfig, ...]
    episode_weeks: int
    capacity_per_product: int
    initial_on_hand: int
    initial_incoming_order: int
    order_quantities: tuple[int, ...]
    maximum_realized_demand: int
    initial_regime: DemandRegime
    regimes: tuple[DemandRegime, ...]
    regime_transition_matrix: Mapping[DemandRegime, Mapping[DemandRegime, float]]

    @classmethod
    def from_json(cls, path: Path = DEFAULT_CONFIG_PATH) -> SimulatorConfig:
        with path.open(encoding="utf-8") as file:
            raw = json.load(file)

        products = tuple(
            ProductConfig(
                product_id=product["product_id"],
                product_name=product["product_name"],
                profile=product["profile"],
                demand_means=product["demand_means"],
            )
            for product in raw["products"]
        )
        return cls(
            products=products,
            episode_weeks=raw["decision_model"]["episode_weeks"],
            capacity_per_product=raw["inventory"]["capacity_per_product"],
            initial_on_hand=raw["inventory"]["initial_on_hand"],
            initial_incoming_order=raw["inventory"]["initial_incoming_order"],
            order_quantities=tuple(raw["actions"]["order_quantities"]),
            maximum_realized_demand=raw["demand"]["maximum_realized_demand"],
            initial_regime=raw["demand"]["initial_regime"],
            regimes=tuple(raw["demand"]["regimes"]),
            regime_transition_matrix=raw["demand"]["regime_transition_matrix"],
        )


@dataclass(frozen=True)
class InventoryState:
    product_id: int
    week: int
    on_hand: int
    incoming_order: int
    demand_regime: DemandRegime

    def as_tuple(self) -> tuple[int, int, int, int, DemandRegime]:
        return (
            self.product_id,
            self.week,
            self.on_hand,
            self.incoming_order,
            self.demand_regime,
        )


@dataclass(frozen=True)
class RewardDrivers:
    """Raw quantities used by the Step 5 reward function."""

    sales_units: int
    ordered_units: int
    holding_units: int
    stockout_units: int
    terminal_excess_units: int


@dataclass(frozen=True)
class TransitionRecord:
    episode_seed: int
    product_name: str
    state: InventoryState
    action: int
    available_inventory: int
    actual_demand: int
    reward_drivers: RewardDrivers
    next_state: InventoryState | None
    terminated: bool

    def to_flat_dict(self) -> dict[str, int | str | bool | None]:
        row: dict[str, int | str | bool | None] = {
            "episode_seed": self.episode_seed,
            "product_id": self.state.product_id,
            "product_name": self.product_name,
            "week": self.state.week,
            "on_hand": self.state.on_hand,
            "incoming_order": self.state.incoming_order,
            "demand_regime": self.state.demand_regime,
            "action_order_quantity": self.action,
            "available_inventory": self.available_inventory,
            "actual_demand": self.actual_demand,
            **asdict(self.reward_drivers),
            "terminated": self.terminated,
            "scalar_reward": None,
            "reward_status": "weights_pending_step_5",
        }
        if self.next_state is None:
            row.update(
                {
                    "next_week": None,
                    "next_on_hand": None,
                    "next_incoming_order": None,
                    "next_demand_regime": None,
                }
            )
        else:
            row.update(
                {
                    "next_week": self.next_state.week,
                    "next_on_hand": self.next_state.on_hand,
                    "next_incoming_order": self.next_state.incoming_order,
                    "next_demand_regime": self.next_state.demand_regime,
                }
            )
        return row


class InventorySimulator:
    """Advance all five independent product states one synchronized week at a time."""

    def __init__(self, config: SimulatorConfig | None = None, seed: int = 0) -> None:
        self.config = config or SimulatorConfig.from_json()
        self._products = {product.product_id: product for product in self.config.products}
        self.reset(seed=seed)

    @property
    def current_week(self) -> int:
        return self._current_week

    @property
    def terminated(self) -> bool:
        return self._terminated

    @property
    def episode_seed(self) -> int:
        return self._episode_seed

    @property
    def states(self) -> dict[int, InventoryState]:
        return dict(self._states)

    def reset(
        self,
        seed: int = 0,
        initial_states: Mapping[int, InventoryState] | None = None,
    ) -> dict[int, InventoryState]:
        self._episode_seed = seed
        self._rng = np.random.default_rng(seed)
        self._current_week = 0
        self._terminated = False

        if initial_states is None:
            self._states = {
                product_id: InventoryState(
                    product_id=product_id,
                    week=0,
                    on_hand=self.config.initial_on_hand,
                    incoming_order=self.config.initial_incoming_order,
                    demand_regime=self.config.initial_regime,
                )
                for product_id in self._products
            }
        else:
            self._validate_initial_states(initial_states)
            self._states = dict(initial_states)
        return self.states

    def feasible_actions(self, product_id: int) -> tuple[int, ...]:
        state = self._states[product_id]
        return tuple(
            action
            for action in self.config.order_quantities
            if state.on_hand + state.incoming_order + action <= self.config.capacity_per_product
        )

    def action_mask(self, product_id: int) -> tuple[bool, ...]:
        feasible = set(self.feasible_actions(product_id))
        return tuple(action in feasible for action in self.config.order_quantities)

    def step_week(
        self,
        actions: Mapping[int, int],
        *,
        actual_demand_overrides: Mapping[int, int] | None = None,
        next_regime_overrides: Mapping[int, DemandRegime] | None = None,
    ) -> list[TransitionRecord]:
        """Advance one week after receiving independently selected product actions.

        Overrides are environment-side test/scenario controls. They are never included
        in the observation state presented to a policy.
        """
        if self._terminated:
            raise RuntimeError("Cannot step a terminated episode; call reset().")

        self._validate_actions(actions)
        actual_demand_overrides = actual_demand_overrides or {}
        next_regime_overrides = next_regime_overrides or {}
        terminal = self._current_week == self.config.episode_weeks - 1
        transitions: list[TransitionRecord] = []
        next_states: dict[int, InventoryState] = {}

        for product_id in sorted(self._products):
            product = self._products[product_id]
            state = self._states[product_id]
            action = actions[product_id]
            available_inventory = state.on_hand + state.incoming_order
            if product_id in actual_demand_overrides:
                actual_demand = actual_demand_overrides[product_id]
            else:
                actual_demand = self._sample_demand(product, state.demand_regime)
            self._validate_actual_demand(actual_demand)

            sales_units = min(available_inventory, actual_demand)
            stockout_units = max(actual_demand - available_inventory, 0)
            ending_on_hand = available_inventory - sales_units
            terminal_excess_units = ending_on_hand + action if terminal else 0

            if terminal:
                next_state = None
            else:
                if product_id in next_regime_overrides:
                    next_regime = next_regime_overrides[product_id]
                else:
                    next_regime = self._sample_next_regime(state.demand_regime)
                self._validate_regime(next_regime)
                next_state = InventoryState(
                    product_id=product_id,
                    week=self._current_week + 1,
                    on_hand=ending_on_hand,
                    incoming_order=action,
                    demand_regime=next_regime,
                )
                next_states[product_id] = next_state

            transitions.append(
                TransitionRecord(
                    episode_seed=self._episode_seed,
                    product_name=product.product_name,
                    state=state,
                    action=action,
                    available_inventory=available_inventory,
                    actual_demand=actual_demand,
                    reward_drivers=RewardDrivers(
                        sales_units=sales_units,
                        ordered_units=action,
                        holding_units=ending_on_hand,
                        stockout_units=stockout_units,
                        terminal_excess_units=terminal_excess_units,
                    ),
                    next_state=next_state,
                    terminated=terminal,
                )
            )

        self._current_week += 1
        self._terminated = terminal
        if not terminal:
            self._states = next_states
        return transitions

    def run_action_schedule(
        self,
        action_schedule: Sequence[Mapping[int, int]],
    ) -> list[TransitionRecord]:
        if len(action_schedule) != self.config.episode_weeks:
            raise ValueError(f"Expected {self.config.episode_weeks} weekly action mappings.")

        transitions: list[TransitionRecord] = []
        for actions in action_schedule:
            transitions.extend(self.step_week(actions))
        return transitions

    def _sample_demand(self, product: ProductConfig, regime: DemandRegime) -> int:
        sampled = int(self._rng.poisson(product.demand_means[regime]))
        return min(sampled, self.config.maximum_realized_demand)

    def _sample_next_regime(self, current_regime: DemandRegime) -> DemandRegime:
        probabilities = [
            self.config.regime_transition_matrix[current_regime][regime]
            for regime in self.config.regimes
        ]
        return self._rng.choice(self.config.regimes, p=probabilities).item()

    def _validate_actions(self, actions: Mapping[int, int]) -> None:
        expected = set(self._products)
        if set(actions) != expected:
            raise ValueError(f"Actions must contain exactly product IDs {sorted(expected)}.")
        for product_id, action in actions.items():
            if action not in self.config.order_quantities:
                raise ValueError(f"Unsupported order quantity {action} for product {product_id}.")
            if action not in self.feasible_actions(product_id):
                raise ValueError(
                    f"Infeasible order quantity {action} for product {product_id}; "
                    f"feasible actions are {self.feasible_actions(product_id)}."
                )

    def _validate_initial_states(self, initial_states: Mapping[int, InventoryState]) -> None:
        if set(initial_states) != set(self._products):
            raise ValueError("Initial states must contain every configured product exactly once.")
        for product_id, state in initial_states.items():
            if state.product_id != product_id or state.week != 0:
                raise ValueError("Initial state product IDs must match and week must be zero.")
            self._validate_inventory_position(state)
            self._validate_regime(state.demand_regime)

    def _validate_inventory_position(self, state: InventoryState) -> None:
        if state.on_hand < 0 or state.incoming_order < 0:
            raise ValueError("Inventory quantities cannot be negative.")
        if state.incoming_order not in self.config.order_quantities:
            raise ValueError("Incoming order must be one of the configured order quantities.")
        if state.on_hand + state.incoming_order > self.config.capacity_per_product:
            raise ValueError("Initial inventory position exceeds product capacity.")

    def _validate_actual_demand(self, actual_demand: int) -> None:
        if not 0 <= actual_demand <= self.config.maximum_realized_demand:
            raise ValueError("Actual demand must be within the configured bounded range.")

    def _validate_regime(self, regime: str) -> None:
        if regime not in self.config.regimes:
            raise ValueError(f"Unknown demand regime: {regime}.")
