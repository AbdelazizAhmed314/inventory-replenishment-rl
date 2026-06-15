"""Transparent scalar reward calculation for the approved inventory MDP."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RewardConfig:
    """Approved Step 5 reward weights in simulated monetary units."""

    units: str
    sales_revenue_per_unit: float
    procurement_cost_per_unit: float
    holding_cost_per_unit_week: float
    stockout_penalty_per_unit: float
    terminal_excess_penalty_per_unit: float
    initial_inventory_cost_treatment: str

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> RewardConfig:
        if raw["status"] != "finalized_step_5":
            raise ValueError("Reward configuration must be finalized before simulation.")
        return cls(
            units=raw["units"],
            sales_revenue_per_unit=float(raw["sales_revenue_per_unit"]),
            procurement_cost_per_unit=float(raw["procurement_cost_per_unit"]),
            holding_cost_per_unit_week=float(raw["holding_cost_per_unit_week"]),
            stockout_penalty_per_unit=float(raw["stockout_penalty_per_unit"]),
            terminal_excess_penalty_per_unit=float(raw["terminal_excess_penalty_per_unit"]),
            initial_inventory_cost_treatment=raw["initial_inventory_cost_treatment"],
        )


@dataclass(frozen=True)
class RewardBreakdown:
    """Financial proxy components and their resulting scalar reward."""

    sales_revenue: float
    procurement_cost: float
    holding_cost: float
    stockout_penalty: float
    terminal_excess_penalty: float
    scalar_reward: float


def calculate_reward(
    *,
    config: RewardConfig,
    sales_units: int,
    ordered_units: int,
    holding_units: int,
    stockout_units: int,
    terminal_excess_units: int,
) -> RewardBreakdown:
    """Calculate one transition reward without hiding its component values."""
    sales_revenue = sales_units * config.sales_revenue_per_unit
    procurement_cost = ordered_units * config.procurement_cost_per_unit
    holding_cost = holding_units * config.holding_cost_per_unit_week
    stockout_penalty = stockout_units * config.stockout_penalty_per_unit
    terminal_excess_penalty = terminal_excess_units * config.terminal_excess_penalty_per_unit
    scalar_reward = (
        sales_revenue - procurement_cost - holding_cost - stockout_penalty - terminal_excess_penalty
    )
    return RewardBreakdown(
        sales_revenue=sales_revenue,
        procurement_cost=procurement_cost,
        holding_cost=holding_cost,
        stockout_penalty=stockout_penalty,
        terminal_excess_penalty=terminal_excess_penalty,
        scalar_reward=scalar_reward,
    )
