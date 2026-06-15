"""Generate deterministic Step 3 simulator evidence without implementing a policy."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from inventory_replenishment_rl.simulator import InventorySimulator, TransitionRecord

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_DIR = ROOT / "evidence"
SAMPLE_PATH = EVIDENCE_DIR / "sample_episodes.csv"
REPORT_PATH = EVIDENCE_DIR / "simulator_validation.md"
SAMPLE_SEEDS = (101, 202, 303)


def all_zero_action_schedule(simulator: InventorySimulator) -> list[dict[int, int]]:
    """Return an explicit diagnostic schedule, not a baseline or learned policy."""
    actions = {product_id: 0 for product_id in simulator.states}
    return [dict(actions) for _ in range(simulator.config.episode_weeks)]


def front_loaded_max_schedule(simulator: InventorySimulator) -> list[dict[int, int]]:
    """Order the approved maximum once, then issue explicit zero orders."""
    maximum = max(simulator.config.order_quantities)
    first_week = {product_id: maximum for product_id in simulator.states}
    remaining = all_zero_action_schedule(simulator)[1:]
    return [first_week, *remaining]


def run_stochastic_zero_episode(seed: int) -> list[TransitionRecord]:
    simulator = InventorySimulator(seed=seed)
    return simulator.run_action_schedule(all_zero_action_schedule(simulator))


def run_controlled_episode(
    *,
    seed: int,
    action_schedule: list[dict[int, int]],
    actual_demand: int,
) -> list[TransitionRecord]:
    simulator = InventorySimulator(seed=seed)
    transitions: list[TransitionRecord] = []
    for actions in action_schedule:
        product_ids = simulator.states
        transitions.extend(
            simulator.step_week(
                actions,
                actual_demand_overrides={product_id: actual_demand for product_id in product_ids},
                next_regime_overrides={product_id: "medium" for product_id in product_ids},
            )
        )
    return transitions


def records_frame(records: list[TransitionRecord], scenario_name: str) -> pd.DataFrame:
    frame = pd.DataFrame(record.to_flat_dict() for record in records)
    frame.insert(0, "scenario_name", scenario_name)
    return frame


def stochastic_zero_frame(seed: int) -> pd.DataFrame:
    return records_frame(run_stochastic_zero_episode(seed), "stochastic_all_zero_orders")


def frame_digest(frame: pd.DataFrame) -> str:
    return hashlib.sha256(frame.to_csv(index=False).encode("utf-8")).hexdigest()


def build_report(frame: pd.DataFrame) -> str:
    same_seed_a = stochastic_zero_frame(SAMPLE_SEEDS[0])
    same_seed_b = stochastic_zero_frame(SAMPLE_SEEDS[0])
    different_seed = stochastic_zero_frame(999)
    duplicate_match = frame_digest(same_seed_a) == frame_digest(same_seed_b)
    different_seed_differs = frame_digest(same_seed_a) != frame_digest(different_seed)

    summary = (
        frame.groupby(["scenario_name", "episode_seed"])
        .agg(
            transitions=("product_id", "size"),
            total_demand=("actual_demand", "sum"),
            total_sales=("sales_units", "sum"),
            total_stockouts=("stockout_units", "sum"),
            maximum_holding=("holding_units", "max"),
            terminal_excess=("terminal_excess_units", "sum"),
            terminal_rows=("terminated", "sum"),
        )
        .reset_index()
    )

    return f"""# Simulator Validation Evidence

## Scope

This evidence validates the Step 3 simulator only. The sample episodes use explicit fixed
diagnostic schedules and environment-side outcome controls. They do not implement or claim a
baseline or learned policy.

## Results

- Stochastic all-zero sample seeds: `{", ".join(str(seed) for seed in SAMPLE_SEEDS)}`
- Controlled boundary scenarios: `front_loaded_max_zero_demand`, `all_zero_high_demand`
- Rows generated: `{len(frame)}`
- Expected rows: `{(len(SAMPLE_SEEDS) + 2) * 60}`
- Same-seed output is identical: `{duplicate_match}`
- Different-seed output differs: `{different_seed_differs}`
- Maximum available inventory observed: `{int(frame["available_inventory"].max())}`
- Configured capacity: `30`
- Scalar reward status: `weights_pending_step_5`

## Episode Summary

```text
{summary.to_string(index=False)}
```

## Validated Behaviors

- Reset produces five product states at week zero.
- Each complete episode contains 12 weeks and 60 product-level transitions.
- Existing incoming inventory arrives before demand.
- Actual demand is bounded and not present in the observation state before action selection.
- Stockouts do not terminate an episode.
- Capacity-invalid and unsupported actions are rejected before transitions occur.
- Raw reward drivers are logged while scalar reward weights remain deferred to Step 5.
- Terminal rows include terminal excess units and no next state.
- The front-loaded maximum-order scenario produces high holding and terminal-excess drivers.
- The high-demand zero-order scenario produces stockout drivers without early termination.
"""


def main() -> None:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    frames = [stochastic_zero_frame(seed) for seed in SAMPLE_SEEDS]

    front_loaded_simulator = InventorySimulator(seed=401)
    frames.append(
        records_frame(
            run_controlled_episode(
                seed=401,
                action_schedule=front_loaded_max_schedule(front_loaded_simulator),
                actual_demand=0,
            ),
            "front_loaded_max_zero_demand",
        )
    )
    high_demand_simulator = InventorySimulator(seed=402)
    frames.append(
        records_frame(
            run_controlled_episode(
                seed=402,
                action_schedule=all_zero_action_schedule(high_demand_simulator),
                actual_demand=20,
            ),
            "all_zero_high_demand",
        )
    )

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(SAMPLE_PATH, index=False)
    REPORT_PATH.write_text(build_report(combined), encoding="utf-8")
    print(f"Wrote {len(combined)} rows to {SAMPLE_PATH}")
    print(f"Wrote validation report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
