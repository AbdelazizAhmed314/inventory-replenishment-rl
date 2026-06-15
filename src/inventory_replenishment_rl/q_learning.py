"""Tabular Q-learning components for the approved inventory MDP."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from inventory_replenishment_rl.simulator import InventorySimulator, InventoryState, SimulatorConfig

DEFAULT_TRAINING_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "q_learning_config.json"
)


@dataclass(frozen=True)
class TrainingConfig:
    algorithm: str
    episodes: int
    learning_rate: float
    discount_factor: float
    epsilon_schedule: str
    epsilon_start: float
    epsilon_end: float
    epsilon_decay_fraction: float
    agent_seed: int
    environment_seed_start: int
    q_initial_value: float
    rolling_window: int

    @classmethod
    def from_json(cls, path: Path = DEFAULT_TRAINING_CONFIG_PATH) -> TrainingConfig:
        with path.open(encoding="utf-8") as file:
            raw = json.load(file)
        config = cls(**{field: raw[field] for field in cls.__dataclass_fields__})
        config.validate()
        return config

    def validate(self) -> None:
        if self.algorithm != "tabular_q_learning":
            raise ValueError("Only tabular_q_learning is supported.")
        if self.episodes <= 0 or self.rolling_window <= 0:
            raise ValueError("Episode count and rolling window must be positive.")
        if not 0 < self.learning_rate <= 1:
            raise ValueError("Learning rate must be in (0, 1].")
        if not 0 <= self.discount_factor <= 1:
            raise ValueError("Discount factor must be in [0, 1].")
        if self.epsilon_schedule != "linear":
            raise ValueError("Only a linear epsilon schedule is supported.")
        if not 0 <= self.epsilon_end <= self.epsilon_start <= 1:
            raise ValueError("Epsilon values must satisfy 0 <= end <= start <= 1.")
        if not 0 < self.epsilon_decay_fraction <= 1:
            raise ValueError("Epsilon decay fraction must be in (0, 1].")

    def to_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


@dataclass(frozen=True)
class ActionSelection:
    action: int
    explored: bool


@dataclass(frozen=True)
class QUpdate:
    td_target: float
    td_error: float
    new_value: float


class StateActionEncoder:
    """Map approved states and action quantities to dense Q-table indices."""

    def __init__(self, simulator_config: SimulatorConfig) -> None:
        self.product_ids = tuple(product.product_id for product in simulator_config.products)
        self.actions = simulator_config.order_quantities
        self.regimes = simulator_config.regimes
        self.episode_weeks = simulator_config.episode_weeks
        self.capacity = simulator_config.capacity_per_product
        self._action_indices = {action: index for index, action in enumerate(self.actions)}
        self._regime_indices = {regime: index for index, regime in enumerate(self.regimes)}

        if self.product_ids != tuple(range(len(self.product_ids))):
            raise ValueError("Product IDs must be contiguous and zero-based.")

    @property
    def q_shape(self) -> tuple[int, int, int, int, int, int]:
        return (
            len(self.product_ids),
            self.episode_weeks,
            self.capacity + 1,
            len(self.actions),
            len(self.regimes),
            len(self.actions),
        )

    def state_index(self, state: InventoryState) -> tuple[int, int, int, int, int]:
        return (
            state.product_id,
            state.week,
            state.on_hand,
            self._action_indices[state.incoming_order],
            self._regime_indices[state.demand_regime],
        )

    def action_index(self, action: int) -> int:
        return self._action_indices[action]

    def feasible_action_indices(self, feasible_actions: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(self.action_index(action) for action in feasible_actions)


class QLearningAgent:
    """Shared feasible-action-aware tabular Q-learning controller."""

    def __init__(
        self,
        *,
        simulator_config: SimulatorConfig,
        seed: int,
        initial_value: float = 0.0,
    ) -> None:
        self.encoder = StateActionEncoder(simulator_config)
        self.q_values = np.full(self.encoder.q_shape, initial_value, dtype=np.float64)
        self.visit_counts = np.zeros(self.encoder.q_shape, dtype=np.int64)
        self.visited_state_action_pairs = 0
        self._rng = np.random.default_rng(seed)

    @classmethod
    def from_q_table_csv(
        cls,
        *,
        simulator_config: SimulatorConfig,
        path: Path,
        seed: int = 0,
    ) -> QLearningAgent:
        agent = cls(simulator_config=simulator_config, seed=seed)
        frame = pd.read_csv(path)
        expected_rows = int(np.prod(agent.encoder.q_shape))
        if len(frame) != expected_rows:
            raise ValueError(f"Expected {expected_rows:,} Q-table rows, received {len(frame):,}.")
        frame["_incoming_index"] = frame["incoming_order"].map(agent.encoder._action_indices)
        frame["_regime_index"] = frame["demand_regime"].map(agent.encoder._regime_indices)
        frame["_action_index"] = frame["action_order_quantity"].map(agent.encoder._action_indices)
        if frame[["_incoming_index", "_regime_index", "_action_index"]].isna().any().any():
            raise ValueError("Q-table contains an unknown incoming order, regime, or action.")
        frame = frame.sort_values(
            [
                "product_id",
                "week",
                "on_hand",
                "_incoming_index",
                "_regime_index",
                "_action_index",
            ]
        )
        agent.q_values = frame["q_value"].to_numpy(dtype=np.float64).reshape(agent.encoder.q_shape)
        agent.visit_counts = (
            frame["visit_count"].to_numpy(dtype=np.int64).reshape(agent.encoder.q_shape)
        )
        agent.visited_state_action_pairs = int(np.count_nonzero(agent.visit_counts))
        return agent

    def choose_action(
        self,
        *,
        state: InventoryState,
        feasible_actions: tuple[int, ...],
        epsilon: float,
    ) -> ActionSelection:
        if not feasible_actions:
            raise ValueError("At least one feasible action is required.")
        if not 0 <= epsilon <= 1:
            raise ValueError("Epsilon must be in [0, 1].")

        if self._rng.random() < epsilon:
            return ActionSelection(
                action=int(self._rng.choice(feasible_actions)),
                explored=True,
            )

        state_index = self.encoder.state_index(state)
        feasible_indices = self.encoder.feasible_action_indices(feasible_actions)
        values = self.q_values[state_index][list(feasible_indices)]
        best_value = values.max()
        best_positions = np.flatnonzero(values == best_value)
        selected_position = int(self._rng.choice(best_positions))
        return ActionSelection(
            action=feasible_actions[selected_position],
            explored=False,
        )

    def greedy_action(
        self,
        *,
        state: InventoryState,
        feasible_actions: tuple[int, ...],
    ) -> int:
        """Return a deterministic frozen-policy action, preferring the smaller tied action."""
        state_index = self.encoder.state_index(state)
        feasible_indices = self.encoder.feasible_action_indices(feasible_actions)
        values = self.q_values[state_index][list(feasible_indices)]
        return feasible_actions[int(np.argmax(values))]

    def greedy_actions(self, simulator: InventorySimulator) -> dict[int, int]:
        return {
            product_id: self.greedy_action(
                state=state,
                feasible_actions=simulator.feasible_actions(product_id),
            )
            for product_id, state in simulator.states.items()
        }

    def update(
        self,
        *,
        state: InventoryState,
        action: int,
        reward: float,
        next_state: InventoryState | None,
        next_feasible_actions: tuple[int, ...],
        terminated: bool,
        learning_rate: float,
        discount_factor: float,
    ) -> QUpdate:
        state_index = self.encoder.state_index(state)
        action_index = self.encoder.action_index(action)
        cell = (*state_index, action_index)
        current_value = self.q_values[cell]

        if terminated:
            best_next = 0.0
        else:
            if next_state is None or not next_feasible_actions:
                raise ValueError("Non-terminal updates require a next state and feasible actions.")
            next_index = self.encoder.state_index(next_state)
            next_action_indices = self.encoder.feasible_action_indices(next_feasible_actions)
            best_next = float(self.q_values[next_index][list(next_action_indices)].max())

        td_target = reward + discount_factor * best_next
        td_error = td_target - current_value
        new_value = current_value + learning_rate * td_error
        self.q_values[cell] = new_value
        if self.visit_counts[cell] == 0:
            self.visited_state_action_pairs += 1
        self.visit_counts[cell] += 1
        return QUpdate(
            td_target=float(td_target),
            td_error=float(td_error),
            new_value=float(new_value),
        )


def epsilon_for_episode(episode: int, config: TrainingConfig) -> float:
    decay_episodes = max(round(config.episodes * config.epsilon_decay_fraction), 1)
    if episode >= decay_episodes:
        return config.epsilon_end
    progress = min(episode / decay_episodes, 1.0)
    return config.epsilon_start + progress * (config.epsilon_end - config.epsilon_start)


def train_q_learning(
    training_config: TrainingConfig,
    simulator_config: SimulatorConfig | None = None,
) -> tuple[QLearningAgent, pd.DataFrame]:
    simulator_config = simulator_config or SimulatorConfig.from_json()
    agent = QLearningAgent(
        simulator_config=simulator_config,
        seed=training_config.agent_seed,
        initial_value=training_config.q_initial_value,
    )
    simulator = InventorySimulator(
        config=simulator_config,
        seed=training_config.environment_seed_start,
    )
    rows: list[dict[str, float | int]] = []

    for episode in range(training_config.episodes):
        epsilon = epsilon_for_episode(episode, training_config)
        environment_seed = training_config.environment_seed_start + episode
        simulator.reset(seed=environment_seed)
        episode_reward = 0.0
        sales_units = 0
        stockout_units = 0
        holding_units = 0
        terminal_excess_units = 0
        ordered_units = 0
        exploratory_actions = 0
        absolute_td_errors: list[float] = []

        while not simulator.terminated:
            actions: dict[int, int] = {}
            for product_id, state in simulator.states.items():
                selection = agent.choose_action(
                    state=state,
                    feasible_actions=simulator.feasible_actions(product_id),
                    epsilon=epsilon,
                )
                actions[product_id] = selection.action
                exploratory_actions += int(selection.explored)

            transitions = simulator.step_week(actions)
            for transition in transitions:
                next_feasible_actions = (
                    ()
                    if transition.terminated
                    else simulator.feasible_actions(transition.state.product_id)
                )
                update = agent.update(
                    state=transition.state,
                    action=transition.action,
                    reward=transition.scalar_reward,
                    next_state=transition.next_state,
                    next_feasible_actions=next_feasible_actions,
                    terminated=transition.terminated,
                    learning_rate=training_config.learning_rate,
                    discount_factor=training_config.discount_factor,
                )
                absolute_td_errors.append(abs(update.td_error))
                episode_reward += transition.scalar_reward
                sales_units += transition.reward_drivers.sales_units
                stockout_units += transition.reward_drivers.stockout_units
                holding_units += transition.reward_drivers.holding_units
                terminal_excess_units += transition.reward_drivers.terminal_excess_units
                ordered_units += transition.reward_drivers.ordered_units

        rows.append(
            {
                "episode": episode,
                "environment_seed": environment_seed,
                "epsilon": epsilon,
                "episode_reward": episode_reward,
                "sales_units": sales_units,
                "stockout_units": stockout_units,
                "holding_units": holding_units,
                "terminal_excess_units": terminal_excess_units,
                "ordered_units": ordered_units,
                "exploratory_actions": exploratory_actions,
                "mean_absolute_td_error": float(np.mean(absolute_td_errors)),
                "visited_state_action_pairs": agent.visited_state_action_pairs,
            }
        )

    history = pd.DataFrame(rows)
    window = training_config.rolling_window
    history["rolling_mean_reward"] = history["episode_reward"].rolling(window, min_periods=1).mean()
    history["rolling_mean_stockouts"] = (
        history["stockout_units"].rolling(window, min_periods=1).mean()
    )
    history["rolling_mean_holding"] = history["holding_units"].rolling(window, min_periods=1).mean()
    return agent, history


def q_table_frame(agent: QLearningAgent) -> pd.DataFrame:
    encoder = agent.encoder
    rows: list[dict[str, float | int | str | bool]] = []
    for product_id in encoder.product_ids:
        for week in range(encoder.episode_weeks):
            for on_hand in range(encoder.capacity + 1):
                for incoming_index, incoming_order in enumerate(encoder.actions):
                    valid_state = on_hand + incoming_order <= encoder.capacity
                    for regime_index, demand_regime in enumerate(encoder.regimes):
                        state_index = (
                            product_id,
                            week,
                            on_hand,
                            incoming_index,
                            regime_index,
                        )
                        for action_index, action in enumerate(encoder.actions):
                            rows.append(
                                {
                                    "product_id": product_id,
                                    "week": week,
                                    "on_hand": on_hand,
                                    "incoming_order": incoming_order,
                                    "demand_regime": demand_regime,
                                    "action_order_quantity": action,
                                    "valid_state": valid_state,
                                    "feasible_action": valid_state
                                    and on_hand + incoming_order + action <= encoder.capacity,
                                    "q_value": agent.q_values[(*state_index, action_index)],
                                    "visit_count": agent.visit_counts[(*state_index, action_index)],
                                }
                            )
    return pd.DataFrame(rows)
