from __future__ import annotations

import random
from typing import TypeAlias


StateKey: TypeAlias = tuple[int, ...]

LEARNING_RATE = 0.20
DISCOUNT_FACTOR = 0.95
INITIAL_EPSILON = 1.0
MIN_EPSILON = 0.05
EPSILON_DECAY = 0.995


class QLearningAgent:
    def __init__(
        self,
        action_count: int,
        learning_rate: float = LEARNING_RATE,
        discount: float = DISCOUNT_FACTOR,
        epsilon: float = INITIAL_EPSILON,
        min_epsilon: float = MIN_EPSILON,
        epsilon_decay: float = EPSILON_DECAY,
        seed: int | None = None,
    ) -> None:
        self.action_count = action_count
        self.learning_rate = learning_rate
        self.discount = discount
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.q_table: dict[StateKey, list[float]] = {}
        self.training_episodes = 0
        self.training_episode_count_known = True
        self.estimated_episode_offset_applied = False
        self.random = random.Random(seed)

    def values_for(self, state: StateKey) -> list[float]:
        return self.q_table.setdefault(state, [0.0] * self.action_count)

    def choose_action(self, state: StateKey, explore: bool = True) -> int:
        if explore and self.random.random() < self.epsilon:
            return self.random.randrange(self.action_count)

        values = self.values_for(state)
        best_value = max(values)
        best_actions = [index for index, value in enumerate(values) if value == best_value]
        return self.random.choice(best_actions)

    def update(
        self,
        state: StateKey,
        action: int,
        reward: float,
        next_state: StateKey,
        terminal: bool,
    ) -> None:
        values = self.values_for(state)
        future_value = 0.0 if terminal else max(self.values_for(next_state))
        target = reward + self.discount * future_value
        values[action] += self.learning_rate * (target - values[action])

    def finish_episode(self) -> None:
        self.training_episodes += 1
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)