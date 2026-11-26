"""Train or run a tabular Q-learning agent on Super Mario Bros. 1-1."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import TypeAlias

from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from environment import MarioEnvironment, MarioState


StateKey: TypeAlias = tuple[int, ...]
DEFAULT_MODEL = Path(__file__).with_name("q_table.json")
STATUS_IDS = {"small": 0, "tall": 1, "fireball": 2}


def encode_state(
    state: MarioState,
    previous_state: MarioState | None = None,
) -> StateKey:
    """Convert game metadata into a compact tabular state representation."""
    delta_x = 0 if previous_state is None else state.x_pos - previous_state.x_pos
    delta_y = 0 if previous_state is None else state.y_pos - previous_state.y_pos
    velocity_x = max(-4, min(4, delta_x // 8))
    velocity_y = max(-4, min(4, delta_y // 8))
    enemies = tuple(state.enemy_types[:5])
    enemies += (0,) * (5 - len(enemies))

    return (
        state.world,
        state.stage,
        state.x_pos // 16,
        state.y_pos // 16,
        velocity_x,
        velocity_y,
        state.player_state,
        STATUS_IDS.get(state.status, 2),
        state.y_viewport,
        *enemies,
    )


def learning_reward(components: dict[str, float]) -> float:
    """Shape upstream rewards, emphasizing level completion and avoiding death."""
    return (
        components.get("progress", 0.0)
        + components.get("time", 0.0) * 0.02
        + components.get("score", 0.0)
        + components.get("coins", 0.0)
        + components.get("powerup", 0.0)
        + components.get("completion", 0.0) * 10.0
        + components.get("death", 0.0) * 4.0
    )


class QLearningAgent:
    """A small Q-table that can be saved and resumed between runs."""

    def __init__(
        self,
        action_count: int,
        learning_rate: float = 0.2,
        discount: float = 0.95,
        epsilon: float = 1.0,
        min_epsilon: float = 0.05,
        epsilon_decay: float = 0.995,
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
        """Get or initialize Q-values for one state."""
        return self.q_table.setdefault(state, [0.0] * self.action_count)

    def choose_action(self, state: StateKey, explore: bool = True) -> int:
        """Choose an exploratory or currently best action."""
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
        """Apply the one-step Q-learning update."""
        values = self.values_for(state)
        future_value = 0.0 if terminal else max(self.values_for(next_state))
        target = reward + self.discount * future_value
        values[action] += self.learning_rate * (target - values[action])

    def finish_episode(self) -> None:
        """Reduce exploration after each training episode."""
        self.training_episodes += 1
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def save(self, path: Path) -> None:
        """Persist the learned table in a portable JSON format."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": 1,
            "actions": SIMPLE_MOVEMENT,
            "epsilon": self.epsilon,
            "training_episodes": self.training_episodes,
            "training_episode_count_known": self.training_episode_count_known,
            "estimated_episode_offset_applied": self.estimated_episode_offset_applied,
            "q_table": [
                {"state": list(state), "values": values}
                for state, values in self.q_table.items()
            ],
        }
        path.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")

    def load(self, path: Path) -> bool:
        """Load a previously saved table; return False if it does not exist."""
        if not path.exists():
            return False

        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("version") != 1 or data.get("actions") != SIMPLE_MOVEMENT:
            raise ValueError(f"Modèle incompatible avec les actions actuelles: {path}")

        loaded = {
            tuple(int(value) for value in row["state"]): [
                float(value) for value in row["values"]
            ]
            for row in data["q_table"]
        }
        if any(len(values) != self.action_count for values in loaded.values()):
            raise ValueError(f"La table Q a un nombre d'actions invalide: {path}")

        self.q_table = loaded
        self.epsilon = float(data.get("epsilon", self.epsilon))
        self.training_episode_count_known = bool(
            data.get("training_episode_count_known", "training_episodes" in data)
        )
        self.training_episodes = int(data.get("training_episodes", 0))
        self.estimated_episode_offset_applied = bool(
            data.get("estimated_episode_offset_applied", False)
        )
        if (
            not self.training_episode_count_known
            and not self.estimated_episode_offset_applied
        ):
            self.training_episodes += 1000
            self.estimated_episode_offset_applied = True
        return True


def run_episode(
    environment: MarioEnvironment,
    agent: QLearningAgent,
    max_steps: int,
    frames_per_action: int,
    explore: bool,
    render: bool,
    seed: int | None = None,
    render_fps: float = 30.0,
) -> tuple[float, int, bool, int]:
    """Run one episode and update the Q-table when exploring."""
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1.")
    if frames_per_action < 1:
        raise ValueError("frames_per_action must be at least 1.")

    state = environment.reset(seed=seed)
    previous_state: MarioState | None = None
    total_reward = 0.0
    progress = state.x_pos
    steps_taken = 0

    for step_index in range(max_steps):
        steps_taken = step_index + 1
        state_key = encode_state(state, previous_state)
        action = agent.choose_action(state_key, explore=explore)
        next_state, terminated, truncated, environment_reward, components = (
            environment.step_action_with_reward(
                action,
                frames_per_action,
                render_fps=render_fps if render else None,
            )
        )
        reward = learning_reward(components) if components else environment_reward
        next_key = encode_state(next_state, state)
        episode_done = terminated or truncated

        if explore:
            agent.update(state_key, action, reward, next_key, episode_done)

        total_reward += reward
        progress = max(progress, next_state.x_pos)
        state, previous_state = next_state, state

        if episode_done:
            break

    return total_reward, progress, state.clear, steps_taken


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--play", action="store_true", help="Jouer sans modifier la Q-table")
    args = parser.parse_args()

    if args.episodes < 1 or args.steps < 1 or args.frames < 1 or args.fps <= 0:
        parser.error("episodes, steps, frames et fps doivent être supérieurs à zéro")

    environment = MarioEnvironment(render=args.render)
    agent = QLearningAgent(
        action_count=len(SIMPLE_MOVEMENT),
        seed=args.seed,
    )
    try:
        model_loaded = agent.load(args.model)
        if model_loaded:
            print(f"Q-table chargée: {len(agent.q_table)} états depuis {args.model}")
            count_label = (
                "Épisodes d'entraînement enregistrés"
                if agent.training_episode_count_known
                else "Épisodes comptabilisés depuis l'ajout du compteur"
            )
            print(f"{count_label}: {agent.training_episodes}")
            if not agent.training_episode_count_known:
                print("Le nombre d'épisodes antérieurs est inconnu.")
        else:
            print("Aucune Q-table existante ; apprentissage depuis zéro.")

        episode_count = 1 if args.play else args.episodes
        for episode in range(episode_count):
            total_reward, progress, cleared, steps = run_episode(
                environment,
                agent,
                max_steps=args.steps,
                frames_per_action=args.frames,
                explore=not args.play,
                render=args.render,
                seed=args.seed + episode,
                render_fps=args.fps,
            )
            if not args.play:
                agent.finish_episode()
                if (episode + 1) % 10 == 0:
                    agent.save(args.model)

            result = "NIVEAU TERMINÉ" if cleared else "niveau non terminé"
            count_label = (
                "total entraînements"
                if agent.training_episode_count_known
                else "entraînements comptabilisés"
            )
            print(
                f"Épisode {episode + 1}/{episode_count}: {result} | "
                f"progression={progress} | récompense={total_reward:.1f} | "
                f"actions={steps} | epsilon={agent.epsilon:.3f} | "
                f"états appris={len(agent.q_table)} | "
                f"{count_label}={agent.training_episodes}"
            )

        if not args.play:
            agent.save(args.model)
            print(f"Q-table sauvegardée dans {args.model}")
    except KeyboardInterrupt:
        if not args.play:
            agent.save(args.model)
            print(f"Entraînement interrompu ; Q-table sauvegardée dans {args.model}")
    finally:
        environment.close()


if __name__ == "__main__":
    main()