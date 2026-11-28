from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import TypeAlias

from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from environment import MarioEnvironment, MarioState

StateKey: TypeAlias = tuple[int, ...]
#STATUS_IDS = {"small": 0, "tall": 1, "fireball": 2}
A_IDS = {i for i, buttons in enumerate(SIMPLE_MOVEMENT) if "A" in buttons}

MODELS_DIR = Path(__file__).with_name("models")
JOURNAL_COLUMNS = ["episode", "x_max", "drapeau", "recompense", "actions", "epsilon", "nb_etats"]

LEARNING_RATE = 0.20
DISCOUNT_FACTOR = 0.95
INITIAL_EPSILON = 1.0
MIN_EPSILON = 0.05
EPSILON_DECAY = 0.995

def encode_state(environment: MarioEnvironment, previous_action: int) -> StateKey:
    """État s = (case_x, au_sol, a_tenu, ennemi, vitesse), lu dans la RAM du jeu."""
    ram = environment.env.unwrapped.ram
    x = int(ram[0x6D]) * 256 + int(ram[0x86])
    case_x = x // 16
    au_sol = int(ram[0x1D] == 0)
    a_tenu = int(previous_action in A_IDS)

    vx = int(ram[0x57])
    vx = 0 if vx > 127 else vx
    vitesse = 0 if vx < 16 else (1 if vx < 32 else 2)

    ennemi = 0
    for i in range(5):
        if ram[0x0F + i]:
            dx = int(ram[0x6E + i]) * 256 + int(ram[0x87 + i]) - x
            if -8 <= dx <= 32:
                ennemi = 1
            elif dx <= 80 and ennemi == 0:
                ennemi = 2

    return (case_x, au_sol, a_tenu, ennemi, vitesse)

"""
def encode_state(
    state: MarioState,
    previous_state: MarioState | None = None,
) -> StateKey:
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
"""

def learning_reward(components: dict[str, float]) -> float:
    return (
        components.get("progress", 0.0)
        + components.get("time", 0.0) * 0.02
        + components.get("score", 0.0)
        + components.get("coins", 0.0)
        + components.get("powerup", 0.0)
        + components.get("completion", 0.0) * 20.0
        + components.get("death", 0.0) * 4.0
    )


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

    def save(self, path: Path) -> None:
        """Enregistre la table Q, ε et le nombre d'épisodes dans un fichier JSON."""
        data = {
            "actions": SIMPLE_MOVEMENT,
            "epsilon": self.epsilon,
            "training_episodes": self.training_episodes,
            "q_table": [
                {"state": list(state), "values": values}
                for state, values in self.q_table.items()
            ],
        }
        path.write_text(json.dumps(data), encoding="utf-8")

    def load(self, path: Path) -> bool:
        """Recharge un modèle sauvegardé ; renvoie False si le fichier n'existe pas."""
        if not path.exists():
            return False
        data = json.loads(path.read_text(encoding="utf-8"))
        if data["actions"] != SIMPLE_MOVEMENT:
            raise ValueError(f"Modèle entraîné avec d'autres actions : {path}")
        self.q_table = {tuple(row["state"]): row["values"] for row in data["q_table"]}
        self.epsilon = data["epsilon"]
        self.training_episodes = data["training_episodes"]
        return True


"""def run_episode(
    environment: MarioEnvironment,
    agent: QLearningAgent,
    max_steps: int,
    frames_per_action: int,
    explore: bool,
    render: bool,
    seed: int | None = None,
    render_fps: float = 30.0,
) -> tuple[float, int, bool, int]:
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

        if explore:
            agent.update(state_key, action, reward, next_key, terminated)

        total_reward += reward
        progress = max(progress, next_state.x_pos)
        state, previous_state = next_state, state

        if terminated or truncated:
            break

    return total_reward, progress, state.clear, steps_taken"""
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
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1.")
    if frames_per_action < 1:
        raise ValueError("frames_per_action must be at least 1.")

    state = environment.reset(seed=seed)
    previous_action = 0
    state_key = encode_state(environment, previous_action)
    total_reward = 0.0
    progress = state.x_pos
    steps_taken = 0

    for step_index in range(max_steps):
        steps_taken = step_index + 1
        action = agent.choose_action(state_key, explore=explore)
        state, terminated, truncated, environment_reward, components = (
            environment.step_action_with_reward(
                action,
                frames_per_action,
                render_fps=render_fps if render else None,
            )
        )
        reward = learning_reward(components) if components else environment_reward
        next_key = encode_state(environment, action)

        if explore:
            agent.update(state_key, action, reward, next_key, terminated)

        total_reward += reward
        progress = max(progress, state.x_pos)
        state_key = next_key

        if terminated or truncated:
            break

    return total_reward, progress, state.clear, steps_taken

def write_journal_line(path: Path, row: list) -> None:
    """Ajoute une ligne au journal CSV ; écrit l'en-tête si le fichier est nouveau."""
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if new_file:
            writer.writerow(JOURNAL_COLUMNS)
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--name", default="q_table", help="Nom de l'essai (modèle et journal)")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--play", action="store_true", help="Jouer sans modifier la Q-table")
    args = parser.parse_args()

    if args.episodes < 1 or args.steps < 1 or args.frames < 1 or args.fps <= 0:
        parser.error("episodes, steps, frames et fps doivent être supérieurs à zéro")

    model_path = MODELS_DIR / f"{args.name}.json"
    journal_path = MODELS_DIR / f"{args.name}.csv"
    MODELS_DIR.mkdir(exist_ok=True)

    environment = MarioEnvironment(render=args.render)
    agent = QLearningAgent(
        action_count=len(SIMPLE_MOVEMENT),
        seed=args.seed,
    )
    if agent.load(model_path):
        print(
            f"Modèle chargé : {len(agent.q_table)} états, "
            f"{agent.training_episodes} épisodes, epsilon={agent.epsilon:.3f}"
        )
    else:
        print("Aucun modèle trouvé : apprentissage depuis zéro.")
    agent.random.seed(args.seed + agent.training_episodes)

    try:
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
                write_journal_line(
                    journal_path,
                    [
                        agent.training_episodes,
                        progress,
                        int(cleared),
                        round(total_reward, 1),
                        steps,
                        round(agent.epsilon, 4),
                        len(agent.q_table),
                    ],
                )
                if (episode + 1) % 10 == 0:
                    agent.save(model_path)

            result = "NIVEAU TERMINÉ" if cleared else "niveau non terminé"
            print(
                f"Épisode {episode + 1}/{episode_count}: {result} | "
                f"progression={progress} | récompense={total_reward:.1f} | "
                f"actions={steps} | epsilon={agent.epsilon:.3f} | "
                f"états appris={len(agent.q_table)} | "
                f"total entraînements={agent.training_episodes}"
            )
    except KeyboardInterrupt:
        print("Entraînement interrompu.")
    finally:
        if not args.play:
            agent.save(model_path)
            print(f"Modèle sauvegardé dans {model_path}")
        environment.close()

if __name__ == "__main__":
    main()