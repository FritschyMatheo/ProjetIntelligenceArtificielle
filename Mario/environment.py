from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace


@dataclass(frozen=True)
class MarioState:
    """ position de mario """

    world: int
    stage: int
    x_pos: int
    y_pos: int
    lives: int
    player_state: int
    status: str
    enemy_types: tuple[int, ...]
    y_viewport: int
    clear: bool
    death: bool

    @classmethod
    def from_info(cls, info: dict[str, Any]) -> MarioState:
        return cls(
            world=int(info.get("world", 1)),
            stage=int(info.get("stage", 1)),
            x_pos=int(info.get("x_pos", 0)),
            y_pos=int(info.get("y_pos", 0)),
            lives=int(info.get("lives", info.get("life", 0))),
            player_state=int(info.get("player_state", 0)),
            status=str(info.get("status", "small")),
            enemy_types=tuple(int(value) for value in info.get("enemy_types", ())),
            y_viewport=int(info.get("y_viewport", 1)),
            clear=bool(info.get("clear", info.get("flag_get", False))),
            death=bool(
                info.get("death", False)
                or info.get("is_dying", False)
                or info.get("is_dead", False)
            ),
        )


class MarioEnvironment:

    def __init__(
        self,
        render: bool = True,
        env_id: str = "SuperMarioBros-1-1-v0",
    ) -> None:
        render_mode = "human" if render else None
        base_env = gym.make(env_id, render_mode=render_mode)
        self.env = JoypadSpace(base_env, SIMPLE_MOVEMENT)
        self.state = MarioState.from_info({})
        self.last_info: dict[str, Any] = {}

    def reset(self, seed: int | None = None) -> MarioState:
        _, info = self.env.reset(seed=seed)
        self.last_info = info
        self.state = MarioState.from_info(info)
        return self.state

    def step_action(
        self,
        action: int,
        frames: int,
    ) -> tuple[MarioState, bool, bool]:
        state, terminated, truncated, _, _ = self.step_action_with_reward(
            action,
            frames,
        )
        return state, terminated, truncated

    def step_action_with_reward(
        self,
        action: int,
        frames: int,
        render_fps: float | None = None,
    ) -> tuple[MarioState, bool, bool, float, dict[str, float]]:
        if render_fps is not None and render_fps <= 0:
            raise ValueError("render_fps must be greater than 0.")

        terminated = False
        truncated = False
        total_reward = 0.0
        reward_components: dict[str, float] = {}

        for _ in range(frames):
            frame_started = time.perf_counter() if render_fps is not None else 0.0
            _, reward, terminated, truncated, info = self.env.step(action)
            self.last_info = info
            self.state = MarioState.from_info(info)
            total_reward += float(reward)
            for name, value in info.get("reward_components", {}).items():
                reward_components[name] = reward_components.get(name, 0.0) + float(value)
            if render_fps is not None:
                self.render()
                remaining = 1.0 / render_fps - (time.perf_counter() - frame_started)
                if remaining > 0:
                    time.sleep(remaining)
            if terminated or truncated or self.state.death:
                break

        return self.state, terminated, truncated, total_reward, reward_components

    def render(self) -> None:
        self.env.render()

    def close(self) -> None:
        self.env.close()