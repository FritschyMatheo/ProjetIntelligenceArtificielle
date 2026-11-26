"""Environnement Mario vu comme un MDP : actions A, un pas = k frames"""
import gymnasium as gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace

ACTIONS: list[list[str]] = RIGHT_ONLY  # A = {NOOP, droite, droite+A, droite+B, droite+A+B}
NIVEAU = "SuperMarioBros-1-1-v0"


def make_env(render: bool = False) -> JoypadSpace:
    """Crée le niveau 1-1 avec les actions ACTIONS ; render=True ouvre une fenêtre."""
    render_mode = "human" if render else None
    return JoypadSpace(gym.make(NIVEAU, render_mode=render_mode), ACTIONS)


def step_k(env: JoypadSpace, a: int, k: int) -> tuple[float, bool, bool, dict]:
    """Un pas t → t+1 : joue l'action a pendant au plus k frames."""
    if k < 1:
        raise ValueError("k doit être >= 1")
    r = 0.0
    terminated = truncated = False
    info: dict = {}
    for _ in range(k):
        _, r_frame, terminated, truncated, info = env.step(a)
        r += float(r_frame)
        if terminated or truncated:
            break
    return r, terminated, truncated, info


if __name__ == "__main__":
    # Démo : Mario court à droite (action 1) jusqu'à mourir
    env = make_env(render=True)
    env.reset()
    term = trunc = False
    while not (term or trunc):
        r, term, trunc, info = step_k(env, 1, k=4)
        print(f"x = {info['x_pos']:4d}   r = {r:6.1f}")
    env.close()