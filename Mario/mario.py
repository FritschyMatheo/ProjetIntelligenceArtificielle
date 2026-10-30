"""Super Mario Bros au clavier.  Lancer : python mario.py   (Échap pour quitter)
Pour changer une touche : modifier TOUCHES. Lettre -> ord("z"). Flèches -> GAUCHE, HAUT, DROITE, BAS.
"""
import itertools

import gymnasium as gym
import gym_super_mario_bros
from nes_py.play import play_human

GAUCHE, HAUT, DROITE, BAS = 0xFF51, 0xFF52, 0xFF53, 0xFF54
ENTREE = ord("\r")

TOUCHES = {
    # Pour les menus
    "right": DROITE,
    "left": GAUCHE,
    "down": BAS,
    "up": HAUT,
    "start": ENTREE,
    "select": ord("e"),
    "B": ord("x"),   # courir / boule de feu
    "A": ord(" "),   # sauter
}

ORDRE_NES = ["right", "left", "down", "up", "start", "select", "B", "A"]


class TouchesPerso(gym.Wrapper):
    def get_keys_to_action(self):
        touches = [TOUCHES[b] for b in ORDRE_NES]
        mapping = {}
        for bits in itertools.product([0, 1], repeat=8):
            octet = int("".join(map(str, bits)), 2)
            mapping[tuple(sorted(t for t, b in zip(touches, bits) if b))] = octet
        return mapping


if __name__ == "__main__":
    play_human(TouchesPerso(gym.make("SuperMarioBros-v0")))
