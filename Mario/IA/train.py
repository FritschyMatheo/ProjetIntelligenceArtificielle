"""Entraînement Q-learning sur Mario 1-1 (brique 4) et rejeu de la politique apprise.

Entraîner : python ia/train.py --episodes 500
Regarder  : python ia/train.py --jouer
"""
import argparse
import csv
import pickle
import time
from pathlib import Path

from agent import QAgent
from env import ACTIONS, make_env, step_k
from etat import phi

K = 4                      # frames par pas
ALPHA = 0.2                # taux d'apprentissage α
GAMMA = 0.95               # facteur d'actualisation γ
EPS_MIN, EPS_DECAY = 0.05, 0.985   # ε ← max(ε_min, ε · decay) après chaque épisode
BLOCAGE = 100              # pas sans nouveau record de x → on coupe l'épisode
DOSSIER = Path(__file__).resolve().parent.parent / "models"
MODELE = DOSSIER / "q_table.pkl"
JOURNAL = DOSSIER / "journal.csv"


def episode(env, agent: QAgent, apprendre: bool, noops: int = 0,
            afficher: bool = False) -> tuple[int, bool, float, int]:
    """Joue un épisode (pseudo-code QLearning du cours) ; renvoie (x_max, drapeau, G, nb_pas)."""
    env.reset()
    for _ in range(noops):                        # départ décalé : on attend sans rien faire
        step_k(env, 0, 1)
    a_prev = 0
    s = phi(env, a_prev)
    x_max, sans_progres, total, info = 0, 0, 0.0, {}
    for t in range(10_000):
        a = agent.act(s) if apprendre else agent.greedy_action(s)   # ε-greedy ou glouton
        r, terminated, truncated, info = step_k(env, a, K)           # r_{t+1}
        s_next = phi(env, a)                                         # s_{t+1}
        if afficher:
            env.render()                 # dessine l'image actuelle dans la fenêtre
            time.sleep(K / 60)           # le jeu tourne à 60 images/s : K frames = K/60 s
        if apprendre:
            agent.update(s, a, r, s_next, terminated)                # formule de Q-learning
        total += r
        if info["x_pos"] > x_max:
            x_max, sans_progres = info["x_pos"], 0
        else:
            sans_progres += 1
        s = s_next
        if terminated or truncated or sans_progres >= BLOCAGE:
            break
    return x_max, bool(info.get("flag_get", False)), total, t + 1


def charger(agent: QAgent) -> int:
    """Recharge Q et ε depuis MODELE s'il existe ; renvoie le nombre d'épisodes déjà faits."""
    if not MODELE.exists():
        return 0
    data = pickle.loads(MODELE.read_bytes())
    agent.Q, agent.epsilon = data["Q"], data["epsilon"]
    return data["episodes"]


def sauver(agent: QAgent, episodes: int) -> None:
    """Sauvegarde Q, ε et le nombre d'épisodes dans MODELE."""
    DOSSIER.mkdir(exist_ok=True)
    MODELE.write_bytes(pickle.dumps({"Q": agent.Q, "epsilon": agent.epsilon, "episodes": episodes}))


def entrainer(n_episodes: int) -> None:
    """Boucle d'entraînement : n_episodes épisodes, journal CSV et sauvegardes régulières."""
    env = make_env(render=False)
    agent = QAgent(len(ACTIONS), ALPHA, GAMMA, epsilon=1.0, seed=0)
    deja = charger(agent)
    print(f"Départ : {deja} épisodes déjà faits, {len(agent.Q)} états connus, ε = {agent.epsilon:.3f}")
    DOSSIER.mkdir(exist_ok=True)
    nouveau = not JOURNAL.exists()
    with JOURNAL.open("a", newline="") as f:
        journal = csv.writer(f)
        if nouveau:
            journal.writerow(["episode", "x_max", "drapeau", "G", "pas", "epsilon", "nb_etats"])
        ep = deja
        try:
            for ep in range(deja + 1, deja + n_episodes + 1):
                x, drapeau, g, pas = episode(env, agent, apprendre=True)
                agent.epsilon = max(EPS_MIN, agent.epsilon * EPS_DECAY)
                journal.writerow([ep, x, int(drapeau), round(g, 1), pas, round(agent.epsilon, 3), len(agent.Q)])
                print(f"ép {ep:5d} | x_max {x:4d} | drapeau {drapeau!s:5} | G {g:7.1f} | "
                      f"ε {agent.epsilon:.3f} | états {len(agent.Q)}")
                if ep % 25 == 0:
                    sauver(agent, ep)
                    f.flush()
                    xg, dg, _, _ = episode(env, agent, apprendre=False)
                    print(f"   → test glouton (ε = 0) : x_max {xg}, drapeau {dg}")
        except KeyboardInterrupt:
            print("\nInterrompu.")
        sauver(agent, ep)
        print(f"Sauvegardé : {MODELE} ({ep} épisodes)")
    env.close()


def jouer(noops: int) -> None:
    """Rejoue la politique gloutonne π(s) = argmax_a Q(s, a) avec la fenêtre ouverte."""
    env = make_env(render=True)
    agent = QAgent(len(ACTIONS), ALPHA, GAMMA, epsilon=0.0, seed=0)
    if charger(agent) == 0:
        print("Aucun modèle : lance d'abord l'entraînement.")
        return
    x, drapeau, g, pas = episode(env, agent, apprendre=False, noops=noops, afficher=True)
    print(f"x_max {x} | drapeau {drapeau} | G {g:.1f} | {pas} pas")
    env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--jouer", action="store_true", help="regarder la politique apprise")
    p.add_argument("--noops", type=int, default=0, help="frames d'attente au départ (avec --jouer)")
    args = p.parse_args()
    if args.jouer:
        jouer(args.noops)
    else:
        entrainer(args.episodes)