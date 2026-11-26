"""Agent Q-learning tabulaire (cours 3, diapos 32 et 41-42)."""
from collections.abc import Hashable

import numpy as np

Etat = Hashable  # un état s : tout objet hachable (ex. un tuple d'entiers)


class QAgent:
    """Agent Q-learning : table Q(s, a), politique ε-greedy, mise à jour TD."""

    def __init__(self, n_actions: int, alpha: float, gamma: float,
                 epsilon: float, seed: int | None = None) -> None:
        """Stocke |A|, α (apprentissage), γ (actualisation), ε (exploration).

        self.Q : dictionnaire état s -> vecteur numpy des |A| valeurs Q(s, ·).
        """
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)
        self.Q: dict[Etat, np.ndarray] = {}

    def q_values(self, s: Etat) -> np.ndarray:
        """Renvoie Q(s, ·) ; si s n'a jamais été vu, crée Q(s, a) = 0 pour toute action a."""
        if s not in self.Q:
            self.Q[s] = np.zeros(self.n_actions)
        return self.Q[s]

    def greedy_action(self, s: Etat) -> int:
        """Renvoie arg max_a Q(s, a) ; égalités départagées au hasard avec self.rng."""
        q = self.q_values(s)
        meilleures = np.flatnonzero(q == q.max())
        return int(self.rng.choice(meilleures))

    def act(self, s: Etat) -> int:
        """ε-greedy : avec proba ε une action au hasard parmi les |A|, sinon greedy_action(s)."""
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return self.greedy_action(s)

    def update(self, s: Etat, a: int, r: float, s_next: Etat, terminated: bool) -> float:
        """Q(s,a) ← Q(s,a) + α · δ  avec  δ = r + γ · max_a' Q(s',a') − Q(s,a).

        Le max vaut 0 si s' est terminal. Renvoie δ.
        """
        q = self.q_values(s)
        futur = 0.0 if terminated else float(self.q_values(s_next).max())
        delta = r + self.gamma * futur - q[a]
        q[a] += self.alpha * delta
        return float(delta)


if __name__ == "__main__":
    ag = QAgent(n_actions=3, alpha=0.1, gamma=0.9, epsilon=0.1, seed=0)
    a = ag.act("s0")
    delta = ag.update("s0", a, 1.0, "s1", terminated=False)
    print("action :", a, " δ :", delta, " Q(s0) :", ag.Q["s0"])