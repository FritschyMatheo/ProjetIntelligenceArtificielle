"""Trace les courbes d'apprentissage de plusieurs essais à partir de leurs journaux CSV."""

import argparse
import csv
from itertools import accumulate
from pathlib import Path

import matplotlib.pyplot as plt

MODELS_DIR = Path(__file__).with_name("models")
REPERES = {"trou 1": 1130, "trou 2": 1420, "drapeau": 3161}


def read_journal(path: Path) -> tuple[list[int], list[int], list[int]]:
    """Lit un journal CSV et renvoie les épisodes, les x_max et les drapeaux (0 ou 1)."""
    episodes, x_max, flags = [], [], []
    with path.open(encoding="utf-8") as file:
        for row in csv.DictReader(file):
            episodes.append(int(row["episode"]))
            x_max.append(int(row["x_max"]))
            flags.append(int(row["drapeau"]))
    return episodes, x_max, flags


def moving_average(values: list[int], window: int) -> list[float]:
    """Moyenne glissante : moyenne des `window` derniers épisodes (moins au tout début)."""
    result = []
    total = 0
    for i, value in enumerate(values):
        total += value
        if i >= window:
            total -= values[i - window]
        result.append(total / min(i + 1, window))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="+", help="Noms des essais (models/<nom>.csv)")
    parser.add_argument("--window", type=int, default=50, help="Taille de la moyenne glissante")
    parser.add_argument("--output", default="courbes.png", help="Image créée dans models/")
    args = parser.parse_args()

    figure, (top, bottom) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    for name in args.names:
        episodes, x_max, flags = read_journal(MODELS_DIR / f"{name}.csv")
        top.plot(episodes, moving_average(x_max, args.window), label=name)
        bottom.plot(episodes, list(accumulate(flags)), label=f"{name} ({sum(flags)})")

    for label, x in REPERES.items():
        top.axhline(x, linestyle="--", color="gray", linewidth=0.8)
        top.text(1, x + 30, label, color="gray", fontsize=8)

    top.set_ylabel(f"x_max (moyenne sur {args.window} épisodes)")
    top.set_title("Courbes d'apprentissage")
    top.legend()
    top.grid(alpha=0.3)
    bottom.set_xlabel("épisode")
    bottom.set_ylabel("drapeaux cumulés")
    bottom.legend()
    bottom.grid(alpha=0.3)
    figure.tight_layout()

    output = MODELS_DIR / args.output
    figure.savefig(output, dpi=150)
    print(f"Courbe enregistrée dans {output}")
    plt.show()


if __name__ == "__main__":
    main()
