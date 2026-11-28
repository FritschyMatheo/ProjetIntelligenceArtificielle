"""Trace la carte des morts : où Mario meurt dans le niveau, pour plusieurs essais."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

MODELS_DIR = Path(__file__).with_name("models")
REPERES = {"trou 1": 1120, "trou 2": 1400, "trou 3": 2460}
NIVEAU_FIN = 3161


def read_deaths(path: Path, last: int | None) -> list[int]:
    """Renvoie le x_max des épisodes sans drapeau, c'est-à-dire l'endroit où Mario est mort."""
    with path.open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if last is not None:
        rows = rows[-last:]
    return [int(row["x_max"]) for row in rows if row["drapeau"] == "0"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="+", help="Noms des essais (models/<nom>.csv)")
    parser.add_argument("--bin", type=int, default=32, help="Largeur d'une tranche en pixels")
    parser.add_argument("--last", type=int, default=None, help="Ne garder que les N derniers épisodes")
    parser.add_argument("--output", default="morts.png", help="Image créée dans models/")
    args = parser.parse_args()

    bins = list(range(0, NIVEAU_FIN + args.bin, args.bin))
    figure, axis = plt.subplots(figsize=(12, 5))

    for name in args.names:
        deaths = read_deaths(MODELS_DIR / f"{name}.csv", args.last)
        axis.hist(deaths, bins=bins, histtype="step", linewidth=1.5, label=f"{name} ({len(deaths)} morts)")

    for label, x in REPERES.items():
        axis.axvline(x, linestyle="--", color="gray", linewidth=0.8)
        axis.text(x + 10, axis.get_ylim()[1] * 0.95, label, color="gray", fontsize=8)

    period = f"{args.last} derniers épisodes" if args.last else "tous les épisodes"
    axis.set_title(f"Carte des morts ({period})")
    axis.set_xlabel("position x dans le niveau")
    axis.set_ylabel(f"nombre de morts (tranches de {args.bin} px)")
    axis.legend()
    axis.grid(alpha=0.3)
    figure.tight_layout()

    output = MODELS_DIR / args.output
    figure.savefig(output, dpi=150)
    print(f"Carte des morts enregistrée dans {output}")
    plt.show()


if __name__ == "__main__":
    main()