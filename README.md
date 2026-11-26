# ProjetIntelligenceArtificielle

Environnement : https://github.com/Kautenja/gym-super-mario-bros (sous-module dans `external/`, licence MIT, émulateur NES nes-py, API Gymnasium).

## Installation (Debian 12)
```bash
sudo apt update
sudo apt install -y git curl libgl1 libglu1-mesa
curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env
git clone --recurse-submodules https://github.com/FritschyMatheo/ProjetIntelligenceArtificielle
cd ProjetIntelligenceArtificielle
uv venv --python 3.13 --seed .venv
source .venv/bin/activate
pip install -r requirements.txt
```
## Installation (MacOS)
```bash
brew install git curl mesa
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
git clone --recurse-submodules https://github.com/FritschyMatheo/ProjetIntelligenceArtificielle
cd ProjetIntelligenceArtificielle
uv venv --python 3.13 --seed .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Jouer
```bash
source .venv/bin/activate
python Mario/mario.py
```
Flèches : bouger | Espace : sauter | X : courir | Entrée : start | Échap : quitter


## Premier agent : Q-learning
L'agent apprend sur le niveau 1-1 avec une Q-table. Il utilise les récompenses fournies par Gymnasium pour la progression, les pièces, les bonus, la mort et la fin du niveau. La table et le nombre d'épisodes d'entraînement sont sauvegardés dans `Mario/q_table.json`, puis rechargés pour poursuivre l'apprentissage ou jouer avec la politique apprise.
```bash
source .venv/bin/activate
python Mario/q_learning.py --episodes 500
```
L'entraînement est sans affichage par défaut pour accélérer les essais. Pour afficher une partie après l'entraînement :
```bash
python Mario/q_learning.py --play --render
```
L'affichage est limité à 30 images par seconde pour laisser le temps de suivre Mario. Pour ralentir davantage : `python Mario/q_learning.py --play --render --fps 12`.
Pour un essai court : `python Mario/q_learning.py --episodes 5 --steps 300`. L'option `--render` affiche aussi l'entraînement, mais le ralentit. Cette première version apprend sur le niveau 1-1 ; réussir le niveau n'est pas garanti et nécessitera plusieurs entraînements et ajustements. L'ancienne Q-table ne contenant pas de compteur, le décompte exact commencera à la prochaine session d'entraînement.