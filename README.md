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