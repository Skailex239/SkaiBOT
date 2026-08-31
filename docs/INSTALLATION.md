# 🛠️ Installation pas-à-pas (vérifiée le 2026-08-31)

## Prérequis (une seule fois)

1. **Node.js 20 LTS** → https://nodejs.org (fournit aussi `npm`).
2. **Python 3.11 ou 3.12** (⚠️ PAS 3.13 : le projet fige `numpy<2.0`).
   → https://www.python.org/downloads/ — coche *"Add Python to PATH"* à l'installation.
3. **Git** → https://git-scm.com.

## Étape 1 — Récupérer le projet

```bash
git clone https://github.com/Skailex239/SkaiBOT.git
cd SkaiBOT
```

## Étape 2 — Installer et compiler le moteur

```bash
cd base-game
npm install
npm run build-dev
```
✅ Attendu : `webpack ... compiled successfully` (1 à 2 min). Les avertissements "vulnerabilities" sont normaux, ignore-les.

## Étape 3 — Tester le pont IA ↔ moteur

```bash
cd ../agent
echo '{"type":"reset","map_name":"australia_100x100_nt","num_players":11}' | npx tsx game_bridge/game_bridge.ts
```
✅ Attendu :
```
[GameBridge] Ready for commands
[GameBridge] Game initialized: map=australia_100x100_nt ...
[GameBridge] RL Agent starting with 52 tiles, 1 cluster(s)
{"success":true,"state":{...}}
```
Fais `Ctrl+C` pour arrêter.

## Étape 4 — Installer les dépendances Python de l'IA

```bash
cd agent
python -m venv venv
# Windows (PowerShell) :
venv\Scripts\activate
# Mac/Linux :
# source venv/bin/activate

pip install -r requirements.txt
```
(Sous Windows ou sans GPU, c'est la version CPU de PyTorch qui s'installe.)

## Étape 5 — Premier entraînement (CPU)

```bash
cd src
python train_gpu.py --map australia_100x100_nt --n-envs 4 --total-timesteps 50000
```
Suivre l'apprentissage (autre terminal, `venv` activé, dossier `agent/`) :
```bash
tensorboard --logdir runs
```
Puis ouvre http://localhost:6006 : la récompense doit monter progressivement.

## Étape 6 — Regarder l'IA jouer

```bash
cd agent
python src/visualize_realtime.py --model src/runs/run_XXXX/checkpoints/best_model.zip --map australia_100x100_nt --crop none
```
(remplace `run_XXXX` par le dossier créé par l'entraînement)

---

## 🚩 Pièges connus

| Problème | Solution |
|---|---|
| `npm run build` introuvable | Utilise `npm run build-dev` (le script `build` n'existe pas dans cette version). |
| Erreur à l'install Python (numpy) | Tu es en Python 3.13 → installe Python 3.11 ou 3.12. |
| `ModuleNotFoundError: sb3_contrib` | `pip install sb3-contrib` (déjà présent dans `requirements.txt`). |
| Pont lent à démarrer | Normal au lancement (chargement de la carte) ; les cartes 100×100 sont plus rapides. |
| Le pont ne trouve pas la carte | Vérifie que tu lances bien depuis `agent/` et que `base-game/resources/maps/` contient bien la carte. |

## 💡 Bonnes pratiques CPU

- Entraîne sur `australia_100x100_nt` ou `australia_256x256`.
- Garde `--n-envs 4` sur un PC classique.
- Pour de gros entraînements (grande carte, self-play), un GPU divise le temps par ~10-50 : on préparera un notebook Google Colab.
