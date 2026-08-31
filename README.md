# 🤖 SkaiBOT

**SkaiBOT** est une intelligence artificielle qui apprend à jouer à **OpenFront.io** (RTS de conquête territoriale), d'abord en s'entraînant toute seule en local, puis pour faire des **speedruns sur la carte Australie** et enfin pour **jouer en 1 contre 1 contre toi**.

> 🎯 Objectif à terme : une IA que tu peux défier en 1v1 sur une partie hébergée chez toi, et qui bat des records de vitesse sur la carte Australie (400 tribus, 0 nation).

---

## ✅ Ce qui fonctionne déjà (socle technique vérifié)

- Moteur du jeu OpenFront intégré et exécutable en local, **sans navigateur** (`base-game/`).
- **Pont TypeScript ↔ Python** qui pilote le moteur et échange l'état du jeu / les ordres en JSON (`agent/game_bridge/`).
- Entraînement par renforcement **PPO** (Stable-Baselines3) avec *action masking* : l'IA n'a le droit qu'aux actions valides (`agent/src/`).
- Observation spatiale multi-échelles + mécanismes d'attention (type AlphaStar) et détection des "grappes" de territoire.
- Cartes **Australie** déjà incluses : `australia`, `australia_100x100`, `australia_100x100_nt`, `australia_256x256`, `australia_500x500`, `australia_500x500_nt`, `australia_1024x1024`.
- Visualisation en direct des décisions de l'IA.

## 🚧 Ce qui reste à construire (la feuille de route est dans `docs/ROADMAP.md`)

1. **Self-play** : l'IA joue contre elle-même pour atteindre un vrai niveau (ossature présente dans `agent/src/self_play_env.py`, à terminer).
2. **Speedrun** : récompense basée sur la *vitesse* de conquête + configuration 400 tribus / 0 nation.
3. **1v1 contre un humain** : brancher l'IA comme joueur sur le serveur local du jeu, que tu rejoins avec ton navigateur.
4. **Imitation** : apprendre à partir de *tes* replays.
5. **Modernisation** : aligner le moteur sur la dernière version officielle d'OpenFront.

---

## 📂 Structure du dépôt

```
SkaiBOT/
├── README.md                 ← ce fichier
├── .gitignore
│
├── base-game/                ← Moteur OpenFront (TypeScript) qui sert d'environnement
│   ├── src/core/             ← logique du jeu (MIT)
│   ├── src/server/           ← serveur de parties (MIT)
│   ├── src/client/           ← client navigateur (GPL v3)
│   └── resources/maps/       ← cartes (seules les variantes Australie sont incluses ici)
│
├── agent/                    ← L'IA d'apprentissage par renforcement (Python)
│   ├── game_bridge/          ← pont TypeScript qui pilote le moteur (JSON via stdin/stdout)
│   ├── src/
│   │   ├── game_wrapper.py          ← communication Python ↔ pont Node
│   │   ├── environment_*.py         ← environnements Gym pour l'entraînement
│   │   ├── model_multiscale_*.py    ← réseau de neurones (CNN + attention)
│   │   ├── self_play_env.py         ← ⚙️ self-play (à terminer)
│   │   └── train_gpu.py             ← script d'entraînement
│   ├── notebooks/            ← carnets Jupyter (entraînement / suivi)
│   └── requirements.txt
│
└── docs/                     ← documentation en français
    ├── INSTALLATION.md       ← installation pas-à-pas
    ├── ARCHITECTURE.md       ← comment tout communique
    └── ROADMAP.md            ← feuille de route + specs (self-play, 1v1, speedrun)
```

---

## ⚡ Démarrage rapide

### 1. Installer le moteur
```bash
cd base-game
npm install
npm run build-dev        # attendre : "webpack ... compiled successfully"
```

### 2. Vérifier que le pont IA ↔ jeu tourne
```bash
cd ../agent
echo '{"type":"reset","map_name":"australia_100x100_nt","num_players":11}' | npx tsx game_bridge/game_bridge.ts
# attendre "[GameBridge] Game initialized ..." puis Ctrl+C
```

### 3. Installer l'IA (Python 3.11 ou 3.12 recommandé)
```bash
cd agent
python -m venv venv
# Windows PowerShell :  venv\Scripts\activate
# Mac/Linux :           source venv/bin/activate
pip install -r requirements.txt
```

### 4. Lancer un premier entraînement (CPU)
```bash
cd src
python train_gpu.py --map australia_100x100_nt --n-envs 4 --total-timesteps 50000
```

Suivi de l'apprentissage : `tensorboard --logdir runs` (ouvrir http://localhost:6006).

👉 Le guide détaillé (pièges inclus) : **[`docs/INSTALLATION.md`](docs/INSTALLATION.md)**.

---

## 🧭 Notes techniques importantes

- **Pas de navigation web pour l'entraînement** : l'IA parle directement au moteur via un pont Node. C'est 100 à 1000× plus rapide qu'automatiser un navigateur, et ça permet d'entraîner des dizaines de parties en parallèle. Le mode navigateur ne servira que pour le 1v1 final contre toi, en partie **privée et auto-hébergée**.
- **CPU** : entraîne-toi sur `australia_100x100_nt` / `australia_256x256`. Pour les grandes cartes, utiliser un GPU (local ou Google Colab gratuit).
- **Ne pas mettre le moteur à jour tout de suite** : la copie dans `base-game/` est alignée avec le pont de l'agent. La modernisation se fera à l'étape 5 de la feuille de route.

## 🙏 Remerciements & licences

- Le moteur vient du projet open source **[OpenFront.io](https://github.com/openfrontio/OpenFrontIO)**.
- Le socle d'apprentissage par renforcement est repris/remis au propre depuis le projet de recherche **[Reinforcement-Learning-Agent-OpenFront.io](https://github.com/Alexis-CAPON/Reinforcement-Learning-Agent-OpenFront.io)** d'Alexis CAPON, puis réorganisé et étendu pour SkaiBOT.
- Licences du moteur : code `src/core/` et `src/server/` sous **MIT** ; code `src/client/` sous **GPL v3**. Voir `base-game/LICENSE`, `base-game/LICENSING.md`. Le code Python de l'agent est fourni pour l'usage personnel et la recherche.
