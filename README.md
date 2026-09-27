# 🤖 SkaiBOT

**SkaiBOT** est une intelligence artificielle qui apprend à jouer à **OpenFront.io** (RTS de conquête territoriale), d'abord en s'entraînant toute seule en local, puis pour faire des **speedruns sur la carte Australie** et enfin pour **jouer en 1 contre 1 contre toi**.

> 🎯 Objectif à terme : une IA que tu peux défier en 1v1 sur une partie hébergée chez toi, et qui bat des records de vitesse sur la carte Australie (400 tribus, 0 nation).

---

## ✅ Ce qui fonctionne déjà (socle technique vérifié)

- Moteur du jeu OpenFront intégré et exécutable en local, **sans navigateur** (`base-game/`).
- **Pont TypeScript ↔ Python** qui pilote le moteur et échange l'état du jeu / les ordres en JSON (`agent/game_bridge/`).
- Entraînement par renforcement **PPO** (Stable-Baselines3). Le chemin `skai/` applique
  les **vraies règles d'OpenFront** (`DefaultConfig`), des **attaques réellement
  directionnelles** et un *action masking* optionnel (`--mask`, `sb3-contrib`).
  ⚠ Lire **[`docs/DIAGNOSTIC.md`](docs/DIAGNOSTIC.md)** avant tout entraînement long :
  le socle "vérifié" de la v1 rendait l'apprentissage **impossible** (env qui ne réagissait
  pas aux actions) ; c'est corrigé, et `python skai/diagnose.py` le revérifie en 1 minute.
- Observation spatiale multi-échelles + mécanismes d'attention (type AlphaStar) et détection des "grappes" de territoire.
- Cartes **Australie** incluses — **à l'état inégal** (vérifiées dans le `.bin`) :

  | carte | terres | utilisable ? |
  |---|---|---|
  | `australia_500x500_nt` | 212 707 (85.1 %) | ✅ **recommandée** (vraie géographie avec océan) |
  | `australia_1024x1024` | 723 165 (69.0 %) | ✅ mais lente, GPU/Colab conseillés |
  | `australia_100x100_nt`, `australia_256x256` | 100 % de la grille | ⚠ **aucun océan** : ce ne sont pas les cartes "Australie", mais des variantes "no terrain" — OK pour déboguer, pas pour apprendre à jouer |
  | `australia_100x100`, `australia_500x500` | — | 🔧 **régénérées** par `skai/make_maps.py --repair` : vrais recadrages de l'Australie (60 % et 83 % de terres) |
  | `australia_real_500x375` / `_1000x750` / `_2000x1500` | 43–44 % | ✅ **la vraie carte entière**, générée depuis OpenFrontIO/main |

  ### 🌏 La vraie carte d'Australie
  ```bash
  python skai/make_maps.py --sizes 500,1000,2000 --repair   # récupère + génère tout
  python skai/make_maps.py --list                            # vérifie le résultat
  ```
  Ensuite : `python skai/diagnose.py --map australia_100x100` puis
  `python skai/train_simple.py --map australia_100x100 --max-ticks 3000`.
  Entraîne sur la 100×100 (vraie géographie, ~620 ticks/s) et garde le continent entier
  pour l'évaluation : à cette échelle, 80 % des terres en 1 500 ticks n'est pas un objectif.
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
├── skai/                     ← outillage Python (entraînement, évaluation, visualisation…)
│   ├── ratio_lab/            ← ⚔️ le bot qui teste tous les ratios d'attaque (voir plus bas)
│   └── viewer/               ← visualiseur en direct dans le navigateur
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

### 4. Vérifier que l'environnement est apprenable, PUIS entraîner
```bash
cd ..                     # racine du repo (SkaiBOT/)
python skai/diagnose.py --fast --map australia_100x100     # verdict GO/NO-GO en ~1 min

# 1) entraîner (court d'abord : ~80 s pour 8 192 pas sur 2 cœurs)
python skai/train_simple.py --steps 8192 --bots 5 --map australia_100x100 --max-ticks 600 --save-best

# 2) tester contre la gloutonne et l'aléatoire (chiffré, sans navigateur)
python skai/evaluate.py --model skai/models/skai_ppo_australia_100x100_best.zip --games 5

# 3) réentraîner à partir du modèle, puis re-comparer
python skai/train_simple.py --resume skai/models/skai_ppo_australia_100x100.zip --steps 16384 \
       --bots 5 --map australia_100x100 --max-ticks 600 --save-best
python skai/evaluate.py --model skai/models/skai_ppo_australia_100x100.zip --model-b skai/models/skai_ppo_australia_100x100_best.zip

# 4) le regarder jouer dans le navigateur
python skai/viewer/server.py --map australia_100x100 --players 6 \
       --model skai/models/skai_ppo_australia_100x100_best.zip     # http://localhost:8080
```

Suivi de l'apprentissage : `tensorboard --logdir skai/models/logs` (ouvrir http://localhost:6006).
👉 **La boucle entière, détaillée avec les chiffres mesurés et les pièges : [`docs/BOUCLE.md`](docs/BOUCLE.md).**

👉 Le guide détaillé (pièges inclus) : **[`docs/INSTALLATION.md`](docs/INSTALLATION.md)**.

---

## ⚔️ RatioLab — le bot qui teste tous les ratios d'attaque

**[`skai/ratio_lab/`](skai/ratio_lab/)** balaye toutes les combinaisons
**ratio d'attaque (%) × cadence d'envoi** sur une **zone totalement verte**
(1 joueur, vraies règles `DefaultConfig`) et mesure la **vitesse de conquête
réelle** de chaque combinaison. Le temps de chaque test est configurable
(défaut : 1 minute de jeu), et une petite **UI web** affiche les **records**.

```bash
# 1) lancer le balayage (≈ 70 combinaisons, 1 min de jeu chacune)
python skai/ratio_lab/sweep.py

# 2) ouvrir le tableau des records
python skai/ratio_lab/server.py          # http://localhost:8080
```

L'UI permet aussi de **lancer un balayage à chaud** (ratios, cadences, durée
configurables) et d'afficher : record absolu, top 10, historique des records,
matrice ratio × cadence, courbes de vitesse. Détails :
**[`skai/ratio_lab/README.md`](skai/ratio_lab/README.md)**.

---

## 🧭 Notes techniques importantes

- **Pas de navigation web pour l'entraînement** : l'IA parle directement au moteur via un pont Node. C'est 100 à 1000× plus rapide qu'automatiser un navigateur, et ça permet d'entraîner des dizaines de parties en parallèle. Le mode navigateur ne servira que pour le 1v1 final contre toi, en partie **privée et auto-hébergée**.
- **CPU** : ~500 ticks/s sur 100×100, ~160 ticks/s sur 500×500 (`obs_size: 64`). Le débit suit
  les **cœurs physiques** : `--n-envs N` ne sert à rien au-delà du nombre de cœurs.
- Un épisode dure ~1 400 ticks : **16 000 pas PPO ≈ 12 épisodes**. Pour un signal net,
  vise 10⁶ pas × 8–16 cœurs (Colab/VM), sinon reste sur la 100×100 en assumant qu'il n'y a
  pas d'océan.
- **Ne pas mettre le moteur à jour tout de suite** : la copie dans `base-game/` est alignée avec le pont de l'agent. La modernisation se fera à l'étape 5 de la feuille de route.

## 🙏 Remerciements & licences

- Le moteur vient du projet open source **[OpenFront.io](https://github.com/openfrontio/OpenFrontIO)**.
- Le socle d'apprentissage par renforcement est repris/remis au propre depuis le projet de recherche **[Reinforcement-Learning-Agent-OpenFront.io](https://github.com/Alexis-CAPON/Reinforcement-Learning-Agent-OpenFront.io)** d'Alexis CAPON, puis réorganisé et étendu pour SkaiBOT.
- Licences du moteur : code `src/core/` et `src/server/` sous **MIT** ; code `src/client/` sous **GPL v3**. Voir `base-game/LICENSE`, `base-game/LICENSING.md`. Le code Python de l'agent est fourni pour l'usage personnel et la recherche.
