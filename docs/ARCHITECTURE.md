# 🏗️ Architecture de SkaiBOT

## Vue d'ensemble : trois couches

```
        ┌──────────────────────────────────────────────┐
        │                AGENT IA (Python)             │
        │  Stable-Baselines3 (PPO) + réseau CNN/attn   │
        │  agent/src/environment_*.py  (obs/récompense)│
        │  agent/src/model_*.py        (cerveau)       │
        └───────────────┬──────────────────────────────┘
                        │  commandes JSON (stdin)  ▲  état JSON (stdout)
                        ▼                           │
        ┌──────────────────────────────────────────────┐
        │          PONT TypeScript (Node / tsx)        │
        │  agent/game_bridge/game_bridge.ts            │
        │  - crée la partie, applique les ordres       │
        │  - lit l'état, détecte les grappes           │
        └───────────────┬──────────────────────────────┘
                        │  appels directs (imports TypeScript)
                        ▼
        ┌──────────────────────────────────────────────┐
        │          MOTEUR OpenFront (TypeScript)       │
        │  base-game/src/core  (Game, tuiles, troupes) │
        │  base-game/src/server (parties réseau)       │
        │  base-game/src/client (rendu navigateur)     │
        └──────────────────────────────────────────────┘
```

## Pourquoi pas de navigateur pour s'entraîner ?

Le moteur d'OpenFront est du TypeScript exécutable **directement côté serveur/Node**, sans rendu graphique. Le pont instancie une partie avec `createGame(...)`, y place l'agent (joueur humain piloté par le code) et des bots, puis fait avancer la partie tick par tick. Conséquences :

- **Vitesse** : pas de rendu, pas de réseau → on peut enchaîner les parties ~100 à 1000× plus vite que le temps réel et lancer plusieurs environnements en parallèle (`--n-envs`).
- **Accès complet à l'état** : tuiles possédées, troupes par tuile, frontières, grappes déconnectées, bâtiments, or, etc.
- **Déterministe/reproductible** : idéal pour l'apprentissage.

Le navigateur n'est réintroduit qu'à la fin, pour le **1v1 contre toi** : on héberge une partie locale avec le serveur du jeu, toi tu joues via le client web, et l'IA se connecte comme un joueur (via le même type de pont, mais branché sur la partie réseau).

## Le pont de communication (`agent/game_bridge/game_bridge.ts`)

Processus Node lancé par Python via `subprocess` (`npx tsx game_bridge.ts`). Il dialogue ligne par ligne en **JSON sur stdin/stdout** :

- Commandes entrantes (Python → TS) :
  - `reset` : crée une nouvelle partie (`map_name`, `num_players`).
  - `tick` : avance la simulation.
  - `attack_direction` : ordonne une attaque (`cluster_id`, `direction` 0-8, `intensity` 0-1).
  - `build_unit`, `launch_nuke` : bâtiments / armes (en cours de finition).
  - `get_state`, `shutdown`.
- État sortant (TS → Python) : `GameState` (voir interface dans le fichier `.ts`) :
  - `territory_map` (2D : 0 neutre, 1 agent, 2+ adversaires), `troop_map`,
  - `clusters` (grappes du territoire, flood-fill, jusqu'à 5),
  - caractéristiques globales (or, population, rang, bâtiments, menaces…).

Chemins importants :
- Cartes chargées depuis `base-game/resources/maps/<nom>/` (chemin relatif `../../base-game/...`, d'où le fait que le dossier garde le nom `base-game`).
- Le wrapper Python : `agent/src/game_wrapper.py` (lance le process, envoie les commandes, parse le JSON).

## L'agent d'apprentissage

- **Algorithme** : PPO avec *action masking* (`MaskablePPO` de `sb3-contrib`) → l'IA ne peut pas choisir d'actions interdites (ex. attaquer une grappe qui n'existe pas).
- **Observation** :
  - carte de propriété des tuiles,
  - forces (troupes) par tuile,
  - distance à la frontière / pression ennemie,
  - empilement des 4 dernières images (contexte temporel),
  - architecture **multi-échelles + attention** : vue globale (carte réduite), vue locale (autour du territoire), vue tactique (frontières).
- **Espace d'actions** : pour chaque grappe `[0-4]` × direction `[0-8]` × intensité `[0-4]` → 225 actions de base (315+ avec les bâtiments).
- **Récompense actuelle** (à adapter pour le speedrun, voir `docs/ROADMAP.md`) :
  territoire conquis, croissance de population, survie, éliminations, pénalité de mort.

## Fichiers clés

| Rôle | Fichier |
|---|---|
| Pont TS | `agent/game_bridge/game_bridge.ts` |
| Wrapper Python | `agent/src/game_wrapper.py` |
| Environnements Gym | `agent/src/environment.py`, `environment_large.py`, `environment_multiscale.py`, `environment_full_game.py` |
| Réseaux de neurones | `agent/src/model.py`, `model_multiscale.py`, `model_multiscale_attention.py` |
| Entraînement | `agent/src/train_gpu.py`, `agent/train_full_game.py` |
| Self-play (à terminer) | `agent/src/self_play_env.py` |
| Visualisation | `agent/src/visualize_realtime.py` |
| Création de parties | `base-game/src/core/game/GameImpl.ts` (`createGame`) |
| Exécution d'ordres | `base-game/src/core/execution/AttackExecution.ts`, `SpawnExecution.ts` |
