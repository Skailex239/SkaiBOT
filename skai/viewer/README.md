# 🎥 Visualiseur en direct (skai/viewer)

Une petite page web pour **regarder l'IA jouer** une partie OpenFront en local, sans aucune
dépendance Python (bibliothèque standard uniquement).

## Lancer

Prérequis : avoir fait `npm install` dans `base-game/` (voir `docs/INSTALLATION.md`).

```bash
# depuis la racine du repo SkaiBOT
python skai/viewer/server.py
# puis ouvrir l'URL affichée :  http://localhost:8080
```

## Ce qu'on voit

- La carte (Australie) avec le territoire de **notre IA en vert**, les **adversaires en couleurs**,
  le terrain neutre en beige et l'eau en fond sombre.
- Rang, tick, troupes, % de territoire, et les **décisions de l'IA** à chaque tick.
- Bouton **↻ Relancer** pour changer de carte (100×100, 256×256) et de nombre de joueurs.

## Comment ça marche

- `bridge.py` : pilote le pont TypeScript (`agent/game_bridge/game_bridge.ts`) en JSON via stdin/stdout.
- `agent_policy.py` : pour l'instant une **politique d'expansion gloutonne** (attaquer le terrain libre
  autour, en gardant des troupes en réserve). C'est une IA de démonstration.
- `server.py` : petit serveur HTTP qui fait tourner la partie en boucle (~8 ticks/s) et expose
  `/api/state` (carte compressée RLE + statistiques), consommé par `index.html`.

## Suite

Quand les modèles RL/self-play seront entraînés (`docs/ROADMAP.md` étapes 1 et 2),
`agent_policy.py` sera remplacé par l'appel au réseau de neurones (`model.predict(obs)`) :
la même page servira à regarder la **vraie** IA.
