# 🗺️ Feuille de route SkaiBOT

État du socle : **le moteur, le pont et l'entraînement PPO de base fonctionnent** (vérifié). Les étapes ci-dessous sont les développements à ajouter, dans l'ordre de priorité qui correspond à ton objectif (1v1).

---

## Étape 0 bis — Environnement apprenable ✅ (fait le 2026-09-01)
Détails et mesures dans **[`DIAGNOSTIC.md`](DIAGNOSTIC.md)**.

- [x] `DefaultConfig` au lieu de `TestConfig` (conquête bridée à 1 tuile/tick supprimée).
- [x] Attaques réellement directionnelles (`sourceTile` transmis à `AttackExecution`).
- [x] `territory_pct` mesuré sur les **terres**, seuil de victoire atteignable.
- [x] Anti-cumul des attaques + `cancel_attacks` (fin de l'auto-appauvrissement).
- [x] Observation avec canal mer/terre séparé ; débit × 3.6 (100×100) et × 13 (500×500).
- [x] `skai/diagnose.py` : test de régression "l'env est-il apprenable ?" (code retour CI).
- [x] `--n-envs` (SubprocVecEnv, motif sûr Windows) et `--mask` (MaskablePPO).
- [x] Cartes réelles : `skai/make_maps.py` rapatrie l'Australie amont (2000×1500 + pyramides)
      et régénère `australia_100x100` / `australia_500x500` (auparavant 0 tuile terrestre).
- [x] `spawn_mode: "land"` + `spawn_seed` par épisode (le placement en anneau ignorait le relief).
- [x] **Boucle fermée** `skai/evaluate.py` (modèle vs glouton vs aléatoire, victoires/territoire/rang)
  et `skai/viewer/server.py --model <zip>` : le réseau entraîné joue enfin à l'écran
  (`skai/model_policy.py` reconstruit l'observation avec les fonctions de `rl_env`, remplacements
  de modèle possibles à chaud via `POST /api/model`). `--resume` + `--save-best` dans `train_simple.py`.
  Guide complet chiffré : `docs/BOUCLE.md`.
- [ ] Spawns réalistes (alignés sur le placement du jeu) et `gameMap` cohérent avec la carte.

## Étape 0 — Socle ⚠ (fait, mais voir Étape 0 bis : il rendait l'apprentissage impossible)
- [x] Moteur OpenFront exécutable en local.
- [x] Pont TS↔Python opérationnel (reset / tick / attaques directionnelles / état complet).
- [x] Entraînement PPO + action masking sur cartes Australie.
- [x] Visualisation en direct.

## Étape 1 — Self-play (priorité haute, pour le 1v1)
Objectif : que l'IA progresse contre **elle-même** plutôt que contre des bots aléatoires (c'est ce qui porte le niveau vers celui d'un humain — cf. AlphaGo/OpenAI Five).

- [ ] Terminer `agent/src/self_play_env.py` (désormais utile : l'espace d'action réagit) :
  - [ ] générer 2 joueurs "humains" dans le pont (au lieu de 1 agent + bots) ;
  - [ ] faire piloter chaque joueur par une politique (l'apprenant + une copie figée d'une version précédente) ;
  - [ ] mettre à jour l'adversaire figé toutes les N steps (pool d'anciens modèles pour éviter les cycles) ;
  - [ ] récompense : gagner = +1, perdre = −1, + shaping territorial.
- [ ] Script d'entraînement self-play dédié (ex. `agent/src/train_selfplay.py`).
- [ ] Suivi du niveau (taux de victoire contre les versions précédentes).

## Étape 2 — Speedrun (carte Australie, 0 nation)
> **Décision de calibrage à prendre** (mesures du 2026-09-01) : sur le continent entier
> (`australia_real_500x375`), une politique gloutonne plafonne à 9 % des terres en 3 000
> ticks et se fait éliminer à 20 joueurs. L'objectif « 100 % de la carte le plus vite »
> doit donc être exprimé soit en **tuiles**, soit en **% bas + horizon long**, sinon le RL
> n'a aucun gradient exploitable. Le mode « 0 nation » = `disableNPCs: true` (déjà
> compris par le moteur).
Objectif : conquérir 100 % de la carte **le plus vite possible** (min de ticks), en mode 400 tribus / 0 nation.

- [ ] Config de partie "0 nation" dans le pont (désactiver la création de nations, garder les tribus).
- [ ] Nouvelle fonction de récompense :
  - forte **pénalité par tick écoulé**,
  - bonus sur progression du % de carte possédée,
  - gros bonus à la conquête totale,
  - fin d'épisode à 100 % ou à une limite de ticks.
- [ ] Environnement dédié `skai/env_speedrun.py` (ou `agent/src/environment_speedrun.py`).
- [ ] Métrique de performance : ticks jusqu'à 100 % (le "temps" du speedrun).

## Étape 3 — 1v1 contre toi (objectif final)
Objectif : tu lances une partie privée auto-hébergée, tu joues dans ton navigateur, l'IA est ton adversaire.

- [ ] Comprendre le protocole réseau du serveur local (`base-game/src/server/`, messages client↔serveur).
- [ ] Connecter le pont à une partie **réseau** (pas seulement `createGame` en mémoire) : l'IA envoie les ordres comme un vrai joueur distant.
- [ ] Héberger : `npm run start:server` + client ; toi tu rejoins, l'IA se connecte comme 2ᵉ joueur.
- [ ] Script `skai/play_vs_human.py` qui charge un modèle entraîné et joue en temps réel contre toi
  (la brique `skai/model_policy.py` existe déjà : il ne reste que le pont « partie réseau »).
- [ ] (Sécurité) rester sur du **privé / auto-hébergé** : ne pas brancher l'IA sur le serveur public officiel.

## Étape 4 — Imitation à partir de tes parties
Objectif : accélérer l'apprentissage en lui montrant *comment tu joues*.

- [ ] Récupérer les replays OpenFront (le jeu enregistre les parties).
- [ ] Convertir un replay en séquences (état → action) pour de l'*imitation learning* / *behavioral cloning*.
- [ ] Pré-entraîner la politique sur tes speedruns, puis affiner en RL/self-play.

## Étape 5 — Modernisation & robustesse
- [ ] Aligner `base-game/` sur la dernière version officielle d'OpenFront (adapter le pont aux nouvelles interfaces).
- [ ] Finir les actions bâtiments/nukes dans le pont (phase 5.5 du projet d'origine).
- [ ] Tests automatisés du pont et des environnements.
- [ ] Notebook Google Colab pour les gros entraînements (GPU gratuit).
- [ ] CI légère (vérif du build Node + import Python).

---

## Conventions de développement pour le nouveau code
- Le code **repris** (socle validé) vit dans `base-game/` et `agent/` → on le touche avec précaution.
- Le code **nouveau** propre à SkaiBOT sera ajouté dans `skai/` au fur et à mesure :
  - `skai/env_selfplay.py`, `skai/env_speedrun.py`, `skai/train_selfplay.py`,
  - `skai/play_vs_human.py`, `skai/replays/` (imitation),
  - réutilisant `agent/game_bridge` et `agent/src/game_wrapper.py`.
- Toute modification du pont doit rester rétro-compatible avec les chemins relatifs (`../../base-game/...`).
