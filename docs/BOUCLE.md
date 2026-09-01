# 🔁 La boucle complète : entraîner → tester → réentraîner → regarder jouer

Toutes les commandes partent de la **racine du repo** (`SkaiBOT/`) et du venv de l'agent
(`source agent/venv/bin/activate`, ou `agent\venv\Scripts\activate` sous Windows).
Le moteur est headless : aucun navigateur n'est nécessaire pour entraîner.

---

## Étape 0 — Vérifier que l'environnement mérite un entraînement (≈ 1 min)

```bash
python skai/diagnose.py --fast --map australia_100x100
echo $?          # 0 = GO, 1 = NO-GO (ne pas entraîner, corriger l'environnement d'abord)
```

Pourquoi cette étape existe : avant les correctifs, **40 des 45 actions ne changeaient rien**
et la conquête était bridée à 1 tuile/tick — l'agent apprenait le vide. Les 5 tests
(`skai/diagnose.py`) verrouillent ce comportement ; `docs/DIAGNOSTIC.md` raconte l'enquête.

## Étape 1 — Entraîner

```bash
python skai/train_simple.py --steps 8192 --bots 5 --map australia_100x100 \
       --max-ticks 600 --save-best
```

| option | à quoi elle sert |
|---|---|
| `--steps` | pas d'entraînement (1 épisode ≈ 600 pas ici) → 8192 pas ≈ **14 épisodes** |
| `--bots` | adversaires (le pont crée `--bots + 1` joueurs) |
| `--map` | pack de terrain dans `base-game/resources/maps/` (`python skai/make_maps.py --list`) |
| `--max-ticks` | fin de partie anticipée ; **l'objectif du jeu réel reste 80 % des terres** |
| `--save-best` | écrit aussi `skai_ppo_<carte>_best.zip` à chaque record de territoire |
| `--mask` | MaskablePPO : ne propose que les directions réellement attaquables |
| `--n-envs N` | N moteurs en parallèle — utile **que** si tu as N cœurs libres |

Sorties :
- modèle : `skai/models/skai_ppo_australia_100x100.zip` (+ `_best.zip`)
- courbes : `skai/models/logs/` → `tensorboard --logdir skai/models/logs` (http://localhost:6006)

**Débit mesuré sur cette machine (2 cœurs) :** 8192 pas en 78 s (103 pas/s) ; 16 384 pas en
200 s (84 pas/s). Un vrai run de 10⁶ pas = **~3 h 20** ici, ~25 min sur 16 cœurs.

### Ce qu'il faut regarder dans les logs (et ce qu'il ne faut pas)

- `ep_rew_mean` qui monte = nécessaire, **pas suffisant**.
- `explained_variance` proche de 1 au bout de 2000 pas = **mauvais signe** : la récompense
  est prévisible au point de ne rien apprendre (ça nous est arrivé avec `TestConfig` :
  0.915 alors que l'agent ne savait pas où était l'eau).
- `ent_coef` / `entropy_loss` : l'entropy qui s'effondre en 3 minutes = l'agent a figé une
  habitude. À corriger avec `--ent-coef 0.05`, pas avec plus de pas.
- **le seul chiffre qui compte : le territoire et le rang** (imprimés par la callback toutes
  les 2 mises à jour).

## Étape 2 — Tester (chiffré, sans navigateur)

```bash
python skai/evaluate.py --model skai/models/skai_ppo_australia_100x100_best.zip \
       --map australia_100x100 --games 5 --max-ticks 600
```

`skai/evaluate.py` fait jouer **le modèle, la gloutonne de référence et l'aléatoire** dans le
même environnement d'entraînement (mêmes observations, mêmes règles) et imprime victoires /
territoire / rang / morts, plus un verdict. Options utiles :

- `--model-b avant.zip` : compare **avant / après** un réentraînement (détecte les régressions)
- `--win-threshold 0.35` : avec 600 ticks, 80 % est hors de portée → abaisser le seuil rend la
  colonne « victoires » lisible. **Ne pas confondre** avec le seuil du jeu (80 % en FFA).
- `--mask` : à mettre **uniquement** si le modèle a été entraîné avec `--mask`.

⚠️ **5 parties ne sont pas significatives.** La variance d'une partie à l'autre vaut ±10 points
de territoire. Garde ce mode pour repérer une régression grossière entre deux paliers ; pour un
chiffre publiable, vise `--games 20` (~20 min de plus) et compare toujours la **même** graine.

## Étape 3 — Réentraîner à partir du modèle

```bash
python skai/train_simple.py --resume skai/models/skai_ppo_australia_100x100.zip \
       --steps 16384 --bots 5 --map australia_100x100 --max-ticks 600 --save-best
```

`--resume` = `Algo.load(...)` + `learn(reset_num_timesteps=False)` : les poids, l'optimiseur et
le compteur global reprennent où ils en étaient (vérifié : 8 192 → 24 576 pas, itération 16 → 32).
Sur un plateau, baisse le Learning Rate (`--lr 1e-4`) plutôt que de repartir de zéro.

**Le piège qu'on a mesuré pour toi** (run du 2026-09-01, carte `australia_100x100`, 5 parties chacun) :

| modèle | `ep_rew_mean` | territoire à 600 ticks |
|---|---|---|
| après 8 192 pas | 65.4 | **46.4 %** |
| après 24 576 pas (réentraîné) | **67.5** ↑ | 37.4 % ↓ |

La récompense a monté, le niveau de jeu a baissé. C'est exactement pour ça que la boucle impose
`--save-best` **et** un `evaluate.py` à chaque palier : le checkpoint à garder est celui qui
gagne, pas le plus récent.

## Étape 4 — Regarder jouer (le maillon qui manquait)

```bash
python skai/viewer/server.py --map australia_100x100 --players 6 \
       --model skai/models/skai_ppo_australia_100x100_best.zip
# puis http://localhost:8080
```

- `--model` (ou la variable `SKAI_MODEL`) : la page affiche « IA : modèle … » et le panneau de
  droite montre la décision du réseau à chaque tick (ex. `attaque S à 30% des troupes libres`).
- changer de modèle **à chaud**, sans relancer le moteur :
  `curl -X POST "http://localhost:8080/api/model?path=skai/models/skai_ppo_australia_100x100.zip"`,
  et `?path=` (vide) pour revenir à la gloutonne de démo. Boutons équivalents dans la page.
- `--stochastic` : échantillonne la politique au lieu de prendre l'action maximale → visible
  pour juger l'exploration, pas seulement le meilleur coup.

Côté code, l'observation est construite par les fonctions **partagées** de `skai/rl_env.py`
(`land_mask_from` / `make_frame` / `global_feat` / `stack_frames`), appelées par `ModelPlayer`.
Un modèle excellent en log et nul à l'écran vient presque toujours d'un décalage
d'observation entre les deux mondes : ici il n'y a plus deux implémentations.

---

## Résultats complets de la boucle (2026-09-01, 2 cœurs CPU, `australia_100x100` réparé)

| évaluation | politique | territoire moyen | rang |
|---|---|---|---|
| 600 ticks, 5 parties | modèle `best` (PPO, 8 192 pas) | **41.5 %** | 1.00 |
| | gloutonne de référence | 33.5 % | 1.00 |
| | aléatoire | 39.3 % | 1.00 |
| 400 ticks, seuil 35 %, 3 parties | modèle après 24 576 pas | 33.7 % | 2/3 victoires |
| | modèle après 8 192 pas | 34.2 % | 1/3 victoires |
| **transfert** sur `australia_real_500x375` (continent entier, jamais vu à l'entraînement), 600 ticks, 2 parties | modèle `best` | **14.4 %** | **1.50** |
| | gloutonne | 6.4 % | 3.50 |
| | aléatoire | 6.2 % | 6.00 |

À lire honnêtement : sur la carte d'entraînement le gain est réel mais modeste (et l'aléatoire
tient la comparaison sur un horizon de 400-600 ticks), tandis que **sur le continent le modèle
fait plus que doubler la gloutonne et finit 1er en moyenne** — c'est là que l'apprentissage
apporte quelque chose, pas sur le petit terrain. 8 192 pas, c'est un aller-retour de validation
de la boucle, pas un entraînement.

## Ordre de travail recommandé

1. `diagnose.py` → 2. petit run (`--steps 8192 --save-best`) → 3. `evaluate.py --games 5`
→ 4. `--resume` × 3 → 5. `evaluate.py --model nouveau --model-b ancien` → 6. `viewer --model`
→ 7. seulement quand la gloutonne est battue sur le continent : passer à
`--map australia_real_500x375 --max-ticks 3000` et au self-play (ROADMAP étape 1).
