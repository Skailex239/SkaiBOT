# 🔬 Diagnostic SkaiBOT — pourquoi l'IA n'apprenait pas

Document établi le 2026-09-01 après **exécution réelle** du pipeline complet dans un
environnement propre (Node 22.22.3, Python 3.11, gymnasium 1.3, SB3 2.9.0, CPU).
Tous les chiffres ci-dessous sont des mesures, pas des suppositions ; ils sont
reproductibles avec `python skai/diagnose.py`.

> **Verdict.** Le code n'était pas « mal réglé » : l'environnement ne réagissait pas
> aux actions de l'agent. Dans cet état, **aucun nombre de pas, aucun algorithme, aucun
> GPU ne pouvait faire apprendre quoi que ce soit**. Cinq causes, toutes corrigées.

---

## 0. Ce qui a été mesuré AVANT toute modification

| Observation | Mesure |
|---|---|
| PPO sur `skai/train_simple.py` (1200 pas) | `ep_rew_mean` ≈ **54.6**, `ep_len_mean` = **1500** (jamais terminé), territoire final **5.5 %** |
| Territoire au cours de la partie | **+1.0 % tous les 100 ticks, exactement**, quelle que soit la politique |
| 4 politiques très différentes (attaque 1 tick sur 1 / 1 sur 3 / 1 sur 5 / jamais) | 0.5 %, 19 %, 27 %, 30 % au lieu de spread attendu… mais **les mêmes 8 directions donnaient un territoire identique** |
| Sensibilité à la direction | Dir **E, S, W → résultats strictement identiques** : 4.5 %, bbox `x[77,98] y[37,57]` au bit près |
| Victoire | `has_won` **jamais déclenché** en 1500 ticks |
| Débit | ~**123 pas/s** côté PPO, 140 ticks/s côté pont (100×100), **6–12 ticks/s** sur `australia_500x500_nt` |

---

## 1. Cinq causes racines

### ① Le pont utilisait la config de jeu des TESTS UNITAIRES (le blocage principal)

`agent/game_bridge/game_bridge.ts` créait la partie avec `TestConfig` — le harnais
Jest de OpenFront, dont les règles de combat sont volontairement ** dégénérées pour être
déterministes en test ** :

```ts
// base-game/tests/util/TestConfig.ts
attackTilesPerTick(...) => 1                    // réel : numAdjacent*2 vs neutre, ~0.5*numAdjacent*3 vs un joueur
attackLogic(...)        => 1 perte / 1 tuile / 1 tick
samHittingChance()      => 1                     // tous les missiles sont toujours interceptés
nukeMagnitudes()        => {inner: 1, outer: 1}  // un nuke = 1 tuile
```

Conséquence directe et fatale : **une attaque conquiert 1 tuile par tick, point**.
Sur une carte 100×100, un épisode de 1500 ticks ne peut donc pas dépasser ~15 % de la
carte — d'où le +1 %/100 ticks parfaitement linéaire mesuré plus haut, et le bonus de
victoire inatteignable. La récompense `(Δterritoire × 200)` était une **dérive constante
indépendante des actions** : le gradient de politique était nul. L'`explained_variance`
à 0.91 n'était pas un bon signe : la fonction de valeur ajustait une droite.

✅ **Corrigé** : le pont utilise `DefaultConfig` (les vraies règles), avec une option
`game_config: "test"` pour revenir à l'ancien comportement.

### ② Les attaques directionnelles n'existaient pas

OpenFront n'a **pas** d'attaque directionnelle : l'intention réseau est
`{type:"attack", targetID, troops}` (cf. `base-game/src/core/Schemas.ts`).
`AttackExecution` accepte un `sourceTile`, et :

```ts
// base-game/src/core/execution/AttackExecution.ts
if (this.sourceTile !== null) this.addNeighbors(this.sourceTile);
else this.refreshToConquer();      // <- toutes les tuiles de frontière du joueur
```

Le pont passait `sourceTile = null` ⇒ l'attaque se déversait **sur toutes les
frontières**, ce qui rendait le choix de direction strictement décoratif (mesuré :
E/S/W identiques au bit près). Les 45 actions de l'agent étaient en réalité ~2 actions
différentes (« attaquer le joueur le plus proche » et « ne rien faire »).

✅ **Corrigé** : la tuile-frontière choisie est transmise comme `sourceTile` ⇒ l'attaque
reste confinée à ce front. Le pont choisit aussi le voisin terrestre avec le plus de
tuiles prenable dans la direction demandée.

### ③ Le pourcentage de territoire était mesuré sur le rectangle entier

`territory_pct = tuiles_possédées / (largeur × hauteur)`, océan compris. Or on ne
possède jamais une tuile d'eau :

| carte | tuiles terrestres | part de la grille |
|---|---|---|
| `australia_100x100_nt` | 10 000 | 100 % (aucun océan : carte « nt », ce n'est **pas** la géographie de l'Australie) |
| `australia_256x256` | 65 536 | 100 % |
| `australia_500x500_nt` | 212 707 | 85.1 % |
| `australia_1024x1024` | 723 165 | 69.0 % ⇒ **gagner à 80 % est mathématiquement impossible** |
| `australia_100x100` | **0** | carte **cassée** : 100 % d'eau, tous les joueurs spawennent en mer |
| `australia_500x500` | **0** | idem, carte **cassée** |

✅ **Corrigé** : `territory_pct` = part des **terres** (l'ancien calcul reste disponible
en `territory_pct_all`), seuil de victoire configurable (`win_threshold`, en fraction des
terres), et `land_tiles` exposé. Les deux cartes à 0 tuile terrestre sont signalées par
`skai/diagnose.py` — il faut les régénérer ou les retirer du README.

### ④ L'agent s'appauvrissait tout seul (troupes verrouillées en vol)

`attackDirection` engageait `troupes_de_la_grappe × intensité` **à chaque tick**, et le
moteur **fusionne** les attaques vers une même cible. Mesure initiale, attaque tous les
ticks à 0.5 : population **25 000 → 120 121**, territoire **0.5 %**, **rang 6/6**.
Aucun `cancel_attack` n'était exposé, donc la situation était irrécupérable : la seule
action « agressive » détruisait l'économie.

✅ **Corrigé** : engagement en fraction de la **réserve**, plafond de fronts simultanés
(`max_attacks`, défaut 4), refus du cumul sur un front déjà assez solide, et commande
`cancel_attacks` exposée côté Python (la retraite coûte 25 % des troupes : le moteur
facture le repli, cf. `malusForRetreat`).

⚠ **Piège rencontré pendant ces correctifs** : j'avais d'abord mappé l'action `WAIT` sur
`cancel_attacks` — ce qui rendait l'action « sûre » activement nocive. Résultat mesuré
sur 16 000 pas PPO : `ep_rew_mean` **−21 → −16**, territoire effondré à 1.4 %, rang 6/6.
`WAIT` est revenu à un vrai no-op. **Leçon : la métrique d'entraînement a détecté un bug
d'environnement en 4 minutes** — c'est exactement ce que `skai/diagnose.py` industrialise.

### ⑤ L'observation était illisible et le pont payait 5 passages complets par tick

- `neutral = (territory_map == 0)` **confondait océan et terre neutre**. Le moteur refuse
  de faire passer des troupes sur l'eau (`isWater` dans `AttackExecution.addNeighbors`) :
  sans canal « mer », l'agent ne peut pas deviner pourquoi il bute.
- `getState()` rescannait la carte entière pour `neutral_tiles`, **puis une fois par
  ennemi** pour `nearest_threat_distance`, **plus** un flood-fill complet pour les
  grappes, **plus** une liste de `canAttack()` dont chacun lance un **BFS sur 200 cases**.

✅ **Corrigé** : canal `water` (4 canaux au lieu de 3), `water_mask` 64×64 calculé une
seule fois à l'init ; un **unique** balayage par tick (`scanMap()`) produit carte + terres
neutres + centroïdes ; grappes calculées sur un bitmap `Uint8Array` mémoïsé par tick ;
`canAttack` limité aux 8 meilleurs candidats ; les index de tuiles des grappes ne sont
plus sérialisés (→ `tile_count`).

| Débit (ticks/s) | avant | après |
|---|---|---|
| 100×100 | 140 | **682** en boucle brute, 506–547 avec ordres d'attaque |
| 500×500 (`obs_size: 64`) | 6–12 | **158** |

`obs_size` est la nouveauté qui rend les grandes cartes exploitables : sans elle le pont
sérialisait 250 000 nombres par tick en JSON.

---

## 2. Résultat après correctifs

Mesuré par `python skai/diagnose.py --fast` (carte `australia_100x100_nt`, 5 adversaires,
200 ticks) :

```
TEST 2/5  config moteur   : test 6.6%  vs  default 25.0%   (× 3.8 de vitesse de conquête)
TEST 3/5  direction       : écart 3.61 points, centres étalés sur 19.6 tuiles en y  ✅
TEST 4/5  signal de récompense : passif 0.5% / glouton 25.0% (rang 1) / aléatoire 18.9%
                                 écart de récompense 49.0     ✅
TEST 5/5  fin de partie   : 25% des terres en 200 ticks, seuil à 80% ≈ 639 ticks extrapolés
```

Les épisodes **se terminent désormais** (`ep_len_mean` 1340–1460 au lieu de 1500 =
100 % de troncature) : la victoire et la mort existent dans le signal.

Court entraînement PPO (16 000 pas, 1 environnement) après correction :

| run | `ep_rew_mean` (début → fin) | territoire | rang moyen |
|---|---|---|---|
| config `test` (règles bridées, utiles pour débuter) | 62 → **89.8** | 51.3 % | **1.0** / 6 |
| config `default` (vraies règles) | 18.2 → −0.4 | 26.8 % | **1.0** / 6 |
| `--n-envs 2` | 18.6 → 15.5 | 14.2 % | 3.8 / 6 |
| **référence avant correctifs** | **plat à ≈ 54.6** (dérive) | 5.5 % | — |

**Lecture honnête.** Ces runs font ~12 épisodes : c'est une **preuve de vie** (le signal
existe, il est directionnel, l'agent atteint la 1ʳᵉ place sur 6), pas une preuve
d'apprentissage. Le seuil utile est plutôt 10⁶ pas **×** plusieurs environnements. Ne pas
se fier à `ep_rew_mean` seul (pénalité de vie + variance de longueur d'épisode) : d'où le
`ProgressCallback` de `train_simple.py` qui affiche **territoire et rang**.

---

## 3. Ce qui bloque encore (par ordre d'impact)

1. **Débit → passerelles multiples.** 16 000 pas = ~12 épisodes. `--n-envs` est branché
   (SubprocVecEnv, motif sûr Windows : fabrique de closure + `if __name__ == "__main__"`).
   Sur la machine de mesure (2 cœurs) le gain n'est que de +10 % : le débit suit le nombre
   de cœurs physiques, pas le nombre d'environnements demandés. Visez 8–16 cœurs
   (Colab/VM) pour 10⁶ pas en ~1 h.
2. **Spawns irréalistes.** Les joueurs sont posés sur un cercle de rayon `min(w,h)×0.4`
   puis « accrochés » à la terre la plus proche (correctif appliqué), mais ce n'est toujours
   pas le placement réel du jeu ; `gameMap: GameMapType.Asia` reste codé en dur dans
   `gameConfig` quelle que soit la carte chargée.
3. **Adversaires = bots `Difficulty.Easy`.** Le plafond de niveau de l'agent est donc le
   niveau *Easy*. La feuille de route (Étape 1, self-play) est la seule réponse ; avec
   l'espace d'action réparé, elle devient enfin exploitable.
4. **Cartes.** `australia_100x100` et `australia_500x500` sont vides (0 terre) : les
   régénérer ou les retirer du README. La carte d'entraînement par défaut (`_nt`) n'a
   **aucun océan** : pour un 1v1 crédible sur l'Australie, il faut entraîner sur
   `australia_500x500_nt` (85 % de terres), maintenant tenable grâce à `obs_size`.
5. **Action masking.** `SkaiEnv.action_masks()` est implémenté (directions légales =
   frontière voisine d'une terre prenable) et `--mask` bascule sur `MaskablePPO`
   (`sb3-contrib`). Le README promettait le masquage : il existait dans `agent/src`, pas
   dans le chemin `skai/`.
6. **Récompense.** `living_reward` par défaut à **−0.005/tick** (pression temporelle,
   cohérente avec l'objectif speedrun). À recalibrer avec le reste de l'étape 2.
7. **Protocole.** Le pont répond *asynchrone* au `reset` : si une commande `tick` arrive
   avant la fin du chargement de la carte, elle échoue (`Game not initialized`, reproduit).
   Inoffensif en aller-retour séquentiel, à corriger avant de paralléliser des commandes.

---

## 4. La VRAIE carte d'Australie (ajout 2026-09-01)

OpenFront amont publie `australia` en **2000×1500** avec un pyramidage déjà calculé
(`map4x` 1000×750, `map16x` 500×375). `skai/make_maps.py` les rapatrie et produit des
paquets au format `base-game` (`map.bin` + `mini_map.bin` + `manifest.json`).

Règle de réduction validée en rejouant le générateur officiel : majorité de tuiles
terrestres par bloc → **99.7 % d'accord** avec le `map4x.bin` publié par OpenFront.

```bash
python skai/make_maps.py --sizes 500,1000,2000   # les 3 paquets "real"
python skai/make_maps.py --repair                # régénère les 2 paquets à 0 terre
python skai/make_maps.py --list                  # état du portefeuille de cartes
```

| paquet | grille | terres | débit mesuré | usage |
|---|---|---|---|---|
| `australia_100x100` (réparé) | 100×100 | 6 004 (60 %) | **620 t/s** | ✅ **entraînement recommandé** : vraie géographie (côte + océan), objectif atteignable |
| `australia_500x500` (réparé) | 500×500 | 207 613 (83 %) | ~190 t/s | entraînement + évaluation |
| `australia_real_500x375` | 500×375 | 81 252 (43.3 %) | 182–195 t/s | **le continent entier**, évaluation/speedrun |
| `australia_real_1000x750` | 1000×750 | 328 153 | 59 t/s | évaluation haute fidélité |
| `australia_real_2000x1500` | 2000×1500 | 1 319 763 | — | client / display ; ignoré par git (régénérable) |

**Ce que la carte réelle change pour l'objectif (mesuré, politique gloutonne à 3 000 ticks) :**

- `australia_100x100` réparé : **41.5 % des terres, rang 1/5** — le seuil de victoire à 80 %
  devient un objectif *difficile mais honnête* (il ne l'était pas du tout avant).
- `australia_real_500x375` (continent entier) : **9.1 %, rang 6/6** ; à 20 joueurs et
  6 000 ticks la gloutonne se fait **éliminer**. Conclusion : sur le continent, « 80 % des
  terres en 1 500 ticks » n'est pas une tâche, c'est une punition. Deux choix s'offrent à toi :
  1. viser un objectif en **tuiles** ou en **% bas** (ex. 15 % en 3 000 ticks) plutôt que 80 % ;
  2. ou entraîner sur la 100×100 réparée (vraie géographie, 30× moins de tuiles) et ne
     monter sur le continent qu'en **évaluation**.
- Le détail qui compte : la part de terres d'une carte réelle est de ~44 %, donc
  `territory_pct` (dénominateur = terres) rend les pourcentages *comparables entre cartes*,
  ce que le décompte « % du rectangle » ne faisait pas.

Licence : les `.bin` de cartes sont des **données** amont, pas du code — elles restent
régies par `base-game/LICENSE-ASSETS` (le dépôt n'a pas de licence de données propre à
l'agent). `australia_real_2000x1500` et `australia_real_1000x750` sont **ignorés par git**
(grosses copies de données amont régénérables en une commande) ; seuls `australia_real_500x375`
et les deux recadrages réparés sont versionnés.

Côté moteur, deux réglages suivent la carte réelle : `spawn_mode: "land"` (échantillonnage
Poisson sur terres, graine variable à chaque épisode — sinon l'agent mémorise un placement)
et `disableNPCs` à `true` pour le mode « 0 nation » de la feuille de route (c'est exactement
ce que fait `GameRunner` : `disableNPCs ⇒ nations = []`).

---

## 5. Comment revérifier après chaque changement

```bash
python skai/make_maps.py --list         # les cartes sont-elles valides ?
python skai/diagnose.py --fast          # ~1 min : verdict GO/NO-GO
python skai/diagnose.py                 # ~4 min : complet
python skai/train_simple.py --steps 16000 --bots 5   # puis regarder territoire/rang
```

`diagnose.py` sort un code de retour non nul si l'environnement n'est pas apprenable :
il peut servir de test de régression dans la CI légère prévue à l'étape 5.
