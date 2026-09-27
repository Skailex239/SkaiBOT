# ⚔️ RatioLab — le bot qui teste tous les ratios d'attaque

RatioLab est un **bot de laboratoire** qui balaye **toutes les combinaisons
« ratio d'attaque × cadence d'envoi »** sur une **zone totalement verte**
(terra nullius, 1 seul joueur, aucun bot) et mesure la **vitesse de conquête
réelle** de chaque combinaison, avec les **vraies règles d'OpenFront**
(`DefaultConfig`, comme en partie réelle).

Il garde en mémoire les **meilleurs records** et les affiche dans une
**petite page web** : classements, matrice, courbes, historique.

> 📁 Emplacement : `skai/ratio_lab/` — zéro dépendance Python (stdlib uniquement),
> il pilote le moteur du jeu via le pont existant `agent/game_bridge/`.

---

## 🎯 Ce que le bot teste exactement

Une partie d'OpenFront laisse au joueur **deux leviers** au moment d'attaquer :

| Levier | Signification | Paramètre du bot |
|---|---|---|
| **Ratio d'attaque** | le **pourcentage de tes troupes** que tu engages dans l'attaque | `--ratios 10,20,…,100` (%) |
| **Cadence d'envoi** | **à quelle vitesse** tu renvoies des attaques (1 vague toutes les X ticks, 10 ticks = 1 s) | `--intervals 1,2,3,5,8,12,20` (ticks) |

Pour chaque combinaison `(ratio, cadence)`, le bot :

1. crée une partie **fraîche** sur la carte (1 joueur ⇒ toute la carte est verte) ;
2. attend la fin de l'**immunité de spawn** (~15 s de jeu, imposée par le moteur) ;
3. joue le **même script déterministe** pendant la durée configurée
   (défaut : **1 minute de jeu**) : tous les `interval` ticks, il envoie une
   **vague d'attaques dans les 8 directions**, chacune engageant `ratio` % de
   ses troupes disponibles (jusqu'à 8 fronts simultanés) ;
4. mesure la **vitesse de conquête** : tuiles gagnées par seconde de jeu.

### 📐 Pourquoi c'est intéressant (mécanique du moteur)

Sur de la terre neutre, OpenFront facture un coût **fixe** en troupes par tuile
(16/20/24 selon le terrain) mais la **vitesse de conquête dépend du nombre de
troupes engagées par attaque** (`DefaultConfig.attackLogic` : `tilesPerTickUsed`
= `33000 / troupes_engagées`, borné entre 5 et 100 par tuile). Autrement dit :

- un petit ratio ⇒ attaques lentes (peu de tuiles par tick) ;
- un gros ratio ⇒ attaques rapides, mais beaucoup de troupes bloquées dans un
  front et moins de fronts en parallèle ;
- envoyer trop souvent ou trop rarement change aussi le résultat.

Il n'y a pas de réponse « théorique » : **c'est exactement ce que mesure le bot.**

### 📊 Métriques enregistrées par test

| Champ | Signification |
|---|---|
| `tiles_per_sec` | ⭐ **vitesse de conquête** (tuiles gagnées / seconde de jeu) |
| `tiles_per_tick` | idème par tick (10 ticks = 1 s) |
| `tiles_gained`, `land_pct` | tuiles gagnées, part de la carte à la fin |
| `t10/t25/t50/t75_ticks` | ticks nécessaires pour atteindre 10 / 25 / 50 / 75 % de la carte |
| `troops_start/end/min` | troupes disponibles au début / à la fin / au point le plus bas |
| `waves_attempted/ok` | vagues d'attaques envoyées / acceptées par le moteur |

---

## 🚀 Comment lancer

### 0. Prérequis (une seule fois)

```bash
cd base-game
npm install --omit=dev --ignore-scripts   # dépendances du moteur (zod, etc.)
```

> `npm install` complet marche aussi ; sur certaines machines le paquet `canvas`
> (tests Jest) échoue à compiler — il n'est pas nécessaire ici.

### 1. Lancer le bot (balayage complet)

```bash
# à la racine de SkaiBOT/ — ≈ 70 combinaisons, 1 min de jeu chacune
python skai/ratio_lab/sweep.py

# test rapide (6 combinaisons, 30 s de jeu chacune)
python skai/ratio_lab/sweep.py --ratios 10,50,100 --intervals 1,5,20 --duration 30

# plus vite : 4 tests en parallèle, 3 répétitions par combinaison
python skai/ratio_lab/sweep.py --jobs 4 --repeats 3
```

À la fin, le bot affiche la **meilleure combinaison** du balayage et enregistre
tout dans `skai/ratio_lab/results/` :

| Fichier | Contenu |
|---|---|
| `runs.jsonl` | chaque test, une ligne JSON (l'historique complet) |
| `runs.csv` | idem, format tableur |
| `records.json` | les records dérivés (classements, matrices…) |

### 2. Voir les records (la petite UI web)

```bash
python skai/ratio_lab/server.py        # puis ouvrir http://localhost:8080
```

La page affiche :

- 🏆 le **record absolu** (combo le plus rapide) et le **top 10** ;
- 🏅 l'**historique des records** (à quel moment un combo a battu le précédent) ;
- 📊 la **matrice ratio × cadence** colorée (clique sur une case = détail) ;
- 📈 les **courbes** vitesse selon le ratio, une ligne par cadence ;
- 📋 l'historique complet des tests ;
- 🧪 un **formulaire pour lancer un nouveau balayage directement depuis la page**
  (ratios, cadences, durée du test, carte, répétitions…).

---

## ⚙️ Référence des options

```
python skai/ratio_lab/sweep.py --help
```

| Option | Défaut | Description |
|---|---|---|
| `--ratios` | `10,20,…,100` | pourcents de troupes engagés par attaque |
| `--intervals` | `1,2,3,5,8,12,20` | ticks entre deux vagues (10 ticks = 1 s de jeu) |
| `--duration` | `60` | durée de **chaque test**, en **secondes de jeu** |
| `--map` | `australia_100x100_nt` | carte (100 % de terres = zone totalement verte) |
| `--repeats` | `1` | répétitions par combinaison (graines de départ décalées) |
| `--jobs` | `1` | tests en parallèle (1 processus Node par job) |
| `--max-attacks` | `8` | fronts simultanés maximum |
| `--players` | `1` | joueurs dans la partie (1 = zone totalement verte) |
| `--seed` | `12345` | graine de placement (comparaisons équitables entre combos) |
| `--out` | `skai/ratio_lab/results` | dossier de résultats |

> ⏱️ **Le temps du test est configurable** (`--duration`), en secondes **de jeu**.
> La simulation tourne plus vite que le temps réel : une minute de jeu prend
> quelques secondes de calcul. L'immunité de spawn (~15 s, règle du jeu) n'est
> **pas** comptée dans la durée — le chronomètre démarre à la première attaque.

---

## 🧪 Contrôles de qualité des comparaisons

- **Même graine de départ** pour toutes les combinaisons (`--seed`) : tout le
  monde part de la même position sur la même carte ;
- **Même script d'attaque** (vague dans les 8 directions, fronts limités à `--max-attacks`) ;
- **Vraies règles** `DefaultConfig` (et non la config de test bridée) ;
- Chaque run est rejouée sur une **partie neuve** (aucun état résiduel) ;
- `--repeats 3` (ou plus) lisse les variations liées au terrain.

## 🗺️ Tester sur d'autres cartes

```bash
python skai/ratio_lab/sweep.py --map australia_real_500x375 --duration 120
```

Les records sont **séparés par (carte, durée)** : comparer une minute de test sur
la 100×100 avec deux minutes sur la carte réelle n'aurait pas de sens, l'UI les
présente donc dans des groupes distincts.

---

## 🧩 Comment ça marche sous le capot

```
skai/ratio_lab/sweep.py ──► skai/viewer/bridge.py ──► agent/game_bridge/game_bridge.ts ──► base-game (moteur OpenFront)
        │                                                                                   règles DefaultConfig
        └──► results/runs.jsonl + records.json ◄── server.py ◄── webui/index.html (l'UI)
```

- le balayage est **parallélisable** (`--jobs`) : chaque worker possède son
  propre processus Node/pont et le referme proprement ;
- l'UI peut lancer un balayage **à chaud** (`POST /api/sweep`) pendant qu'elle
  affiche les records déjà obtenus ;
- tout est écrit au fil de l'eau : on peut rafraîchir la page à tout moment.

## ⚠️ Note honnête

Ce bot est un **instrument de mesure** pour comprendre les mécaniques du jeu
(localement, hors ligne, sur ta propre copie du moteur). Ce n'est pas un
automate de jeu en ligne : il ne se connecte à aucun serveur public d'OpenFront.
