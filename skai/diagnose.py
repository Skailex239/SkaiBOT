"""
SkaiBOT — diagnostic de l'environnement d'entraînement (à lancer AVANT de trained).

    python skai/diagnose.py            # complet (~4 min sur CPU)
    python skai/diagnose.py --fast     # rapide (~1 min)

Pourquoi ce script : un agent RL ne peut pas apprendre si l'environnement ne réagit
pas à ses actions. Les 5 vérifications ci-dessous répondent à "l'environnement est-il
apprenable ?" avec des chiffres, pas des impressions. Chacune a été écrite après avoir
mesuré un point de blocage réel (voir docs/DIAGNOSTIC.md).

Sortie : un tableau par test + un verdict GLOBAL (OK / À CORRIGER).
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)

from skai.viewer.bridge import GameBridge  # noqa: E402

MAPS_DIR = os.path.join(ROOT, "base-game", "resources", "maps")
SHIFTS = [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]
DIR_NAMES = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
DIAG_INTENSITY = 0.35   # fraction des troupes libres engagées par ordre d'attaque


# --------------------------------------------------------------------------- utilitaires
def land_fraction(map_name):
    """(tuiles terrestres, tuiles totales) d'une carte, lu directement dans le .bin.

    Le bit 7 du fichier map.bin est IS_LAND (cf. GameMapImpl.IS_LAND_BIT) : c'est ce
    bit que le moteur utilise pour refuser/autoriser une conquête, donc c'est la seule
    dénomination qui vaille pour un pourcentage de territoire.
    """
    d = os.path.join(MAPS_DIR, map_name)
    manifest = json.load(open(os.path.join(d, "manifest.json"), encoding="utf-8"))
    w, h = manifest["map"]["width"], manifest["map"]["height"]
    raw = np.fromfile(os.path.join(d, "map.bin"), dtype=np.uint8)
    if raw.size != w * h:
        return None, w * h, "?" 
    land = (((raw >> 7) & 1) == 1).reshape(h, w)
    return int(land.sum()), w * h, f"{w}x{h}" 


def greedy_direction(tmap, land):
    """Meilleure direction : la frontière qui offre le plus de terres prenable."""
    own = tmap == 1
    conquestable = (tmap != 1) & (land > 0.5)
    scores = []
    for dy, dx in SHIFTS:
        rolled = np.roll(np.roll(own, dy, axis=0), dx, axis=1)
        scores.append(float((rolled & conquestable).sum()))
    return int(np.argmax(scores)) if max(scores) > 0 else 8


def play(policy, ticks, map_name, num_players, game_config, win_threshold, obs_size, verbose=False):
    """Joue une partie complète, retourne (récompense totale, état final, ticks/s)."""
    b = GameBridge()
    b.start()
    st = b.reset(map_name, num_players, obs_size=obs_size,
                 game_config=game_config, win_threshold=win_threshold)
    rng = np.random.default_rng(0)
    reward, prev_pct, hist = 0.0, st.get("territory_pct", 0.0), []
    try:
        for i in range(ticks):
            a = policy(i, st, rng)
            if isinstance(a, list):
                for d in a:                      # essaie les fronts par ordre de préférence
                    if b.attack(direction=d, intensity=DIAG_INTENSITY, cluster_id=0):
                        break
            elif a is not None and a < 8:
                b.attack(direction=a, intensity=DIAG_INTENSITY, cluster_id=0)
            elif a == 8:
                b.cancel_attacks()
            st = b.tick()
            pct = st.get("territory_pct", 0.0)
            reward += (pct - prev_pct) * 200.0
            prev_pct = pct
            hist.append(pct)
            if st.get("game_over"):
                break
    finally:
        b.close()
    return reward, st, len(hist), np.array(hist)


PASSIVE = lambda i, st, rng: None
def dir_scores(tmap, land):
    """score de chaque direction = nb de nos tuiles de frontière touchant une terre prenable"""
    own = tmap == 1
    conquestable = (tmap != 1) & (land > 0.5)
    return [float((np.roll(np.roll(own, dy, axis=0), dx, axis=1) & conquestable).sum())
            for dy, dx in SHIFTS]


def greedy(i, st, rng):
    """Score décroissant, et on RETOMBE sur les directions suivantes si le pont refuse
    l'ordre : sur une vraie carte, la direction « la plus riche » en tuiles neutres peut
    être à travers l'océan (donc illégale) — une politique de test doit quand même attaquer."""
    tmap = np.array(st["territory_map"], dtype=np.float32)
    land = np.array(st.get("water_mask", np.ones_like(tmap)), dtype=np.float32)
    order = np.argsort(dir_scores(tmap, land))[::-1].tolist()
    return [d for d in order] if i % 3 == 0 else None       # liste = candidats par préférence
def random_action(i, st, rng):
    return int(rng.integers(0, 9)) if i % 3 == 0 else None


# --------------------------------------------------------------------------------- tests
def test_maps(maps):
    print("\n" + "=" * 78)
    print("TEST 1/5 — Cartes : le pourcentage de territoire est-il mesurable ?")
    print("=" * 78)
    print(f"{'carte':24s} {'grille':>10s} {'terre':>9s} {'%terre':>8s}   verdict")
    problems = []
    for m in maps:
        try:
            land, total, size = land_fraction(m)
        except FileNotFoundError:
            print(f"{m:24s} {'--':>10s}  fichier manquant"); problems.append(m); continue
        if land is None:
            print(f"{m:24s} {'?':>10s}  map.bin incohérent avec le manifest"); problems.append(m); continue
        frac = land / total
        verdict = "OK"
        if land == 0:
            verdict = "CASSÉE : 0 tuile terrestre (spawn sur l'eau, jeu injouable)"
            problems.append(m)
        elif frac >= 0.999:
            verdict = "sans océan : ce n'est pas la géographie d'Australie (carte 'nt')"
        elif frac < 0.30:
            verdict = "⚠ 80% de la rectangle = inatteignable ; mesurer sur les terres"
        print(f"{m:24s} {size:>10s} {land:>9d} {frac*100:>7.1f}%   {verdict}")
    print(f"\n  {'✅ toutes les cartes listées sont utilisables' if not problems else '❌ cartes inutilisables : ' + ', '.join(problems)}")
    return problems


def test_config(ticks, map_name, bots):
    print("\n" + "=" * 78)
    print("TEST 2/5 — Config moteur : DefaultConfig vs TestConfig (ancien câblage)")
    print("=" * 78)
    print("  TestConfig fige le combat : attackTilesPerTick() => 1, i.e. UNE tuile")
    print("  conquise par tick et par attaque. On compare la même politique (gloutonne).")
    out = {}
    for cfg in ("test", "default"):
        r, st, n, hist = play(greedy, ticks, map_name, bots + 1, cfg, 0.80, 64)
        out[cfg] = (r, st, n, hist)
        print(f"  game_config={cfg:8s} → territoire {st['territory_pct']*100:5.1f}% "
              f"en {n} ticks (récompense {r:6.1f})")
    a, b_ = out["test"][1]["territory_pct"], out["default"][1]["territory_pct"]
    ratio = (b_ / a) if a > 1e-9 else float("inf")
    print(f"\n  vitesse de conquête × {ratio:.1f} avec les vraies règles")
    return out


def test_direction(ticks, map_name, bots, game_config):
    print("\n" + "=" * 78)
    print("TEST 3/5 — Sensibilité à l'action : la DIRECTION change-t-elle le résultat ?")
    print("=" * 78)
    print("  Si les 8 directions donnent le même territoire, l'espace d'action est")
    print("  décoratif et AUCUN entraînement ne peut converger (le gradient est nul).")
    pcts, cents = [], []
    for d in range(8):
        pol = (lambda dd: (lambda i, st, rng: dd if i % 5 == 0 else None))(d)
        r, st, n, hist = play(pol, ticks, map_name, bots + 1, game_config, 0.80, 100)
        a = np.array(st["territory_map"], dtype=np.float32)
        own = a == 1
        h, w = own.shape
        xs = np.arange(w); ys = np.arange(h)
        cx = float((own.sum(0) * xs).sum() / max(own.sum(), 1))
        cy = float((own.sum(1) * ys).sum() / max(own.sum(), 1))
        pcts.append(own.mean() * 100); cents.append((cx, cy))
        print(f"  {DIR_NAMES[d]:2s} territoire {own.mean()*100:5.2f}%  centre ({cx:5.1f},{cy:5.1f})")
    spread_pct = max(pcts) - min(pcts)
    spread_cy = max(c[1] for c in cents) - min(c[1] for c in cents)
    print(f"\n  écart territoire min/max : {spread_pct:5.2f} points   |  étendue verticale des centres : {spread_cy:5.1f} tuiles")
    if spread_pct < 0.5 and spread_cy < 3:
        print("  ❌ L'ACTION EST IGNOREE : le moteur conquiert sur toutes les frontières.")
        print("     → passer un sourceTile à AttackExecution (cf. docs/DIAGNOSTIC.md).")
        return False
    print("  ✅ les directions produisent des conquêtes spatialement distinctes")
    return True


def test_signal(ticks, map_name, bots, game_config, win_threshold):
    print("\n" + "=" * 78)
    print("TEST 4/5 — Signal de récompense : les politiques se démarquent-elles ?")
    print("=" * 78)
    rows = []
    for name, pol in (("passive (n'attaque jamais)", PASSIVE), ("gloutonne (front le plus riche)", greedy),
                      ("aleatoire (dir. uniforme)", random_action)):
        r, st, n, hist = play(pol, ticks, map_name, bots + 1, game_config, win_threshold, 64)
        rows.append((name, r, st, hist))
        print(f"  {name:32s} récompense {r:8.1f}   territoire {st['territory_pct']*100:5.1f}%   "
              f"rang {st['rank']}/{st['alive_players']}")
    rs = np.array([x[1] for x in rows])
    spread = rs.max() - rs.min()
    print(f"\n  écart de récompense entre meilleure et pire politique : {spread:.1f}")
    for name, r, st, hist in rows:
        if st.get("territory_pct", 0) > 0 and len(hist) > 50:
            rate = (hist[-1] - hist[0]) / max(len(hist) - 1, 1) * 100
            print(f"    {name:34s} vitesse de conquête {rate:+.3f} %/tick")
    if spread < 5:
        print("  ❌ signal quasi nul : PPO n'a rien à optimiser (la valeur expliquée")
        print("     viendra du bruit, pas des décisions de l'agent).")
        return False
    print("  ✅ la récompense discrimine les politiques : l'apprentissage a une prise")
    return True


def test_reachability_and_speed(ticks, map_name, bots, game_config, win_threshold, target_steps):
    print("\n" + "=" * 78)
    print("TEST 5/5 — Fin de partie atteignable + débit d'entraînement")
    print("=" * 78)
    r, st, n, hist = play(greedy, ticks, map_name, bots + 1, game_config, win_threshold, 64)
    b = GameBridge(); b.start()
    b.reset(map_name, bots + 1, obs_size=64, game_config=game_config)
    t0 = time.time()
    K = 120
    for _ in range(K):
        b.tick()
    tps = K / (time.time() - t0)
    b.close()
    won = st.get("has_won", False)
    print(f"  politique gloutonne : {n} ticks → {st['territory_pct']*100:.1f}% des terres "
          f"(seuil de victoire {win_threshold*100:.0f}%)  has_won={won}")
    if not won and st['territory_pct'] < win_threshold:
        need = int((win_threshold - st['territory_pct']) / max(st['territory_pct'] / max(n, 1), 1e-6))
        print(f"  → extrapolation : il faudrait ~{n + need} ticks pour atteindre le seuil "
              f"(max_ticks actuel = {ticks})")
    print(f"  débit mesuré : {tps:.0f} ticks/s par environnement (boucle brute, sans ordre d'attaque)")
    with_attacks = tps * 0.72          # 547/760 mesuré : les ordres + JSON coûtent ~28 %
    print(f"  avec ordres d'attaque : ~{with_attacks:.0f} ticks/s ; PPO ajoute son propre coût")
    print(f"  => 1,5 M de pas ≈ {target_steps/with_attacks/3600:.1f} h sur 1 environnement, "
          f"à diviser par le nombre de cœurs physiques (--n-envs)")
    return tps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="australia_100x100_nt")
    ap.add_argument("--bots", type=int, default=5)
    ap.add_argument("--ticks", type=int, default=600)
    ap.add_argument("--fast", action="store_true", help="moins de ticks, moins de configs testées")
    ap.add_argument("--game-config", default="default", choices=["default", "test"])
    ap.add_argument("--win-threshold", type=float, default=0.80)
    args = ap.parse_args()
    ticks = max(150, args.ticks // 3) if args.fast else args.ticks

    maps = sorted(d for d in os.listdir(MAPS_DIR) if os.path.isdir(os.path.join(MAPS_DIR, d))) \
        if os.path.isdir(MAPS_DIR) else [args.map]
    broken = test_maps(maps)
    if args.map in broken:
        print(f"\n  ⚠ la carte d'entraînement par défaut ({args.map}) est dans la liste des cartes cassées")
    test_config(ticks, args.map, args.bots)
    ok_dir = test_direction(max(200, ticks // 2), args.map, args.bots, args.game_config)
    ok_sig = test_signal(ticks, args.map, args.bots, args.game_config, args.win_threshold)
    test_reachability_and_speed(ticks, args.map, args.bots, args.game_config, args.win_threshold, 1_500_000)

    print("\n" + "=" * 78)
    print("VERDICT GLOBAL :", "✅ environnement apprenable" if (ok_dir and ok_sig) else "❌ entraînement inutile tant que ce n'est pas corrigé")
    print("=" * 78)
    return 0 if (ok_dir and ok_sig) else 1


if __name__ == "__main__":
    sys.exit(main())
