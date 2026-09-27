"""RatioLab — le bot qui teste toutes les combinaisons « ratio d'attaque × cadence ».

Principe
--------
On lance une partie sur une ZONE TOTALEMENT VERTE (terra nullius, 1 seul joueur,
aucun bot) et on fait jouer un bot déterministe :

    tous les `interval` ticks, il envoie une VAGUE d'attaques dans les 8 directions,
    chaque attaque engage `ratio` % de ses troupes disponibles.

Chaque combinaison (ratio, cadence) est rejouée sur une partie fraîche, et le bot
mesure la VITESSE DE CONQUÊTE réelle (tuiles gagnées par seconde de jeu). Les
résultats s'accumulent dans `results/` et les records sont affichés par la petite
UI web (`server.py`).

Rappel de mécanique OpenFront (1 tick = 100 ms → 10 ticks = 1 s de jeu) :
- conquête de terre neutre : coût fixe de 16/20/24 troupes par tuile (plaines/
  hauts-plateaux/montagnes), mais la VITESSE dépend du nombre de troupes engagées
  par attaque — d'où l'intérêt de balayer les ratios ;
- immunité de spawn : ~15 s (100 ticks de phase de spawn + 50 ticks d'immunité)
  pendant lesquelles le moteur refuse les attaques.

Exemples
--------
# Le grand balayage par défaut (≈ 70 combinaisons, 1 min de jeu chacune) :
python skai/ratio_lab/sweep.py

# Test rapide :
python skai/ratio_lab/sweep.py --ratios 10,50,100 --intervals 1,5,20 --duration 30

# Paralléliser sur 4 cœurs, 3 répétitions par combinaison :
python skai/ratio_lab/sweep.py --jobs 4 --repeats 3
"""
from __future__ import annotations

import argparse
import os
import queue as queue_mod
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

# Permet de lancer le script directement (python skai/ratio_lab/sweep.py)
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

from skai.viewer.bridge import GameBridge  # noqa: E402
from skai.ratio_lab import records as rec  # noqa: E402

# --- Constantes de jeu (OpenFront, base-game/src/core/configuration/DefaultConfig.ts)
TICKS_PER_SECOND = 10          # 1 tick = 100 ms
DIRECTIONS = list(range(8))    # N, NE, E, SE, S, SW, W, NW

DEFAULT_MAP = "australia_100x100_nt"   # 100 % de terres => zone totalement verte
DEFAULT_DURATION_S = 60                # « une minute » par test, configurable
DEFAULT_RATIOS = "10,20,30,40,50,60,70,80,90,100"
DEFAULT_INTERVALS = "1,2,3,5,8,12,20"
DEFAULT_SEED = 12345
DEFAULT_MAX_ATTACKS = 8

_file_lock = threading.Lock()
_thread_local = threading.local()


# --------------------------------------------------------------------------- utilitaires

def parse_int_list(text: str, name: str) -> List[int]:
    out: List[int] = []
    for part in str(text).replace(" ", "").split(","):
        if not part:
            continue
        try:
            out.append(int(part))
        except ValueError:
            raise SystemExit(f"valeur invalide pour {name}: {part!r} (attendu: 1,2,3)")
    if not out:
        raise SystemExit(f"{name} ne peut pas être vide")
    return out


def _worker_bridge() -> GameBridge:
    """Un pont Node par thread de travail, réutilisé d'une run à l'autre."""
    b = getattr(_thread_local, "bridge", None)
    if b is None or b.proc is None or b.proc.poll() is not None:
        b = GameBridge()
        b.start()
        _thread_local.bridge = b
    return b


def close_worker_bridges() -> None:
    b = getattr(_thread_local, "bridge", None)
    if b is not None:
        try:
            b.close()
        except Exception:  # noqa: BLE001
            pass
        _thread_local.bridge = None


# --------------------------------------------------------------------------- une run

def run_single(
    ratio_pct: int,
    interval: int,
    cfg: Dict[str, Any],
    repeat_idx: int = 0,
) -> Dict[str, Any]:
    """Une partie = un test d'une combinaison (ratio %, cadence en ticks).

    Mesure la conquête pendant `cfg['duration_s']` SECONDES DE JEU à partir de la
    première attaque acceptée par le moteur (l'immunité de spawn n'est pas comptée).
    """
    bridge = _worker_bridge()
    intensity = max(0.01, min(1.0, ratio_pct / 100.0))
    seed = int(cfg.get("seed", DEFAULT_SEED)) + repeat_idx * 7919
    duration_s = float(cfg["duration_s"])
    duration_ticks = max(1, round(duration_s * TICKS_PER_SECOND))
    map_name = cfg["map"]
    max_attacks = int(cfg.get("max_attacks", DEFAULT_MAX_ATTACKS))
    players = int(cfg.get("players", 1))

    t_wall = time.time()
    base: Dict[str, Any] = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "map": map_name,
        "duration_s": duration_s,
        "ratio_pct": int(ratio_pct),
        "interval": int(interval),
        "seed": seed,
        "repeat_idx": int(repeat_idx),
        "max_attacks": max_attacks,
    }

    try:
        state = bridge.reset(
            map_name,
            players,
            obs_size=1,                 # inutile d'envoyer la carte : on mesure des compteurs
            game_config="default",      # vraies règles d'OpenFront
            max_attacks=max_attacks,
            spawn_mode="land",
            spawn_seed=seed,
            allow_multi_front=True,     # explorer plusieurs fronts fait partie du test
        )
    except Exception as e:  # noqa: BLE001
        base.update({"status": "error", "error": f"reset: {e}", "wall_time_s": round(time.time() - t_wall, 2)})
        return base

    land_tiles = max(1, int(state.get("land_tiles") or state.get("total_tiles") or 1))
    intensity_ok = False
    t0: Optional[int] = None
    tiles_start = int(state["tiles_owned"])

    # --- Phase 1 : attendre la fin de l'immunité de spawn (~15 s de jeu).
    # Une seule direction testée par tick (rotatif) pour rester rapide ; dès qu'une
    # attaque passe, la vague 1 est complétée dans les autres directions.
    first_dir = -1
    for probe in range(400):
        d = probe % 8
        if bridge.attack(d, intensity):
            t0 = int(state["tick"])
            first_dir = d
            tiles_start = int(state["tiles_owned"])
            intensity_ok = True
            break
        state = bridge.tick()
        if state.get("game_over"):
            break

    if not intensity_ok or t0 is None:
        base.update({
            "status": "no_attack",
            "error": "aucune attaque acceptée en 400 ticks (immunité ? carte sans cible ?)",
            "wall_time_s": round(time.time() - t_wall, 2),
        })
        return base

    # --- Vague 1 : compléter les directions restantes (la probe en a déjà lancé une)
    waves_attempted = 1
    waves_ok = 1
    for d in DIRECTIONS:
        if d == first_dir:
            continue
        bridge.attack(d, intensity)   # vague 1 déjà comptée ci-dessus

    # --- Phase 2 : le cœur du test, `duration_ticks` ticks de conquête
    milestones: Dict[str, Optional[int]] = {"t10": None, "t25": None, "t50": None, "t75": None}
    troops_start = int(state.get("population", 0))
    troops_min = troops_start
    active_ticks = 0
    status = "ok"

    while True:
        state = bridge.tick()
        tick = int(state["tick"])
        active_ticks = tick - t0
        tiles = int(state["tiles_owned"])
        pop = int(state.get("population", 0))
        troops_min = min(troops_min, pop)

        pct = tiles / land_tiles
        for key, threshold in (("t10", 0.10), ("t25", 0.25), ("t50", 0.50), ("t75", 0.75)):
            if milestones[key] is None and pct >= threshold:
                milestones[key] = active_ticks

        if state.get("game_over"):
            status = "won" if state.get("has_won") else "game_over"
            break
        if active_ticks >= duration_ticks:
            break

        # --- Vague suivante tous les `interval` ticks ("la vitesse d'envoi")
        if active_ticks % interval == 0:
            waves_attempted += 1
            ok = 0
            for d in DIRECTIONS:
                if bridge.attack(d, intensity):
                    ok += 1
            if ok:
                waves_ok += 1

    tiles_end = int(state["tiles_owned"])
    tiles_gained = max(0, tiles_end - tiles_start)
    secs = active_ticks / TICKS_PER_SECOND
    base.update({
        "status": status,
        "tiles_start": tiles_start,
        "tiles_end": tiles_end,
        "tiles_gained": tiles_gained,
        "land_tiles": land_tiles,
        "land_pct": round(tiles_end / land_tiles, 4),
        "active_ticks": active_ticks,
        "tiles_per_sec": round(tiles_gained / secs, 3) if secs > 0 else 0.0,
        "tiles_per_tick": round(tiles_gained / active_ticks, 4) if active_ticks > 0 else 0.0,
        "troops_start": troops_start,
        "troops_end": int(state.get("population", 0)),
        "troops_min": troops_min,
        "waves_attempted": waves_attempted,
        "waves_ok": waves_ok,
        "t10_ticks": milestones["t10"],
        "t25_ticks": milestones["t25"],
        "t50_ticks": milestones["t50"],
        "t75_ticks": milestones["t75"],
        "wall_time_s": round(time.time() - t_wall, 2),
    })
    return base


# --------------------------------------------------------------------------- le balayage

def build_task_list(cfg: Dict[str, Any]) -> List[Tuple[int, int, int]]:
    tasks: List[Tuple[int, int, int]] = []
    for repeat_idx in range(int(cfg.get("repeats", 1))):
        for ratio in cfg["ratios"]:
            for interval in cfg["intervals"]:
                tasks.append((int(ratio), int(interval), repeat_idx))
    return tasks


def run_sweep(
    cfg: Dict[str, Any],
    status_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> List[Dict[str, Any]]:
    """Balaye toutes les combinaisons et enregistre chaque run au fur et à mesure.

    `status_cb` reçoit des événements : {"event": "run_start"|"run_done"|"sweep_done", ...}.
    """
    def emit(ev: Dict[str, Any]) -> None:
        if status_cb is not None:
            try:
                status_cb(ev)
            except Exception:  # noqa: BLE001
                pass

    results_dir = rec.results_dir(cfg.get("results_dir"))
    tasks = build_task_list(cfg)
    jobs = max(1, int(cfg.get("jobs", 1)))
    emit({"event": "sweep_start", "total": len(tasks), "cfg": _public_cfg(cfg)})

    runs: List[Dict[str, Any]] = []
    done = 0

    def _one(task: Tuple[int, int, int]) -> Dict[str, Any]:
        ratio, interval, repeat_idx = task
        emit({"event": "run_start", "ratio_pct": ratio, "interval": interval, "repeat_idx": repeat_idx})
        try:
            return run_single(ratio, interval, cfg, repeat_idx)
        except Exception as e:  # noqa: BLE001
            return {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "map": cfg["map"], "duration_s": cfg["duration_s"],
                "ratio_pct": ratio, "interval": interval, "repeat_idx": repeat_idx,
                "status": "error", "error": str(e),
                "wall_time_s": 0.0,
            }

    def _record(run: Dict[str, Any]) -> None:
        nonlocal done
        runs.append(run)
        done += 1
        with _file_lock:
            rec.append_run(results_dir, run)
        emit({"event": "run_done", "run": run, "done": done, "total": len(tasks)})

    if jobs == 1:
        try:
            for task in tasks:
                _record(_one(task))
        finally:
            close_worker_bridges()
    else:
        # Chaque thread de travail possède SON pont Node : il le ferme à la fin de
        # son file de tâches (et le rouvre au besoin), aucun processus ne fuit.
        queue: "queue_mod.Queue[Tuple[int, int, int]]" = queue_mod.Queue()
        for t in tasks:
            queue.put(t)

        def _worker_loop() -> None:
            try:
                while True:
                    try:
                        task = queue.get_nowait()
                    except queue_mod.Empty:
                        break
                    _record(_one(task))
            finally:
                close_worker_bridges()

        threads = [threading.Thread(target=_worker_loop, daemon=True) for _ in range(jobs)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

    rec.save_records(results_dir, None)
    emit({"event": "sweep_done", "done": done, "total": len(tasks)})
    return runs


def _public_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "map": cfg["map"],
        "duration_s": cfg["duration_s"],
        "ratios": cfg["ratios"],
        "intervals": cfg["intervals"],
        "repeats": cfg.get("repeats", 1),
        "jobs": cfg.get("jobs", 1),
        "max_attacks": cfg.get("max_attacks", DEFAULT_MAX_ATTACKS),
        "seed": cfg.get("seed", DEFAULT_SEED),
        "players": cfg.get("players", 1),
    }


# --------------------------------------------------------------------------- CLI

def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(
        description="RatioLab — teste toutes les combinaisons ratio d'attaque × cadence sur zone verte.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--ratios", default=DEFAULT_RATIOS, help="pourcents de troupes envoyés par attaque")
    ap.add_argument("--intervals", default=DEFAULT_INTERVALS, help="ticks entre deux vagues d'attaque (10 ticks = 1 s)")
    ap.add_argument("--duration", type=float, default=DEFAULT_DURATION_S, help="durée de chaque test en SECONDES de jeu")
    ap.add_argument("--map", default=DEFAULT_MAP, help="carte du dépôt (zone verte = 1 joueur)")
    ap.add_argument("--repeats", type=int, default=1, help="répétitions par combinaison (graines décalées)")
    ap.add_argument("--jobs", type=int, default=1, help="tests en parallèle (1 processus Node par job)")
    ap.add_argument("--max-attacks", type=int, default=DEFAULT_MAX_ATTACKS, help="fronts simultanés maximum")
    ap.add_argument("--players", type=int, default=1, help="joueurs dans la partie (1 = zone totalement verte)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED, help="graine de placement de départ")
    ap.add_argument("--out", default=None, help="dossier de résultats (défaut: skai/ratio_lab/results)")
    ap.add_argument("--quiet", action="store_true", help="n'affiche que le résumé final")
    args = ap.parse_args(argv)

    cfg = {
        "map": args.map,
        "duration_s": args.duration,
        "ratios": parse_int_list(args.ratios, "--ratios"),
        "intervals": parse_int_list(args.intervals, "--intervals"),
        "repeats": max(1, args.repeats),
        "jobs": max(1, args.jobs),
        "max_attacks": args.max_attacks,
        "seed": args.seed,
        "players": max(1, args.players),
        "results_dir": args.out,
    }

    total = len(build_task_list(cfg))
    print(f"🧪 RatioLab — {total} tests × {cfg['duration_s']:g} s de jeu "
          f"(carte {cfg['map']}, {cfg['players']} joueur(s))")
    print(f"   ratios : {cfg['ratios']}")
    print(f"   cadences (ticks entre 2 vagues) : {cfg['intervals']}")
    print()

    t0 = time.time()

    def cli_status(ev: Dict[str, Any]) -> None:
        if args.quiet:
            return
        if ev["event"] == "run_done":
            r = ev["run"]
            if r.get("status") == "ok":
                print(f"  [{ev['done']}/{ev['total']}] ratio {r['ratio_pct']:>3} % | "
                      f"cadence {r['interval']:>2} ticks | "
                      f"→ {r['tiles_per_sec']:>7.2f} tuiles/s "
                      f"({r['tiles_gained']} tuiles en {r['active_ticks'] / 10:.0f} s, "
                      f"{r.get('land_pct', 0) * 100:.1f} % de la carte)")
            else:
                print(f"  [{ev['done']}/{ev['total']}] ratio {r['ratio_pct']} % | "
                      f"cadence {r['interval']} | ✗ {r.get('status')}: {r.get('error', '')[:80]}")
        elif ev["event"] == "sweep_done":
            print(f"\n✅ Balayage terminé en {time.time() - t0:.0f} s.")

    runs = run_sweep(cfg, status_cb=cli_status)

    ok = [r for r in runs if r.get("status") == "ok"]
    print()
    if ok:
        best = max(ok, key=lambda r: r["tiles_per_sec"])
        print("🏆 Meilleure combinaison de ce balayage :")
        print(f"   ratio {best['ratio_pct']} % | cadence 1 vague / {best['interval']} tick(s)")
        print(f"   → {best['tiles_per_sec']} tuiles/s "
              f"({best['tiles_gained']} tuiles en {best['duration_s']:g} s de jeu, "
              f"{best['land_pct'] * 100:.1f} % de la carte)")
    else:
        print("⚠️  Aucune run valide — vérifie la carte et les paramètres.")
    print(f"\n📂 Résultats : {rec.results_dir(cfg.get('results_dir'))}")
    print("📊 Pour voir les records : python skai/ratio_lab/server.py")


if __name__ == "__main__":
    main()
