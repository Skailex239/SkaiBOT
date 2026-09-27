"""RatioLab — lecture des runs, calcul des records et exports.

Aucune dépendance externe (Python stdlib uniquement).

Fichiers produits dans le dossier `results/` :
- `runs.jsonl`    : une ligne JSON par run (append au fil des tests)
- `runs.csv`      : même chose au format tableur (réécrit à chaque run)
- `records.json`  : records dérivés des runs (classements, matrices, records par catégorie)
"""
from __future__ import annotations

import csv
import json
import os
import time
from typing import Any, Dict, List, Optional

CSV_COLUMNS = [
    "ts", "map", "duration_s", "ratio_pct", "every_s", "interval_ticks", "seed", "repeat_idx",
    "max_attacks", "tiles_start", "tiles_end", "tiles_gained", "land_tiles",
    "land_pct", "tiles_per_sec", "tiles_per_tick", "active_ticks",
    "troops_start", "troops_end", "troops_min", "waves_attempted", "waves_ok",
    "t10_ticks", "t25_ticks", "t50_ticks", "t75_ticks", "wall_time_s", "status",
]


def results_dir(explicit: Optional[str] = None) -> str:
    d = explicit or os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(d, exist_ok=True)
    return d


def runs_path(directory: str) -> str:
    return os.path.join(directory, "runs.jsonl")


def records_path(directory: str) -> str:
    return os.path.join(directory, "records.json")


def load_runs(directory: str) -> List[Dict[str, Any]]:
    """Toutes les runs enregistrées (les lignes corrompues sont ignorées)."""
    path = runs_path(directory)
    runs: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return runs
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                runs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return runs


def append_run(directory: str, run: Dict[str, Any]) -> None:
    """Ajoute une run au journal JSONL puis régénère le CSV et les records."""
    with open(runs_path(directory), "a", encoding="utf-8") as f:
        f.write(json.dumps(run, ensure_ascii=False) + "\n")
    runs = load_runs(directory)
    write_csv(directory, runs)
    save_records(directory, runs)


def write_csv(directory: str, runs: List[Dict[str, Any]]) -> None:
    path = os.path.join(directory, "runs.csv")
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in runs:
            w.writerow(r)


def clear_results(directory: str) -> None:
    for name in ("runs.jsonl", "runs.csv", "records.json"):
        p = os.path.join(directory, name)
        if os.path.exists(p):
            os.remove(p)


# --------------------------------------------------------------------------- records

def _group_key(run: Dict[str, Any]) -> str:
    return f"{run.get('map')}|{run.get('duration_s')}"


def _summarize(run: Dict[str, Any]) -> Dict[str, Any]:
    """Les champs qui intéressent le tableau des records."""
    keys = [
        "ts", "map", "duration_s", "ratio_pct", "every_s", "interval_ticks", "seed", "repeat_idx",
        "tiles_gained", "land_pct", "tiles_per_sec", "tiles_per_tick", "active_ticks",
        "troops_start", "troops_end", "troops_min", "waves_attempted", "waves_ok",
        "t25_ticks", "t50_ticks", "t75_ticks", "status",
    ]
    return {k: run.get(k) for k in keys}


def build_records(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Dérive les records de toutes les runs valides.

    Une run est valide si `status == "ok"` et que `tiles_per_sec` est un nombre.
    Les records sont séparés par (carte, durée) : comparer une run de 30 s avec
    une run de 120 s n'aurait aucun sens.
    """
    valid = [
        r for r in runs
        if r.get("status") in ("ok", "won") and isinstance(r.get("tiles_per_sec"), (int, float))
    ]
    groups: Dict[str, Dict[str, Any]] = {}
    for r in valid:
        key = _group_key(r)
        g = groups.setdefault(key, {"map": r.get("map"), "duration_s": r.get("duration_s"), "runs": []})
        g["runs"].append(r)

    out_groups: List[Dict[str, Any]] = []
    for key, g in sorted(groups.items(), key=lambda kv: (str(kv[1]["map"]), float(kv[1]["duration_s"] or 0))):
        gruns = g["runs"]
        best_overall = max(gruns, key=lambda r: r["tiles_per_sec"])
        leaderboard = sorted(gruns, key=lambda r: -r["tiles_per_sec"])[:20]

        # matrice ratio × fréquence : meilleure run de chaque case
        matrix: Dict[str, Any] = {}
        for r in gruns:
            cell = f"{r['ratio_pct']}x{r['every_s']}"
            cur = matrix.get(cell)
            if cur is None or r["tiles_per_sec"] > cur["tiles_per_sec"]:
                matrix[cell] = _summarize(r)

        def _best_index(field: str) -> Dict[str, Any]:
            acc: Dict[str, Any] = {}
            for r in gruns:
                k = str(r.get(field))
                cur = acc.get(k)
                if cur is None or r["tiles_per_sec"] > cur["tiles_per_sec"]:
                    acc[k] = _summarize(r)
            return acc

        out_groups.append({
            "map": g["map"],
            "duration_s": g["duration_s"],
            "n_runs": len(gruns),
            "best_overall": _summarize(best_overall),
            "leaderboard": [_summarize(r) for r in leaderboard],
            "matrix": matrix,
            "best_by_ratio": _best_index("ratio_pct"),   # meilleure fréquence pour un ratio donné
            "best_by_every": _best_index("every_s"),     # meilleur ratio pour une fréquence donnée
        })

    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_runs": len(runs),
        "n_runs_valid": len(valid),
        "groups": out_groups,
    }


def save_records(directory: str, runs: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    if runs is None:
        runs = load_runs(directory)
    records = build_records(runs)
    with open(records_path(directory), "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    return records
