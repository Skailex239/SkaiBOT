"""RatioLab — petite UI web qui affiche les records du bot.

Aucune dépendance (Python stdlib uniquement). Elle sert :
- les fichiers de `webui/` (la page) ;
- `/api/state`  : runs + records + état du balayage en cours ;
- `/api/sweep`  : LANCE un balayage depuis la page (ratios, fréquence, durée configurables) ;
- `/api/clear`  : efface les résultats.

Démarrage :
    python skai/ratio_lab/server.py                 # http://localhost:8080
    python skai/ratio_lab/server.py --port 9000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional
from urllib.parse import urlparse

# Permet de lancer le script directement (python skai/ratio_lab/server.py)
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

from skai.ratio_lab import records as rec  # noqa: E402
from skai.ratio_lab import sweep  # noqa: E402

WEBUI_DIR = os.path.join(_HERE, "webui")

DEFAULTS = {
    "map": sweep.DEFAULT_MAP,
    "duration_s": sweep.DEFAULT_DURATION_S,
    "ratios": sweep.DEFAULT_RATIOS,
    "every": sweep.DEFAULT_EVERY,
    "repeats": 1,
    "jobs": max(1, min(4, (os.cpu_count() or 2))),
    "max_attacks": sweep.DEFAULT_MAX_ATTACKS,
    "seed": sweep.DEFAULT_SEED,
}


class SweepStatus:
    """État partagé du balayage en cours (consulté par /api/state)."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.data: Dict[str, Any] = {
            "running": False, "done": 0, "total": 0, "current": None,
            "params": None, "started_at": None, "finished_at": None, "error": None,
        }

    def snapshot(self) -> Dict[str, Any]:
        with self.lock:
            return dict(self.data)

    def update(self, **kw: Any) -> None:
        with self.lock:
            self.data.update(kw)

    def on_event(self, ev: Dict[str, Any]) -> None:
        if ev["event"] == "sweep_start":
            self.update(running=True, done=0, total=ev["total"], params=ev.get("cfg"),
                        started_at=time.strftime("%H:%M:%S"), finished_at=None, error=None)
        elif ev["event"] == "run_start":
            self.update(current=f"ratio {ev['ratio_pct']} % | toutes les {ev['every_s']:g} s")
        elif ev["event"] == "run_done":
            self.update(done=ev["done"], total=ev["total"])
        elif ev["event"] == "sweep_done":
            self.update(running=False, current=None, finished_at=time.strftime("%H:%M:%S"))


STATUS = SweepStatus()
_sweep_thread: Optional[threading.Thread] = None


def launch_sweep(params: Dict[str, Any]) -> None:
    global _sweep_thread
    if STATUS.snapshot()["running"]:
        raise RuntimeError("un balayage est déjà en cours")

    cfg = {
        "map": str(params.get("map") or DEFAULTS["map"]),
        "duration_s": float(params.get("duration_s") or DEFAULTS["duration_s"]),
        "ratios": sweep.parse_int_list(params.get("ratios") or DEFAULTS["ratios"], "ratios"),
        "every": sweep.parse_float_list(params.get("every") or DEFAULTS["every"], "every"),
        "repeats": max(1, int(params.get("repeats") or DEFAULTS["repeats"])),
        "jobs": max(1, min(8, int(params.get("jobs") or DEFAULTS["jobs"]))),
        "max_attacks": max(1, int(params.get("max_attacks") or DEFAULTS["max_attacks"])),
        "seed": int(params.get("seed") or DEFAULTS["seed"]),
        "players": 1,
        "results_dir": None,
    }

    def _run() -> None:
        try:
            sweep.run_sweep(cfg, status_cb=STATUS.on_event)
        except Exception as e:  # noqa: BLE001
            STATUS.update(running=False, error=str(e), finished_at=time.strftime("%H:%M:%S"))

    _sweep_thread = threading.Thread(target=_run, daemon=True)
    _sweep_thread.start()


class Handler(BaseHTTPRequestHandler):
    server_version = "RatioLab/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:  # silencieux
        pass

    # ------------------------------------------------------------------ réponses
    def _json(self, obj: Any, code: int = 200) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path: str, content_type: str) -> None:
        try:
            with open(path, "rb") as f:
                body = f.read()
        except OSError:
            self._json({"error": "introuvable"}, 404)
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            return {}

    # ------------------------------------------------------------------ GET
    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._file(os.path.join(WEBUI_DIR, "index.html"), "text/html; charset=utf-8")
        elif path == "/api/state":
            directory = rec.results_dir()
            runs = rec.load_runs(directory)
            try:
                with open(rec.records_path(directory), "r", encoding="utf-8") as f:
                    records = json.load(f)
            except (OSError, json.JSONDecodeError):
                records = rec.build_records(runs)
            self._json({
                "runs": runs[-400:],
                "records": records,
                "status": STATUS.snapshot(),
                "defaults": DEFAULTS,
            })
        else:
            self._json({"error": "route inconnue"}, 404)

    # ------------------------------------------------------------------ POST
    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        body = self._read_body()
        try:
            if path == "/api/sweep":
                launch_sweep(body)
                self._json({"ok": True, "message": "balayage lancé"}, 202)
            elif path == "/api/clear":
                if STATUS.snapshot()["running"]:
                    self._json({"ok": False, "error": "balayage en cours"}, 409)
                else:
                    rec.clear_results(rec.results_dir())
                    self._json({"ok": True})
            else:
                self._json({"error": "route inconnue"}, 404)
        except Exception as e:  # noqa: BLE001
            self._json({"ok": False, "error": str(e)}, 400)


def main() -> None:
    ap = argparse.ArgumentParser(description="RatioLab — UI web des records du bot.")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"📊 RatioLab — tableau des records sur http://localhost:{args.port}")
    print("   (Ctrl+C pour arrêter)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nArrêt.")


if __name__ == "__main__":
    main()
