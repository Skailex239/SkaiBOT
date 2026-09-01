"""
SkaiBOT — visualiseur en direct (aucune dépendance, Python stdlib uniquement).

Lance une partie OpenFront en local, fait jouer l'IA (politique de démo pour l'instant)
et diffuse l'état via une petite API HTTP consommée par la page web `index.html`.

Démarrage :
    python skai/viewer/server.py                                  # politique de démo (gloutonne)
    python skai/viewer/server.py --model skai/models/skai_ppo_australia_100x100.zip
                                                                  # le VRAI réseau entraîné
puis ouvre l'URL affichée (ex. http://localhost:8080).

On peut changer de modèle à chaud sans redémarrer :
    curl -X POST "http://localhost:8080/api/model?path=skai/models/skai_ppo_australia_100x100_best.zip"
    curl -X POST "http://localhost:8080/api/model?path="        -> revient à la gloutonne
"""
import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

# Permet de lancer le script directement (python skai/viewer/server.py)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from skai.viewer.bridge import GameBridge          # noqa: E402
from skai.viewer.agent_policy import play_one_tick  # noqa: E402

MODEL_PATH = os.environ.get("SKAI_MODEL", "")   # vide => politique de démo

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MAP = "australia_100x100_nt"
DEFAULT_PLAYERS = 11
TICK_SLEEP = 0.12  # ≈ 8 ticks / seconde


def encode_rle(tmap):
    """Run-length encoding de la carte (plate, ligne par ligne). Renvoie [valeur, compte, ...]."""
    runs = []
    prev = None
    count = 0
    for row in tmap:
        for v in row:
            if v == prev:
                count += 1
            else:
                if prev is not None:
                    runs.append(prev)
                    runs.append(count)
                prev = v
                count = 1
    if prev is not None:
        runs.append(prev)
        runs.append(count)
    return runs


class GameSession:
    def __init__(self):
        self.bridge = GameBridge()
        self.lock = threading.Lock()
        self.state = None
        self.actions = []
        self.status = "arrêté"
        self.map_name = DEFAULT_MAP
        self.num_players = DEFAULT_PLAYERS
        self.last_reset = 0.0
        self.player = None                 # ModelPlayer si un .zip est chargé
        self.policy_name = "glouton (démo)"
        self.model_path = ""
        self.max_ticks = 1500

    def set_model(self, path):
        """Charge un modèle SB3 (path vide = politique de démo). Peut être appelé à chaud."""
        prev, prev_name, prev_path = self.player, self.policy_name, self.model_path
        with self.lock:
            if not path:
                self.player, self.model_path = None, ""
                self.policy_name = "glouton (démo)"
                return "politique de démo (glouton) réactivée"
            try:
                from skai.model_policy import ModelPlayer
                self.player = ModelPlayer(path, max_ticks=self.max_ticks)
                self.model_path = path
                self.policy_name = self.player.name
                return f"modèle chargé : {path}"
            except Exception as e:  # noqa: BLE001
                # on NE casse pas ce qui jouait déjà : un chemin tapé de travers dans la
                # page ne doit pas éteindre le réseau en train de jouer.
                self.player, self.model_path, self.policy_name = prev, prev_path, prev_name
                return f"impossible de charger {path} : {e} — modèle en cours conservé ({prev_name})"

    def reset(self, map_name=DEFAULT_MAP, num_players=DEFAULT_PLAYERS):
        with self.lock:
            self.status = "chargement"
            self.map_name = map_name
            self.num_players = num_players
        # Le reset charge la carte (plusieurs secondes) : on le fait hors du verrou long
        try:
            self.bridge.start()
            state = self.bridge.reset(map_name, num_players, game_config="default")
            if self.player is not None and hasattr(self.player, "reset"):
                self.player.reset()   # l'historique de frames doit repartir de zéro
            with self.lock:
                self.state = state
                self.actions = []
                self.status = "en_cours"
                self.last_reset = time.time()
        except Exception as e:  # noqa: BLE001
            with self.lock:
                self.status = "erreur"
                self.actions = [f"ERREUR: {e}"]

    def step(self):
        with self.lock:
            if self.status != "en_cours" or not self.state:
                return
            if self.state.get("game_over"):
                self.status = "terminé"
                return
            state = self.state
        # Décision de l'IA + avance d'un tick
        try:
            player = self.player
            actions = player.play_one_tick(self.bridge, state) if player is not None \
                else play_one_tick(self.bridge, state)
            new_state = self.bridge.tick()
            with self.lock:
                self.state = new_state
                self.actions = actions
        except Exception as e:  # noqa: BLE001
            with self.lock:
                self.actions = [f"ERREUR tick: {e}"]

    def snapshot(self):
        with self.lock:
            if not self.state:
                return {"status": self.status, "map": self.map_name,
                        "players": self.num_players, "actions": self.actions,
                        "error": self.actions[-1] if self.actions else ""}
            s = self.state
            w = len(s["territory_map"][0])
            h = len(s["territory_map"])
            clusters = [
                {"id": c["id"], "x": c["center_x"] / w, "y": c["center_y"] / h,
                 "tiles": c.get("tile_count", len(c.get("tiles", []))), "troops": c.get("troop_count", 0)}
                for c in s.get("clusters", [])
            ]
            return {
                "status": self.status,
                "map": self.map_name,
                "players": self.num_players,
                "width": w,
                "height": h,
                "tick": s.get("tick"),
                "territory_pct": s.get("territory_pct"),
                "tiles_owned": s.get("tiles_owned"),
                "neutral_tiles": s.get("neutral_tiles"),
                "rank": s.get("rank"),
                "total_players": s.get("total_players"),
                "alive_players": s.get("alive_players"),
                "population": s.get("population"),
                "gold": s.get("gold"),
                "game_over": s.get("game_over"),
                "has_won": s.get("has_won"),
                "has_lost": s.get("has_lost"),
                "clusters": clusters,
                "policy": self.policy_name,
                "model": self.model_path,
                "actions": self.actions,
                "rle": encode_rle(s["territory_map"]),
            }


SESSION = GameSession()


def background_loop():
    # Attend que la première partie soit lancée
    while True:
        if SESSION.status == "en_cours":
            SESSION.step()
        time.sleep(TICK_SLEEP)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # silence
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        url = urlparse(self.path)
        if url.path in ("/", "/index.html"):
            with open(os.path.join(HERE, "index.html"), "rb") as f:
                self._send(200, f.read(), "text/html; charset=utf-8")
        elif url.path == "/api/state":
            self._send(200, SESSION.snapshot())
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        url = urlparse(self.path)
        if url.path == "/api/reset":
            qs = parse_qs(url.query)
            map_name = qs.get("map", [DEFAULT_MAP])[0]
            players = int(qs.get("players", [DEFAULT_PLAYERS])[0])
            threading.Thread(target=SESSION.reset, args=(map_name, players), daemon=True).start()
            self._send(200, {"ok": True, "status": "chargement"})
        elif url.path == "/api/model":
            qs = parse_qs(url.query)
            msg = SESSION.set_model(qs.get("path", [""])[0])
            self._send(200, {"ok": True, "message": msg, "policy": SESSION.policy_name})
        else:
            self._send(404, {"error": "not found"})


def main():
    ap = argparse.ArgumentParser(description="Visualiseur SkaiBOT (stdlib uniquement)")
    ap.add_argument("--model", default=MODEL_PATH,
                    help=".zip sauvegardé par skai/train_simple.py — sinon la gloutonne de démo joue")
    ap.add_argument("--map", default=DEFAULT_MAP)
    ap.add_argument("--players", type=int, default=DEFAULT_PLAYERS)
    ap.add_argument("--max-ticks", type=int, default=1500)
    ap.add_argument("--stochastic", action="store_true",
                    help="échantillonne au lieu de prendre l'action maximale (voit l'exploration)")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8080")))
    args = ap.parse_args()

    SESSION.map_name, SESSION.num_players, SESSION.max_ticks = args.map, args.players, args.max_ticks
    if args.model:
        print(SESSION.set_model(args.model))
        if args.stochastic and SESSION.player is not None:
            SESSION.player.deterministic = False

    threading.Thread(target=background_loop, daemon=True).start()
    # Lancer une première partie automatiquement
    threading.Thread(target=SESSION.reset, args=(SESSION.map_name, SESSION.num_players), daemon=True).start()
    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print("=== SkaiBOT visualiseur ===")
    print(f"Ouvre ton navigateur sur :  http://localhost:{args.port}")
    print(f"Carte : {SESSION.map_name} ({SESSION.num_players} joueurs) — IA : {SESSION.policy_name}")
    print("Pour relancer : POST /api/reset?map=...&players=... — changer d'IA : POST /api/model?path=...")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
