"""
Communication avec le pont TypeScript (game_bridge) qui pilote le moteur OpenFront.

Le pont est un processus Node (`npx tsx game_bridge.ts`) qui échange des commandes
JSON sur stdin/stdout, une commande par ligne.
"""
import json
import os
import subprocess
import sys
import threading
import time
from typing import Any, Dict, Optional


# Direction: 0=N, 1=NE, 2=E, 3=SE, 4=S, 5=SW, 6=W, 7=NW, 8=ATTENTE
DIR_VECTORS = {
    0: (0, -1),
    1: (1, -1),
    2: (1, 0),
    3: (1, 1),
    4: (0, 1),
    5: (-1, 1),
    6: (-1, 0),
    7: (-1, -1),
}
DIR_NAMES = ["N", "NE", "E", "SE", "S", "SW", "W", "NW", "WAIT"]


class GameBridge:
    def __init__(self, bridge_ts: Optional[str] = None):
        if bridge_ts is None:
            here = os.path.dirname(os.path.abspath(__file__))
            # skai/viewer/bridge.py -> agent/game_bridge/game_bridge.ts
            bridge_ts = os.path.join(here, "..", "..", "agent", "game_bridge", "game_bridge.ts")
        self.bridge_ts = os.path.abspath(bridge_ts)
        self.proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()

    def start(self):
        if self.proc is not None and self.proc.poll() is None:
            return
        cwd = os.path.dirname(self.bridge_ts)
        is_windows = os.name == "nt" or sys.platform.startswith("win")

        if is_windows:
            # Sur Windows, `npx` est un fichier npx.cmd : il faut passer par le shell.
            # On passe le chemin du script entre guillemets (les espaces sont gérés).
            cmd = f'npx tsx "{self.bridge_ts}"'
            self.proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=cwd,
                text=True,
                bufsize=1,
                shell=True,
            )
        else:
            self.proc = subprocess.Popen(
                ["npx", "tsx", self.bridge_ts],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=cwd,
                text=True,
                bufsize=1,
            )

        # Vide stderr en arrière-plan pour éviter qu'un buffer plein ne bloque le process.
        self._stderr_tail = []
        def _drain():
            assert self.proc is not None and self.proc.stderr is not None
            for line in self.proc.stderr:
                self._stderr_tail.append(line.rstrip())
                if len(self._stderr_tail) > 200:
                    self._stderr_tail.pop(0)
        threading.Thread(target=_drain, daemon=True).start()
        # Laisse le temps au process Node de démarrer (au premier lancement npx installe tsx).
        time.sleep(1.0)

    def _send(self, command: Dict[str, Any]) -> Dict[str, Any]:
        if self.proc is None or self.proc.poll() is not None:
            raise RuntimeError("Le pont n'est pas démarré (ou il a crashé).")
        line = json.dumps(command) + "\n"
        with self._lock:
            assert self.proc.stdin is not None
            self.proc.stdin.write(line)
            self.proc.stdin.flush()
            assert self.proc.stdout is not None
            resp_line = self.proc.stdout.readline()
            if not resp_line:
                err = "\n".join(getattr(self, "_stderr_tail", [])[-25:])
                rc = self.proc.poll()
                raise RuntimeError(
                    "Le pont n'a pas répondu. Vérifie que Node.js est installé et que "
                    f"`npm install` a été fait dans base-game. (code sortie={rc})\nDétail Node:\n{err}"
                )
            return json.loads(resp_line)

    def reset(
        self,
        map_name: str = "australia_100x100_nt",
        num_players: int = 11,
        obs_size: Optional[int] = None,
        game_config: Optional[str] = None,
        win_threshold: Optional[float] = None,
        max_attacks: Optional[int] = None,
        spawn_mode: Optional[str] = None,
        spawn_seed: Optional[int] = None,
        verbose: bool = False,
        allow_multi_front: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Démarre une partie.

        obs_size     : le pont renvoie territory_map déjà réduit à obs_size x obs_size
                       (évite de sérialiser 250 000 nombres par tick sur les grandes cartes).
        game_config  : "default" = vraies règles d'OpenFront, "test" = ancien comportement
                       (TestConfig, conquête bridée à 1 tuile/tick).
        win_threshold: fraction des TUILES TERRESTRES nécessaire pour `has_won`.
        allow_multi_front : autorise plusieurs fronts simultanés contre le même
                       adversaire (utilisé par le bot ratio-lab). Désactivé par défaut.
        """
        cmd: Dict[str, Any] = {"type": "reset", "map_name": map_name, "num_players": num_players}
        if obs_size:
            cmd["obs_size"] = int(obs_size)
        if game_config:
            cmd["game_config"] = game_config
        if win_threshold is not None:
            cmd["win_threshold"] = float(win_threshold)
        if max_attacks is not None:
            cmd["max_attacks"] = int(max_attacks)
        if spawn_mode:
            cmd["spawn_mode"] = spawn_mode
        if spawn_seed is not None:
            cmd["spawn_seed"] = int(spawn_seed) & 0x7FFFFFFF
        if verbose:
            cmd["verbose"] = True
        if allow_multi_front is not None:
            cmd["allow_multi_front"] = bool(allow_multi_front)
        r = self._send(cmd)
        return r.get("state", r)

    def tick(self) -> Dict[str, Any]:
        r = self._send({"type": "tick"})
        return r.get("state", r)

    def attack(self, direction: int, intensity: float = 1.0, cluster_id: int = 0) -> bool:
        r = self._send({
            "type": "attack_direction",
            "cluster_id": int(cluster_id),
            "direction": int(direction),
            "intensity": float(intensity),
        })
        return bool(r.get("success", False))

    def cancel_attacks(self) -> int:
        """Rappelle toutes les attaques en cours (WAIT). Coûte 25 % des troupes
        engagées : c'est le seul moyen de stopper l'hémorragie de population."""
        r = self._send({"type": "cancel_attacks"})
        return int(r.get("cancelled", 0))

    def close(self):
        if self.proc is None:
            return
        try:
            self._send({"type": "shutdown"})
        except Exception:
            pass
        try:
            self.proc.terminate()
        except Exception:
            pass
        self.proc = None


if __name__ == "__main__":
    # Test rapide
    b = GameBridge()
    b.start()
    t0 = time.time()
    state = b.reset("australia_100x100_nt", 11)
    print("reset ok — tick", state["tick"], "tuiles", state["tiles_owned"], f"({time.time()-t0:.1f}s)")
    state = b.tick()
    print("tick ok — tick", state["tick"])
    b.close()
