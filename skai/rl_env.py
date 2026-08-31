"""
Environnement de renforcement SkaiBOT, branché directement sur le pont du visualiseur.

- Léger : observation 64×64 (parfait sur CPU et carte 100×100).
- Actions : 45 (9 directions × 5 intensités), exactement comme le moteur.
- Récompense : progression du territoire + survie + bonus de victoire / malus de mort.
- Un seul environnement (pas de multiprocessing) => fiable sous Windows.
"""
import os
import sys

import gymnasium as gym
import numpy as np
from gymnasium import spaces

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

from skai.viewer.bridge import GameBridge  # noqa: E402

OBS_SIZE = 64
FRAME_STACK = 3
DIRECTIONS = 9        # 0..7 directions, 8 = attente
INTENSITIES = 5       # 5 niveaux d'intensité
INTENSITY_VALUES = [0.15, 0.30, 0.45, 0.60, 0.80]


def _downsample(grid, size=OBS_SIZE):
    """Réduit une grille (h,w) en (size,size) : redimensionnement 'plus proche voisin'
    sur les masques binaires (très rapide, sans boucle Python)."""
    grid = np.asarray(grid, dtype=np.float32)
    h, w = grid.shape
    ys = (np.arange(size) * h / size).astype(np.int64)
    xs = (np.arange(size) * w / size).astype(np.int64)
    ys = np.clip(ys, 0, h - 1)
    xs = np.clip(xs, 0, w - 1)
    return grid[ys][:, xs]


class SkaiEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, map_name="australia_100x100_nt", num_players=6, max_ticks=1500, verbose=False):
        super().__init__()
        self.map_name = map_name
        self.num_players = num_players
        self.max_ticks = max_ticks
        self.verbose = verbose
        self.bridge = GameBridge()

        self.action_space = spaces.Discrete(DIRECTIONS * INTENSITIES)
        self.observation_space = spaces.Dict({
            "map": spaces.Box(0.0, 1.0, (OBS_SIZE, OBS_SIZE, 3 * FRAME_STACK), dtype=np.float32),
            "global": spaces.Box(-1.0, 1.0, (6,), dtype=np.float32),
        })
        self._frames = []
        self._prev_pct = 0.0
        self.tick = 0

    def _make_obs_frame(self, state):
        tmap = np.array(state["territory_map"], dtype=np.float32)
        own = (tmap == 1).astype(np.float32)
        enemy = (tmap >= 2).astype(np.float32)
        neutral = (tmap == 0).astype(np.float32)
        # Note: le "0" inclut aussi l'eau ; le moteur gère ça, c'est suffisant pour démarrer.
        f = np.stack([
            _downsample(own),
            _downsample(enemy),
            _downsample(neutral),
        ], axis=2).astype(np.float32)
        return f

    def _stacked(self):
        while len(self._frames) < FRAME_STACK:
            self._frames.append(self._frames[-1] if self._frames else np.zeros((OBS_SIZE, OBS_SIZE, 3), np.float32))
        mp = np.concatenate(self._frames[-FRAME_STACK:], axis=2)
        return {"map": mp.astype(np.float32), "global": self._global.astype(np.float32)}

    def _global_feat(self, state):
        total = max(state.get("total_players", 1), 1)
        return np.array([
            state.get("territory_pct", 0.0),
            min(state.get("population", 0) / 5000.0, 1.0),
            (state.get("rank", 1) - 1) / total,
            state.get("alive_players", 1) / total,
            min(state.get("tick", 0) / self.max_ticks, 1.0),
            min(len(state.get("clusters", [])) / 5.0, 1.0),
        ], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.bridge.start()
        state = self.bridge.reset(self.map_name, self.num_players)
        self.state = state
        self.tick = 0
        self._frames = [self._make_obs_frame(state)]
        self._global = self._global_feat(state)
        self._prev_pct = state.get("territory_pct", 0.0)
        return self._stacked(), {}

    def step(self, action):
        direction = action // INTENSITIES
        intensity_idx = action % INTENSITIES
        # Attaque depuis la grappe principale (0) ; on n'attaque pas si "WAIT" (8)
        if direction < 8:
            intensity = INTENSITY_VALUES[intensity_idx]
            self.bridge.attack(direction=direction, intensity=intensity, cluster_id=0)
        state = self.bridge.tick()
        self.state = state
        self.tick += 1

        self._frames.append(self._make_obs_frame(state))
        self._global = self._global_feat(state)

        pct = state.get("territory_pct", 0.0)
        reward = (pct - self._prev_pct) * 200.0       # progression du territoire
        self._prev_pct = pct
        reward += 0.01                                 # petit bonus de survie

        terminated = bool(state.get("has_lost", False))
        if state.get("has_won") or pct >= 0.80:
            reward += 50.0
            terminated = True
        if state.get("has_lost"):
            reward -= 20.0

        truncated = self.tick >= self.max_ticks
        if self.verbose and self.tick % 100 == 0:
            print(f"  tick {self.tick}: territoire {pct*100:.1f}% rang {state.get('rank')}")

        info = {"tick": self.tick, "territory_pct": pct, "rank": state.get("rank")}
        return self._stacked(), reward, terminated, truncated, info

    def close(self):
        self.bridge.close()
