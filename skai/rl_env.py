"""
Environnement de renforcement SkaiBOT, branché directement sur le pont du visualiseur.

- Léger : observation 64×64 (le pont renvoie déjà une carte réduite => rapide).
- Actions : 45 (9 directions × 5 intensités), appliquées par le pont comme de VRAIES
  attaques directionnelles (front limité à la tuile-source choisie).
- Récompense : progression du territoire (en % des TUILES TERRESTRES) + bonus de
  victoire / malus de mort.
- Un seul environnement (pas de multiprocessing) => fiable sous Windows.

Voir docs/DIAGNOSTIC.md pour les 4 corrections du pont dont cet environnement dépend :
  1. config de jeu = DefaultConfig (et non TestConfig, qui plafonnait la conquête à
     1 tuile par tick) ;
  2. attaques avec sourceTile (sinon le moteur ignore la direction choisie) ;
  3. territory_pct mesuré sur le nombre de tuiles terrestres (sinon 80 % est
     mathématiquement inatteignable sur une carte avec de l'océan) ;
  4. anti-cumul des attaques + WAIT qui rappelle les troupes (sinon le agent s'appauvrit
     tout seul).
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
CHANNELS = 4          # own, enemy, neutral land, water/land mask
DIRECTIONS = 9        # 0..7 directions, 8 = attente (rappelle les attaques en cours)
INTENSITIES = 5       # 5 niveaux d'intensité
INTENSITY_VALUES = [0.15, 0.30, 0.45, 0.60, 0.80]


def _downsample(grid, size=OBS_SIZE):
    """Réduit une grille (h,w) en (size,size) : redimensionnement 'plus proche voisin'
    sur les masques binaires (très rapide, sans boucle Python)."""
    grid = np.asarray(grid, dtype=np.float32)
    h, w = grid.shape
    if (h, w) == (size, size):
        return grid
    ys = np.clip((np.arange(size) * h / size).astype(np.int64), 0, h - 1)
    xs = np.clip((np.arange(size) * w / size).astype(np.int64), 0, w - 1)
    return grid[ys][:, xs]


# --------------------------------------------------------------------------- observations
# Ces fonctions sont AU NIVEAU DU MODULE (et non des méthodes) pour une raison précise :
# le visualiseur et le script d'évaluation doivent construire l'observation EXACTEMENT
# comme pendant l'entraînement. Un décalage d'un canal ou d'une normalisation entre
# entraînement et test suffit à rendre un modèle excellent dans les logs et nul en jeu.

SHIFTS = [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]


def land_mask_from(state):
    """Masque de terres réduit à OBS_SIZE (le pont l'envoie déjà à 64x64)."""
    return _downsample(np.array(state.get("water_mask", np.ones((OBS_SIZE, OBS_SIZE))), dtype=np.float32))


def make_frame(state, land):
    """4 canaux : à nous / ennemis / terres neutres / mer.

    L'ancien code confondait `0` (neutre) et eau : le moteur refuse de faire passer des
    troupes sur l'eau (isWater dans AttackExecution.addNeighbors), donc sans ce canal
    l'agent ne peut pas deviner pourquoi il ne progresse pas.
    """
    # on réduit à OBS_SIZE ici : le visualiseur, lui, reçoit le territory_map à la
    # résolution complète de la carte (100x100, 500x500...) pour un affichage net.
    tmap = _downsample(np.array(state["territory_map"], dtype=np.float32))
    own = (tmap == 1).astype(np.float32)
    enemy = (tmap >= 2).astype(np.float32)
    neutral = ((tmap == 0) & (land > 0.5)).astype(np.float32)
    water = (land <= 0.5).astype(np.float32)
    return np.stack([own, enemy, neutral, water], axis=2).astype(np.float32)


def global_feat(state, max_ticks):
    total = max(state.get("total_players", 1), 1)
    return np.array([
        state.get("territory_pct", 0.0),                    # % des TERRES
        min(state.get("population", 0) / 50000.0, 1.0),
        (state.get("rank", 1) - 1) / total,
        state.get("alive_players", 1) / total,
        min(state.get("tick", 0) / max(max_ticks, 1), 1.0),
        min(len(state.get("clusters", [])) / 5.0, 1.0),
        min(state.get("border_tiles", 0) / 500.0, 1.0),
        state.get("territory_change", 0.0),
    ], dtype=np.float32)


def stack_frames(frames, glob):
    frames = list(frames)
    if not frames:
        frames = [np.zeros((OBS_SIZE, OBS_SIZE, CHANNELS), np.float32)]
    while len(frames) < FRAME_STACK:
        frames = [frames[0]] + frames
    return {"map": np.concatenate(frames[-FRAME_STACK:], axis=2).astype(np.float32),
            "global": np.asarray(glob, dtype=np.float32)}


def action_to_command(action):
    """Index d'action (0..44) -> (direction, intensité) ; direction 8 = WAIT."""
    action = int(action)
    direction = action // INTENSITIES
    if direction >= 8:
        return 8, 0.0
    return direction, INTENSITY_VALUES[action % INTENSITIES]


def legal_directions(tmap, land):
    """Directions ayant au moins une tuile de frontière touchant une terre prenable."""
    own = tmap == 1
    conquestable = (tmap != 1) & (land > 0.5)
    return [d for d, (dx, dy) in enumerate(SHIFTS)
            if bool((np.roll(np.roll(own, dy, axis=0), dx, axis=1) & conquestable).any())]


class SkaiEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        map_name="australia_100x100_nt",
        num_players=6,
        max_ticks=1500,
        win_threshold=0.80,
        game_config="default",
        spawn_mode="land",
        living_reward=-0.005,   # pression temporelle (0.0 = neutre, positif = récompense de survie)
        spawn_seed=None,        # None = une graine différente à chaque épisode
        verbose=False,
    ):
        super().__init__()
        self.map_name = map_name
        self.num_players = num_players
        self.max_ticks = max_ticks
        self.win_threshold = win_threshold
        self.game_config = game_config
        self.spawn_mode = spawn_mode
        self.living_reward = living_reward
        self.verbose = verbose
        self._spawn_seed = spawn_seed
        self.bridge = GameBridge()

        self.action_space = spaces.Discrete(DIRECTIONS * INTENSITIES)
        self.observation_space = spaces.Dict({
            "map": spaces.Box(0.0, 1.0, (OBS_SIZE, OBS_SIZE, CHANNELS * FRAME_STACK), dtype=np.float32),
            "global": spaces.Box(-1.0, 1.0, (8,), dtype=np.float32),
        })
        self._frames = []
        self._prev_pct = 0.0
        self.tick = 0
        self._land = None

    # ------------------------------------------------------------------ observations
    def _make_obs_frame(self, state):
        return make_frame(state, self._land)

    def _stacked(self):
        self._frames = self._frames[-FRAME_STACK:]
        return stack_frames(self._frames, self._global)

    def _global_feat(self, state):
        return global_feat(state, self.max_ticks)

    # ------------------------------------------------------------------ action mask
    def action_masks(self) -> np.ndarray:
        """Masque d'actions valides (convention sb3-contrib / MaskablePPO).

        Une direction est légale si au moins une de nos tuiles de frontière a un
        voisin terrestre non possédé dans cette direction. `attente` est toujours
        légale (elle rappelle les attaques en cours)."""
        mask = np.zeros(self.action_space.n, dtype=np.float32)
        tmap = getattr(self, "_tmap", None)
        land = getattr(self, "_land", None)
        if tmap is None or land is None:
            mask[:] = 1.0
            return mask
        for d in legal_directions(tmap, land):
            mask[d * INTENSITIES:(d + 1) * INTENSITIES] = 1.0
        mask[8 * INTENSITIES:9 * INTENSITIES] = 1.0
        return mask

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.bridge.start()
        # graine de placement différente à chaque épisode (sauf si on veut du reproduisible)
        seed = self._spawn_seed
        if seed is None and self.np_random is not None:
            seed = int(self.np_random.integers(1, 2**31 - 1))
        state = self.bridge.reset(
            self.map_name,
            self.num_players,
            obs_size=OBS_SIZE,
            game_config=self.game_config,
            win_threshold=self.win_threshold,
            spawn_mode=self.spawn_mode,
            spawn_seed=seed,
        )
        self.tick = 0
        self._prep(state)
        self._frames = [self._make_obs_frame(state)]
        self._prev_pct = state.get("territory_pct", 0.0)
        return self._stacked(), {}

    def _prep(self, state):
        """Carte réduite (OBS_SIZE) + masque de terres, recalculés une fois par tick."""
        self.state = state
        tmap = _downsample(np.array(state["territory_map"], dtype=np.float32))
        self._tmap = tmap
        self._land = _downsample(np.array(state.get("water_mask", np.ones((OBS_SIZE, OBS_SIZE))), dtype=np.float32))
        self._global = self._global_feat(state)

    def step(self, action):
        direction, intensity = action_to_command(action)
        if direction < 8:
            self.bridge.attack(direction=direction, intensity=intensity, cluster_id=0)
        # direction 8 = WAIT : vrai no-op. On ne rappelle PAS les attaques ici : le
        # moteur facture la retraite 25 % des troupes engagées, donc "ne rien faire"
        # ne doit rien coûter. `bridge.cancel_attacks()` reste disponible pour une
        # action dédiée (utile en fin de speedrun pour reconsolider l'arrière).

        state = self.bridge.tick()
        self.tick += 1
        self._prep(state)

        pct = state.get("territory_pct", 0.0)
        reward = (pct - self._prev_pct) * 200.0       # progression du territoire
        self._prev_pct = pct
        reward += self.living_reward

        terminated = bool(state.get("has_lost", False))
        if state.get("has_won") or pct >= self.win_threshold:
            reward += 50.0
            terminated = True
        if state.get("has_lost"):
            reward -= 20.0

        truncated = self.tick >= self.max_ticks
        if self.verbose and self.tick % 100 == 0:
            print(f"  tick {self.tick}: territoire {pct*100:.1f}% rang {state.get('rank')} "
                  f"troupes {int(state.get('population', 0))}")

        info = {"tick": self.tick, "territory_pct": pct, "rank": state.get("rank")}
        return self._stacked(), reward, terminated, truncated, info

    def close(self):
        self.bridge.close()
