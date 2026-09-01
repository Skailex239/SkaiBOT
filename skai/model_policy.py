"""
SkaiBOT — faire jouer un MODÈLE ENTRAÎNÉ (et pas la politique de démo).

Le visualiseur utilisait jusqu'ici `agent_policy.play_one_tick`, une gloutonne écrite à
la main : impossible de voir le réseau de neurones jouer. Cette classe branche un `.zip`
sauvegardé par `skai/train_simple.py` sur le même pont, en reconstruisant l'observation
AVREC LES MÊMES FONCTIONS que l'environnement d'entraînement (`skai/rl_env`) — c'est le
seul garant qu'un modèle bon en log reste bon à l'écran.

Usage direct :
    python -m skai.model_policy skai/models/skai_ppo_australia_100x100.zip 6
"""
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .rl_env import (
    OBS_SIZE,
    _downsample,
    action_to_command,
    global_feat,
    land_mask_from,
    legal_directions,
    make_frame,
    stack_frames,
)
from skai.viewer.bridge import GameBridge  # noqa: F401  (réexporté pour le test rapide ci-dessous)


class ModelPlayer:
    """Joue un tick à partir de l'état courant, comme le faisait l'entraînement."""

    def __init__(self, model_path: str, max_ticks: int = 1500, deterministic: bool = True,
                 device: str = "cpu"):
        from stable_baselines3 import PPO  # import différé : le visualiseur doit rester léger

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"modèle introuvable : {model_path}  (chemin relatif au dossier où tu lances la commande)")
        self.model = PPO.load(model_path, device=device)
        self.model.verbose = 0
        self.max_ticks = max_ticks
        self.deterministic = deterministic
        self.maskable = type(self.model).__name__ == "MaskablePPO"
        self._frames: List[np.ndarray] = []
        self.last_action: Optional[int] = None
        self.name = f"modèle {model_path.split('/')[-1].replace('.zip','')}"

    def reset(self) -> None:
        self._frames = []
        self.last_action = None

    def decide(self, state: Dict[str, Any]) -> Tuple[int, str]:
        """(index d'action, description lisible) à partir de l'état du pont."""
        land = land_mask_from(state)
        self._frames.append(make_frame(state, land))
        self._frames = self._frames[-3:]
        obs = stack_frames(self._frames, global_feat(state, self.max_ticks))

        kwargs: Dict[str, Any] = {"deterministic": self.deterministic}
        if self.maskable:
            # même calcul que SkaiEnv.action_masks() : on ne peut pas laisser un modèle
            # entraîné avec masque jouer sans masque (les logits ne sont pas comparables).
            tmap = _downsample(np.array(state["territory_map"], dtype=np.float32))
            mask = np.zeros(45, dtype=np.float32)
            for d in legal_directions(tmap, land):
                mask[d * 5:(d + 1) * 5] = 1.0
            mask[40:45] = 1.0                      # WAIT toujours autorisé
            if mask.sum() <= 5:                    # rien d'attaquable -> WAIT forcé
                mask[:] = 0.0
                mask[40:45] = 1.0
            kwargs["action_masks"] = {"action_masks": mask}

        action, _ = self.model.predict(obs, **kwargs)
        action = int(action)
        self.last_action = action
        direction, intensity = action_to_command(action)
        if direction == 8:
            return action, "attente (ne rien engager)"
        return action, f"attaque {['N','NE','E','SE','S','SW','W','NW'][direction]} à {intensity*100:.0f}% des troupes libres"

    def predict(self, obs: Dict[str, np.ndarray], env=None) -> int:
        """Action pour une observation déjà construite (chemin de l'évaluation).

        On passe par l'observation de `env` plutôt que par `decide()` : c'est la garantie
        absolue que le modèle reçoit exactement ce qu'il a reçu pendant l'entraînement.
        """
        kwargs: Dict[str, Any] = {"deterministic": self.deterministic}
        if self.maskable and env is not None and hasattr(env, "action_masks"):
            kwargs["action_masks"] = {"action_masks": env.action_masks()}
        action, _ = self.model.predict(obs, **kwargs)
        return int(action)

    def play_one_tick(self, bridge, state: Dict[str, Any]) -> List[str]:
        """Même signature que `agent_policy.play_one_tick` => remplaçant direct."""
        action, desc = self.decide(state)
        direction, intensity = action_to_command(action)
        if direction < 8:
            bridge.attack(direction=direction, intensity=intensity, cluster_id=0)
        return [desc]


if __name__ == "__main__":
    import sys
    import time

    path = sys.argv[1] if len(sys.argv) > 1 else None
    if not path:
        raise SystemExit("usage : python -m skai.model_policy <modele.zip> [nb_joueurs]")
    n_players = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    player = ModelPlayer(path)
    bridge = GameBridge()
    bridge.start()
    state = bridge.reset("australia_100x100", n_players, obs_size=OBS_SIZE)
    t0 = time.time()
    for i in range(600):
        player.play_one_tick(bridge, state)
        state = bridge.tick()
        if state.get("game_over"):
            break
        if i % 100 == 0:
            print(f"  tick {state['tick']:4d}  territoire {state['territory_pct']*100:5.2f}%  rang {state['rank']}")
    print(f"fin : {state['tick']} ticks en {time.time()-t0:.1f}s — territoire {state['territory_pct']*100:.1f}%, "
          f"rang {state['rank']}/{state['alive_players']}, victoire={state['has_won']}")
    bridge.close()
