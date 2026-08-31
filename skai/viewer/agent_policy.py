"""
Politique de démo pour faire JOUER l'IA tout de suite, avant les modèles RL entraînés.

Stratégie : expansion gloutonne. Pour chaque grappe de territoire, on regarde les tuiles
à la frontière et on attaque dans la direction qui offre le plus de terrain à prendre
(neutre = fortement valorisé, ennemi = prudemment). C'est une IA "bête mais active" :
le but est de visualiser une vraie partie en marche ; le vrai niveau viendra du RL/self-play.
"""
from typing import Any, Dict, List, Tuple

from .bridge import DIR_VECTORS, DIR_NAMES


def decide_for_cluster(state: Dict[str, Any], cluster: Dict[str, Any]) -> Tuple[int, float, str]:
    """Retourne (direction, intensité, description) pour une grappe."""
    width = len(state["territory_map"][0])
    height = len(state["territory_map"])
    tmap = state["territory_map"]

    scores = [0.0] * 8
    for tile_idx in cluster.get("border_tiles", []):
        x = tile_idx % width
        y = tile_idx // width
        for d, (dx, dy) in DIR_VECTORS.items():
            nx, ny = x + dx, y + dy
            if not (0 <= nx < width and 0 <= ny < height):
                continue
            v = tmap[ny][nx]
            if v == 1:
                continue  # déjà à nous
            if v == 0:
                scores[d] += 2.0      # terrain neutre : on s'étend
            elif v == 999:
                scores[d] += 0.5
            else:
                scores[d] += 0.3      # ennemi : risqué, peu de poids en early game

    best_dir = max(range(8), key=lambda d: scores[d])
    if scores[best_dir] <= 0:
        return 8, 0.0, "attente (rien à conquérir autour)"

    intensity = 0.8
    return best_dir, intensity, f"attaque {DIR_NAMES[best_dir]} (score {scores[best_dir]:.0f})"


# On n'attaque pas tous les ticks : on laisse les troupes se reconstituer.
ATTACK_EVERY = 3
MIN_TROOPS_TO_ATTACK = 120   # garde un matelas de défense
ATTACK_INTENSITY = 0.35      # fraction des troupes envoyées (on garde 65 % chez soi)


def play_one_tick(bridge, state: Dict[str, Any]) -> List[str]:
    """Applique les ordres de l'IA pour toutes ses grappes. Retourne la liste des actions."""
    actions: List[str] = []
    tick = state.get("tick", 0)
    if tick % ATTACK_EVERY != 0:
        return actions  # tick de "pause" : on laisse les troupes remonter
    clusters = state.get("clusters", [])
    for i, cluster in enumerate(clusters[:5]):
        if cluster.get("troop_count", 0) < MIN_TROOPS_TO_ATTACK:
            continue
        direction, _intensity, desc = decide_for_cluster(state, cluster)
        if direction == 8:
            continue
        ok = bridge.attack(direction=direction, intensity=ATTACK_INTENSITY, cluster_id=i)
        if ok:
            actions.append(f"grappe {i}: {desc} ({int(ATTACK_INTENSITY*100)}% troupes)")
    return actions
