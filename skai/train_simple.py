"""
SkaiBOT — Entraînement simple (1 environnement, fiable sur Windows / CPU).

Utilise l'environnement léger skai/rl_env.py (observation 64×64) branché sur le
même pont que le visualiseur. Un seul environnement => pas de multiprocessing.

Usage (depuis la racine du repo) :
    py skai\train_simple.py
    py skai\train_simple.py --steps 30000 --bots 5 --map australia_100x100_nt
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

from stable_baselines3 import PPO                 # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402
from skai.rl_env import SkaiEnv                    # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20000, help="Nombre de pas d'entraînement")
    parser.add_argument("--bots", type=int, default=5, help="Nombre d'adversaires")
    parser.add_argument("--map", type=str, default="australia_100x100_nt")
    parser.add_argument("--out", type=str, default="skai/models")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    model_path = os.path.join(args.out, "skai_ppo_australia")

    print("=" * 62)
    print("  SkaiBOT — entraînement PPO (CPU, 1 environnement, obs 64x64)")
    print(f"  Carte : {args.map}  |  Adversaires : {args.bots}  |  Pas : {args.steps:,}")
    print("=" * 62)
    print("Démarrage du moteur (≈20-40 s au premier lancement)...")

    env = Monitor(SkaiEnv(map_name=args.map, num_players=args.bots + 1, verbose=True))

    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        verbose=1,
        device="cpu",
        n_steps=1024,
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.99,
        ent_coef=0.01,          # un peu d'exploration
        tensorboard_log=os.path.join(args.out, "logs"),
    )

    print("\nEntraînement en cours... (regarde 'ep_rew_mean' monter dans TensorBoard)\n")
    model.learn(total_timesteps=args.steps)

    model.save(model_path)
    print(f"\n✅ Modèle sauvegardé : {model_path}.zip")
    print("Prochaine étape : brancher ce modèle dans le visualiseur pour le regarder jouer.")


if __name__ == "__main__":
    main()
