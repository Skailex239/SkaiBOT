"""
SkaiBOT — Entraînement simple (1 environnement, fiable sur Windows / CPU).

Utilise l'environnement léger skai/rl_env.py branché sur le même pont que le
visualiseur. Un seul environnement => pas de multiprocessing.

Usage (depuis la racine du repo) :
    py skai\\train_simple.py
    py skai\\train_simple.py --steps 30000 --bots 5 --map australia_100x100_nt
    py skai\\train_simple.py --map australia_500x500_nt --obs 64 --mask

AVANT de lancer un entraînement long : `py skai\\diagnose.py --fast` (5 min).
Si le verdict est « ❌ entraînement inutile », aucun nombre de pas n'y changera rien :
c'est l'environnement qu'il faut corriger (voir docs/DIAGNOSTIC.md).
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

from stable_baselines3 import PPO                     # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402
from skai.rl_env import SkaiEnv                        # noqa: E402


class ProgressCallback(BaseCallback):
    """Affiche territoire moyen / rang toutes les N mises à jour, pour voir bouger
    ce qui compte (ep_rew_mean seul est trompeur quand la récompense est une dérive)."""

    def __init__(self, every=5, verbose=1, best_path=None, model=None):
        super().__init__(verbose)
        self.every = every
        self.ep = []
        self.best_path = best_path
        self.best_model = model
        self.best_pct = -1.0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos") or [{}]
        for info in infos:
            if isinstance(info, dict) and "territory_pct" in info:
                self.ep.append(info)
        if self.n_calls % (self.every * 512) < 1 and self.verbose > 0 and self.ep:
            pct = sum(e["territory_pct"] for e in self.ep[-50:]) / len(self.ep[-50:])
            rank = sum(e["rank"] for e in self.ep[-50:]) / len(self.ep[-50:])
            print(f"   [{self.n_calls:>7d} pas] territoire moyen récent {pct*100:5.1f}%  rang moyen {rank:4.1f}")
            if self.best_path and self.best_model is not None and pct > self.best_pct:
                self.best_pct = pct
                self.best_model.save(self.best_path)
                print(f"      ↑ nouveau record ({pct*100:.1f}%) → {os.path.basename(self.best_path)}")
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20000, help="Nombre de pas d'entraînement")
    parser.add_argument("--bots", type=int, default=5, help="Nombre d'adversaires")
    parser.add_argument("--map", type=str, default="australia_100x100_nt")
    parser.add_argument("--out", type=str, default="skai/models")
    parser.add_argument("--max-ticks", type=int, default=1500, help="Longueur max d'un épisode")
    parser.add_argument("--win-threshold", type=float, default=0.80, help="Fraction des TERRES pour gagner")
    parser.add_argument("--game-config", type=str, default="default", choices=["default", "test"],
                        help="'default' = vraies règles OpenFront ; 'test' = ancien bridage (1 tuile/tick)")
    parser.add_argument("--n-steps", type=int, default=512, help="Pas collectés par environnement par mise à jour")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--ent-coef", type=float, default=0.02)
    parser.add_argument("--living-reward", type=float, default=0.0,
                        help="bonus/malus par tick (négatif = pression temporelle, utile en speedrun)")
    parser.add_argument("--n-envs", type=int, default=1,
                        help="Environnements en parallèle (sous-processus). xN sur le debit, "
                             "donc xN d'episodes vus : c'est LE levier principal de vitesse.")
    parser.add_argument("--mask", action="store_true", help="MaskablePPO (sb3-contrib) : n' propose que des directions légales")
    parser.add_argument("--resume", type=str, default=None,
                        help="reprendre un .zip existant (réentraînement progressif) au lieu de repartir de zéro")
    parser.add_argument("--save-best", action="store_true",
                        help="sauvegarde aussi <model>_best.zip dès que le territoire moyen bat le meilleur connu")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    tag = os.path.basename(args.map)
    model_path = os.path.join(args.out, f"skai_ppo_{tag}")

    print("=" * 62)
    print("  SkaiBOT — entraînement PPO (CPU, 1 environnement, obs 64x64)")
    print(f"  Carte : {args.map}  |  Adversaires : {args.bots}  |  Pas : {args.steps:,}")
    print(f"  Config moteur : {args.game_config}  |  Masque d'actions : {args.mask}")
    print("=" * 62)
    print("Démarrage du moteur (≈20-40 s au premier lancement)...")

    def make_env(rank: int):
        """Fabrique un environnement.

        Pattern OBLIGATOIRE sous Windows (démarrage 'spawn') : une fonction de niveau
        module + un `if __name__ == "__main__"`, sinon les sous-processus ré-exécutent
        le script et bouclent. Ici le `rank` ne change que la graine, pour que les
        environnements ne jouent pas la même partie.
        """
        def _init():
            env = SkaiEnv(
                map_name=args.map,
                num_players=args.bots + 1,
                max_ticks=args.max_ticks,
                win_threshold=args.win_threshold,
                game_config=args.game_config,
                living_reward=args.living_reward,
                verbose=(rank == 0),
            )
            return Monitor(env)
        return _init

    if args.n_envs > 1:
        from stable_baselines3.common.vec_env import SubprocVecEnv
        env = SubprocVecEnv([make_env(i) for i in range(args.n_envs)])
        print(f"  {args.n_envs} environnements en sous-processus (débit x{args.n_envs} visé)")
    else:
        env = make_env(0)()

    Algo = PPO
    if args.mask:
        try:
            from sb3_contrib import MaskablePPO as Algo  # noqa: N813
        except Exception as e:  # pragma: no cover
            print(f"⚠ sb3-contrib indisponible ({e}) — installation : pip install sb3-contrib")
            print("  → entraînement en PPO classique (les actions illégales sont ignorées par le pont).")

    tb_log = os.path.join(args.out, "logs")
    if args.resume:
        if not os.path.exists(args.resume):
            raise SystemExit(f"❌ --resume : fichier introuvable : {args.resume}")
        model = Algo.load(args.resume, env=env, tensorboard_log=tb_log, device="cpu")
        print(f"\n↻ Reprise depuis {args.resume} (les weights + l'optimizer sont chargés).")
        print("  ⚠ la phase d'exploration (ent_coef) repart du même réglage : si le policy gradient")
        print("    s'effondre, baisse --lr (ex. 1e-4) plutôt que de tout recommencer.")
    else:
        model = Algo(
            policy="MultiInputPolicy",
            env=env,
            verbose=1,
            device="cpu",
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=args.ent_coef,          # un peu d'exploration
            tensorboard_log=tb_log,
        )

    print("\nEntraînement en cours... (regarde 'ep_rew_mean' monter dans TensorBoard)\n")
    best = os.path.join(args.out, f"skai_ppo_{tag}_best.zip") if args.save_best else None
    model.learn(total_timesteps=args.steps,
                reset_num_timesteps=not bool(args.resume),
                callback=ProgressCallback(every=2, best_path=best, model=model))

    model.save(model_path)
    print(f"\n✅ Modèle sauvegardé : {model_path}.zip" + (f"  (record: {os.path.basename(best)})" if best else ""))
    print("\n--- Boucle complète (à copier) ---")
    print(f"  python skai/evaluate.py --model {model_path}.zip --map {args.map} --games 5"
          + (" --mask" if args.mask else ""))
    print(f"  python skai/viewer/server.py --map {args.map} --model {model_path}.zip   # le voir jouer dans le navigateur")
    print(f"  python skai/train_simple.py --resume {model_path}.zip --steps {args.steps*3} --map {args.map}"
          + (" --mask" if args.mask else "") + "   # réentraîner plus longtemps")


if __name__ == "__main__":
    main()
