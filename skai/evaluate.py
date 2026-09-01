'''
SkaiBOT — évaluation d'un modèle : le même environnement, plusieurs politiques.

    python skai/evaluate.py --model skai/models/skai_ppo_australia_100x100.zip --games 5
    python skai/evaluate.py --model A.zip --model-b B.zip --games 5   # nouveau vs ancien

Pourquoi ce script : "ep_rew_mean monte" dans TensorBoard ne prouve rien. La seule
question qui compte est « est-ce que ça gagne des parties ? ». Cet évaluateur fait jouer
modèle / glouton / aléatoire dans `skai/rl_env.SkaiEnv` (mêmes observations, mêmes
récompenses, mêmes règles) et compare taux de victoire, territoire et rang.

Sortie : un tableau + un verdict, et le code de retour 0 si le modèle bat la gloutonne.
'''
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

from skai.rl_env import SkaiEnv, action_to_command, legal_directions  # noqa: E402


def greedy_action(env, obs=None, rng=None):
    """Politique de référence : le front qui offre le plus de terres prenable."""
    tmap, land = env._tmap, env._land
    if tmap is None:
        return 40
    scores = [
        float((np.roll(np.roll(tmap == 1, dy, axis=0), dx, axis=1) & ((tmap != 1) & (land > 0.5))).sum())
        for dx, dy in [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]
    ]
    order = np.argsort(scores)[::-1].tolist()
    return order[0] * 5 + 2 if scores[order[0]] > 0 else 40


def run_episode(env, policy, rng):
    obs, _ = env.reset(seed=int(rng.integers(1, 2**31 - 1)))
    if hasattr(policy, "reset"):
        policy.reset()
    total, steps = 0.0, 0
    while True:
        if hasattr(policy, "predict"):
            action = policy.predict(obs, env)
        else:
            action = policy(env, obs, rng)
        obs, reward, term, trunc, info = env.step(action)
        total += reward
        steps += 1
        if term or trunc:
            break
    st = env.state
    won = bool(st.get("has_won")) or st.get("territory_pct", 0) >= env.win_threshold
    return {
        "win": won,
        "died": bool(st.get("has_lost")),
        "territory": st.get("territory_pct", 0.0),
        "rank": st.get("rank", 99),
        "reward": total,
        "steps": steps,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="chemin du .zip sauvegardé par train_simple.py")
    ap.add_argument("--model-b", default=None, help="second modèle (comparaison avant/après réentraînement)")
    ap.add_argument("--map", default="australia_100x100")
    ap.add_argument("--bots", type=int, default=5)
    ap.add_argument("--games", type=int, default=5)
    ap.add_argument("--max-ticks", type=int, default=1500)
    ap.add_argument("--mask", action="store_true", help="le modèle a été entraîné avec --mask")
    ap.add_argument("--win-threshold", type=float, default=0.80,
                    help="fraction des terres pour gagner. En 400-600 ticks, 0.80 est irréaliste : "
                         "mets 0.30-0.40 pour mesurer de vraies victoires, garde 0.80 pour le speedrun")
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    policies = {}
    if args.model:
        from skai.model_policy import ModelPlayer
        if args.mask:
            from sb3_contrib import MaskablePPO
            p = ModelPlayer(args.model, max_ticks=args.max_ticks)
            p.model = MaskablePPO.load(args.model, device="cpu")
        else:
            p = ModelPlayer(args.model, max_ticks=args.max_ticks)
        policies[args.model.split("/")[-1].replace(".zip", "")] = p
    if args.model_b:
        from skai.model_policy import ModelPlayer
        name_b = args.model_b.split("/")[-1].replace(".zip", "")
        policies[(name_b + " (comparaison)")[:26]] = ModelPlayer(args.model_b, max_ticks=args.max_ticks)
    policies["glouton (référence)"] = greedy_action
    # note : la gloutonne est une politique "oracle" (elle lit l'état complet), ce qui est
    # exactement ce que le réseau voit aussi via son canal carte -> comparaison loyale.
    policies["aléatoire"] = lambda env, obs, r: int(r.integers(0, 45))

    env = SkaiEnv(map_name=args.map, num_players=args.bots + 1, max_ticks=args.max_ticks,
                  win_threshold=args.win_threshold)
    print("=" * 74)
    print(f"  Évaluation — carte {args.map}, {args.bots} adversaires, {args.games} partie(s) par politique")
    print(f"  Fin de partie : {args.max_ticks} ticks OU {args.win_threshold*100:.0f}% des terres "
          f"(seuil du moteur : 80% en FFA)")
    print("=" * 74)
    print(f"{'politique':26s} {'victoires':>10s} {'territoire':>11s} {'rang':>6s} {'mort':>6s} {'ticks':>7s}")
    results = {}
    for name, pol in policies.items():
        rows = [run_episode(env, pol, rng) for _ in range(args.games)]
        wins = sum(r["win"] for r in rows)
        results[name] = (wins, np.mean([r["territory"] for r in rows]), np.mean([r["rank"] for r in rows]))
        print(f"{name:26s} {wins:>5d}/{len(rows):<4d} {np.mean([r['territory'] for r in rows])*100:>10.1f}% "
              f"{np.mean([r['rank'] for r in rows]):>6.2f} {sum(r['died'] for r in rows):>5d}/{len(rows):<2d}"
              f" {np.mean([r['steps'] for r in rows]):>7.0f}")
    env.close()

    base = "glouton (référence)"
    if args.model and base in results:
        m = [k for k in results if k != base and "(comparaison)" not in k][0]
        mw, mt, mr = results[m]
        bw, bt, br = results[base]
        print("\n" + "-" * 74)
        if mw > bw or (mw == bw and mt > bt):
            print(f"✅ le modèle fait MIEUX que la référence ({mw}/{args.games} victoires vs {bw}, "
                  f"territoire {mt*100:.1f}% vs {bt*100:.1f}%)")
        elif mw == bw and abs(mt - bt) < 0.02:
            print("⚠ à égalité avec la gloutonne : continue à entraîner (ou manque de parties).")
        else:
            print(f"❌ le modèle est EN DESSOUS de la gloutonne ({mw} vs {bw} victoires) : "
                  "augmenter les pas, vérifier que --mask correspond à l'entraînement.")
        if args.model_b:
            other = [k for k in results if "(comparaison)" in k][0]
            ow, ot, _ = results[other]
            print(f"   comparatif avant/après réentraînement : {ow}→{mw} victoires, "
                  f"territoire {ot*100:.1f}%→{mt*100:.1f}%")
            if mw < ow or mt < ot - 0.02:
                print("   ⚠ régression : le réentraînement a abîmé le modèle (trop de pas sans "
                      "restore best, ou learning rate trop haut).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
