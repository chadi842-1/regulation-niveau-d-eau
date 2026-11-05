"""Entraînement de l'agent DDPG sur la cuve.

Usage : python scripts/train_ddpg.py [--episodes 300] [--seed 0]
Sorties : results/actor.npz (meilleure politique), results/training_log.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.controllers import PolicyController  # noqa: E402
from src.ddpg_numpy import DDPG, ReplayBuffer  # noqa: E402
from src.evaluation import evaluate  # noqa: E402
from src.tank_env import TankEnv, TankParams, make_scenarios  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--eval_every", type=int, default=10)
    args = ap.parse_args()

    env = TankEnv(TankParams())
    agent = DDPG(seed=args.seed)
    buf = ReplayBuffer(obs_dim=4)
    val_scenarios = make_scenarios(20, seed=7)   # validation (≠ test, graine 123)
    out = Path("results")
    out.mkdir(exist_ok=True)

    log, best_val, steps, t0 = [], np.inf, 0, time.time()
    for ep in range(args.episodes):
        sigma = max(0.03, 0.2 * (1 - ep / (0.7 * args.episodes)))   # bruit d'exploration décroissant
        obs, _ = env.reset(seed=args.seed * 100_000 + ep)
        ep_ret, done, losses = 0.0, False, []
        while not done:
            a = agent.act(obs, sigma) if steps >= args.warmup else float(env.np_random.uniform(0, 1))
            obs2, r, _, done, _ = env.step([a])
            buf.add(obs, a, r, obs2)    # pas de flag "terminal" : fin d'épisode = limite de temps
            obs, ep_ret, steps = obs2, ep_ret + r, steps + 1
            if steps >= args.warmup:
                losses.append(agent.update(buf.sample(args.batch, agent.rng)))

        row = {"episode": ep, "return": ep_ret, "sigma": sigma,
               "critic_loss": float(np.mean(losses)) if losses else None}
        if (ep + 1) % args.eval_every == 0:
            res = evaluate(PolicyController(agent.actor), val_scenarios)
            val = float(np.mean(res["iae"]) + env.p.du_penalty * np.mean(res["du_total_variation"]))
            row["val_cost"] = val
            if val < best_val:
                best_val = val
                np.savez(out / "actor.npz", **agent.actor.state())
            print(f"ep {ep+1:4d}  retour={ep_ret:8.2f}  coût val={val:6.2f}  meilleur={best_val:6.2f}"
                  f"  σ={sigma:.3f}  {time.time()-t0:5.0f}s", flush=True)
        log.append(row)

    (out / "training_log.json").write_text(json.dumps(log, indent=1))
    print(f"terminé en {time.time()-t0:.0f}s, meilleur coût de validation : {best_val:.3f}")


if __name__ == "__main__":
    main()
