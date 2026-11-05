"""Réglage du PID par recherche aléatoire sur des scénarios d'apprentissage.

Le PID est réglé sur les mêmes scénarios (graine 0) que ceux vus par l'agent DDPG,
puis les deux sont comparés sur des scénarios de test jamais vus (graine 123).
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.controllers import PIDController  # noqa: E402
from src.evaluation import evaluate  # noqa: E402
from src.tank_env import TankParams, make_scenarios  # noqa: E402


def cost(gains, scenarios):
    kp, ki, kd = gains
    res = evaluate(PIDController(kp, ki, kd), scenarios)
    # même critère que le reward de l'agent RL : IAE + du_penalty * variation totale de u
    return float(np.mean(np.array(res["iae"]) + TankParams().du_penalty * np.array(res["du_total_variation"])))


def main():
    rng = np.random.default_rng(0)
    scenarios = make_scenarios(30, seed=0)

    best, best_c = (2.0, 0.05, 0.0), cost((2.0, 0.05, 0.0), scenarios)
    print(f"départ  kp={best[0]:.3f} ki={best[1]:.4f} kd={best[2]:.3f}  IAE={best_c:.2f}")

    for it in range(400):
        scale = 1.0 if it < 200 else 0.2
        cand = (
            float(np.clip(best[0] * np.exp(rng.normal(0, 0.3 * scale)), 0.1, 50)),
            float(np.clip(best[1] * np.exp(rng.normal(0, 0.3 * scale)), 0.0005, 50)),
            float(np.clip(best[2] + rng.normal(0, 2.0 * scale), 0.0, 50)),
        )
        c = cost(cand, scenarios)
        if c < best_c:
            best, best_c = cand, c
            print(f"it {it:3d}  kp={best[0]:.3f} ki={best[1]:.4f} kd={best[2]:.3f}  IAE={best_c:.2f}")

    out = {"kp": best[0], "ki": best[1], "kd": best[2], "train_cost": best_c}
    Path("results").mkdir(exist_ok=True)
    Path("results/pid_gains.json").write_text(json.dumps(out, indent=2))
    print("gains retenus :", out)


if __name__ == "__main__":
    main()
