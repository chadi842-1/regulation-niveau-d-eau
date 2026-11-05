"""Comparaison quantitative PID vs DDPG sur 100 scénarios de test (graine 123, jamais vus).

Sorties : results/results.json, results/results.md, results/*.png
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.controllers import PIDController, PolicyController  # noqa: E402
from src.ddpg_numpy import MLP  # noqa: E402
from src.evaluation import evaluate, rollout, summarize  # noqa: E402
from src.tank_env import make_scenarios  # noqa: E402

OUT = Path("results")


def load_actor():
    d = np.load(OUT / "actor.npz")
    actor = MLP([4, 64, 64, 1], True, np.random.default_rng(0), lr=0.0)
    actor.load_state(d)
    return actor


def main():
    gains = json.loads((OUT / "pid_gains.json").read_text())
    pid = PIDController(gains["kp"], gains["ki"], gains["kd"])
    ddpg = PolicyController(load_actor())
    test = make_scenarios(100, seed=123)

    per = {"PID": evaluate(pid, test), "DDPG": evaluate(ddpg, test)}
    summary = {k: summarize(v) for k, v in per.items()}

    # victoires par scénario (IAE plus faible)
    iae_pid, iae_rl = np.array(per["PID"]["iae"]), np.array(per["DDPG"]["iae"])
    wins = {"DDPG_better_iae": int(np.sum(iae_rl < iae_pid)), "n": len(test)}

    labels = {
        "iae": "IAE [m·s]",
        "ise": "ISE [m²·s]",
        "overshoot_m": "Dépassement [m]",
        "settling_s": "Temps de réponse 2 % [s]",
        "final_abs_err_m": "Erreur finale |e| [m]",
        "du_total_variation": "Variation totale de u",
    }
    lines = ["| Métrique | PID | DDPG |", "|---|---|---|"]
    for k, lab in labels.items():
        a, b = summary["PID"][k], summary["DDPG"][k]
        lines.append(f"| {lab} | {a['mean']:.4f} ± {a['std']:.4f} | {b['mean']:.4f} ± {b['std']:.4f} |")
    lines.append("")
    lines.append(f"DDPG a un IAE plus faible que le PID sur {wins['DDPG_better_iae']}/{wins['n']} scénarios de test.")
    (OUT / "results.md").write_text("\n".join(lines) + "\n")
    (OUT / "results.json").write_text(json.dumps({"pid_gains": gains, "summary": summary, "wins": wins}, indent=2))
    print("\n".join(lines))

    # --- figure : 3 scénarios (médian, meilleur et pire cas pour DDPG relativement au PID) ---
    ratio = iae_rl / iae_pid
    order = np.argsort(ratio)
    picks = [("meilleur cas DDPG", order[0]), ("cas médian", order[len(order) // 2]), ("pire cas DDPG", order[-1])]
    fig, axes = plt.subplots(2, 3, figsize=(13, 6), sharex=True)
    for j, (title, idx) in enumerate(picks):
        sc = test[idx]
        for ctrl, color in ((pid, "tab:blue"), (ddpg, "tab:red")):
            run = rollout(ctrl, sc)
            axes[0, j].plot(run["h"], color=color, label=ctrl.name)
            axes[1, j].step(range(len(run["u"])), run["u"], color=color)
        axes[0, j].axhline(sc.h_ref, color="k", ls="--", lw=0.8, label="consigne")
        axes[0, j].axvline(sc.t_dist, color="gray", ls=":", lw=0.8)
        axes[0, j].set_title(f"{title} (scénario {idx})")
        axes[1, j].set_xlabel("temps [s]")
    axes[0, 0].set_ylabel("niveau h [m]")
    axes[1, 0].set_ylabel("ouverture vanne u")
    axes[0, 0].legend()
    fig.suptitle("PID vs DDPG — pointillé gris : début de la perturbation de soutirage")
    fig.tight_layout()
    fig.savefig(OUT / "comparaison_scenarios.png", dpi=130)

    # --- figure : courbe d'apprentissage ---
    log = json.loads((OUT / "training_log.json").read_text())
    ep = [r["episode"] for r in log]
    ret = np.array([r["return"] for r in log])
    k = 20
    smooth = np.convolve(ret, np.ones(k) / k, mode="valid")
    val = [(r["episode"], r["val_cost"]) for r in log if "val_cost" in r]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ep, ret, alpha=0.3)
    ax[0].plot(ep[k - 1:], smooth, color="tab:red")
    ax[0].set_xlabel("épisode")
    ax[0].set_ylabel("retour d'entraînement")
    ax[0].set_title("Retour par épisode (moyenne glissante 20)")
    ax[1].plot(*zip(*val), marker="o", ms=3)
    # le PID est évalué sur les MÊMES scénarios de validation (graine 7) que la courbe DDPG
    pid_val = evaluate(pid, make_scenarios(20, seed=7))
    pid_val_cost = np.mean(pid_val["iae"]) + 0.05 * np.mean(pid_val["du_total_variation"])
    ax[1].axhline(pid_val_cost, color="tab:blue", ls="--", label="PID (mêmes scénarios de validation)")
    ax[1].set_xlabel("épisode")
    ax[1].set_ylabel("coût de validation (IAE + pénalité)")
    ax[1].set_title("Validation de la politique déterministe")
    ax[1].set_ylim(bottom=3.5)
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(OUT / "apprentissage.png", dpi=130)


if __name__ == "__main__":
    main()
