"""Boucle de simulation commune et métriques de comparaison PID / DDPG."""

from __future__ import annotations

from typing import Callable

import numpy as np

from .tank_env import Scenario, TankEnv, TankParams


def rollout(controller: Callable, sc: Scenario, params: TankParams | None = None) -> dict:
    """Simule un scénario.

    `controller` expose reset() et act(h, h_ref, obs) -> u dans [0, 1].
    """
    env = TankEnv(params)
    obs, _ = env.reset(options={"scenario": sc})
    controller.reset()
    h = [sc.h0]
    u = []
    done = False
    while not done:
        a = controller.act(env.h, sc.h_ref, obs)
        obs, _, _, done, info = env.step([a])
        h.append(info["h"])
        u.append(info["u"])
    return {"h": np.array(h), "u": np.array(u), "h_ref": sc.h_ref, "t_dist": sc.t_dist}


def metrics(run: dict, dt: float = 1.0) -> dict:
    h, u, ref, t_d = run["h"][1:], run["u"], run["h_ref"], run["t_dist"]
    e = ref - h
    step_amp = abs(ref - run["h"][0])

    # phase 1 : suivi de consigne, avant la perturbation
    h1, e1 = h[:t_d], e[:t_d]
    overshoot = float(max(0.0, np.max(np.sign(ref - run["h"][0]) * (h1 - ref)))) if step_amp > 1e-9 else 0.0
    band = 0.02 * max(step_amp, 1e-9)
    outside = np.where(np.abs(e1) > band)[0]
    settling = float((outside[-1] + 1) * dt) if len(outside) else 0.0
    if len(outside) and outside[-1] == len(e1) - 1:
        settling = float("nan")  # n'a pas convergé avant la perturbation

    return {
        "iae": float(np.sum(np.abs(e)) * dt),
        "ise": float(np.sum(e**2) * dt),
        "overshoot_m": overshoot,
        "settling_s": settling,
        "final_abs_err_m": float(np.mean(np.abs(e[-20:]))),
        "du_total_variation": float(np.sum(np.abs(np.diff(u)))),
    }


def evaluate(controller, scenarios: list[Scenario], params: TankParams | None = None) -> dict:
    rows = [metrics(rollout(controller, sc, params)) for sc in scenarios]
    keys = rows[0].keys()
    return {k: [r[k] for r in rows] for k in keys}


def summarize(per_scenario: dict) -> dict:
    out = {}
    for k, v in per_scenario.items():
        arr = np.array(v, dtype=float)
        out[k] = {
            "mean": float(np.nanmean(arr)),
            "std": float(np.nanstd(arr)),
            "n_nan": int(np.isnan(arr).sum()),
        }
    return out
