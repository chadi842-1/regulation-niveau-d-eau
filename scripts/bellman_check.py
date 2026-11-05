"""Vérifications numériques de la théorie (docs/MDP.md) sur la cuve :

 1. l'opérateur de Bellman d'évaluation T^pi est une γ-contraction (norme sup) ;
 2. son point fixe V^pi coïncide avec l'estimation Monte Carlo du retour actualisé ;
 3. le processus M_t = sum_{i<t} γ^i r_i + γ^t V(s_t) est une martingale :
    E[M_t] = V(s_0) pour tout t.

Cadre : état s = h (1-D), perturbation de soutirage aléatoire d ~ U[0, D_MAX] à chaque pas,
consigne fixe, politique fixe linéaire saturée u = clip(u_eq + K (h_ref - h)), reward r = -|h_ref - h|.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.tank_env import TankParams, equilibrium_command, tank_step  # noqa: E402

P = TankParams()
GAMMA = 0.98
H_REF = 1.0
K = 5.0
D_MAX = 0.002
D_SAMPLES = np.linspace(0.0, D_MAX, 9)   # quadrature de l'espérance (loi uniforme)
GRID = np.linspace(0.0, P.h_max, 401)
U_EQ = equilibrium_command(H_REF, 0.0, P)


def policy(h):
    return float(np.clip(U_EQ + K * (H_REF - h), 0.0, 1.0))


def reward(h):
    return -abs(H_REF - h)


def main():
    rng = np.random.default_rng(0)

    # transitions pré-calculées sur la grille pour chaque valeur de perturbation
    nxt = np.array([[tank_step(h, policy(h), d, P) for d in D_SAMPLES] for h in GRID])
    r = np.array([reward(h) for h in GRID])

    def bellman(V):
        # (T V)(h) = r(h) + γ E_d[ V(h') ] , V interpolée linéairement entre les points de grille
        return r + GAMMA * np.mean(np.interp(nxt, GRID, V), axis=1)

    # 1. contraction : ||V_{k+1} - V_k|| <= γ ||V_k - V_{k-1}||
    V = np.zeros_like(GRID)
    diffs = []
    for _ in range(2000):
        V_new = bellman(V)
        diffs.append(float(np.max(np.abs(V_new - V))))
        V = V_new
        if diffs[-1] < 1e-12:
            break
    ratios = [diffs[k + 1] / diffs[k] for k in range(len(diffs) - 1) if diffs[k] > 1e-6]
    max_ratio = float(max(ratios))
    print(f"[1] contraction : {len(diffs)} itérations, ratio max ||ΔV_(k+1)||/||ΔV_k|| = {max_ratio:.4f} "
          f"(borne théorique γ = {GAMMA})")
    assert max_ratio <= GAMMA + 1e-6

    def V_of(h):
        return float(np.interp(h, GRID, V))

    # 2. Monte Carlo du retour actualisé depuis plusieurs états initiaux
    horizon = 600  # γ^600 ~ 5e-6 : troncature négligeable
    n_paths = 4000
    starts = [0.3, 0.7, 1.0, 1.4]
    mc = {}
    for h0 in starts:
        returns = np.zeros(n_paths)
        for k in range(n_paths):
            h, g, disc = h0, 0.0, 1.0
            for _ in range(horizon):
                g += disc * reward(h)
                h = tank_step(h, policy(h), rng.uniform(0, D_MAX), P)
                disc *= GAMMA
            returns[k] = g
        mean, half = returns.mean(), 1.96 * returns.std(ddof=1) / np.sqrt(n_paths)
        mc[h0] = (float(mean), float(half), V_of(h0))
        print(f"[2] h0={h0:.1f}  Monte Carlo = {mean:8.4f} ± {half:.4f}   point fixe de Bellman = {V_of(h0):8.4f}")
        assert abs(mean - V_of(h0)) < 3 * half + 0.05

    # 3. martingale : E[M_t] = V(s_0)
    h0, T_list, n_m = 0.3, [1, 5, 20, 60, 150], 4000
    Ms = {t: np.zeros(n_m) for t in T_list}
    for k in range(n_m):
        h, g, disc = h0, 0.0, 1.0
        for t in range(1, max(T_list) + 1):
            g += disc * reward(h)
            h = tank_step(h, policy(h), rng.uniform(0, D_MAX), P)
            disc *= GAMMA
            if t in Ms:
                Ms[t][k] = g + disc * V_of(h)
    mart = {}
    for t in T_list:
        m, half = Ms[t].mean(), 1.96 * Ms[t].std(ddof=1) / np.sqrt(n_m)
        mart[t] = (float(m), float(half))
        print(f"[3] t={t:3d}  E[M_t] = {m:8.4f} ± {half:.4f}   V(s0) = {V_of(h0):8.4f}")
        assert abs(m - V_of(h0)) < 3 * half + 0.02

    Path("results").mkdir(exist_ok=True)
    Path("results/bellman_check.json").write_text(json.dumps({
        "gamma": GAMMA, "iterations": len(diffs), "max_contraction_ratio": max_ratio,
        "monte_carlo_vs_fixed_point": {str(k): {"mc_mean": v[0], "mc_ci95": v[1], "bellman": v[2]} for k, v in mc.items()},
        "martingale": {str(k): {"mean": v[0], "ci95": v[1], "V_s0": V_of(h0)} for k, v in mart.items()},
    }, indent=2))
    print("OK — résultats dans results/bellman_check.json")


if __name__ == "__main__":
    main()
