"""Simulateur de cuve et environnement Gymnasium.

Modèle physique (bilan de masse, loi de Torricelli pour la vidange) :

    A * dh/dt = q_max * u - c * sqrt(h) - d(t)

    h : niveau d'eau [m]
    u : ouverture de la vanne d'entrée, dans [0, 1]
    d : débit de soutirage perturbateur [m^3/s] (échelon à un instant aléatoire)

Observation de l'agent (identique côté C embarqué, voir c/rl_controller.c) :

    obs = [ e, I, h / H_MAX, u_prev ]   avec  e = h_ref - h
                                              I = clip(integrale(e dt) / INT_SCALE, -2, 2)
"""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass(frozen=True)
class TankParams:
    area: float = 0.5        # section de la cuve A [m^2]
    c_out: float = 0.01      # coefficient de vidange [m^2.5/s]
    q_max: float = 0.02      # débit d'entrée vanne ouverte [m^3/s]
    h_max: float = 2.0       # hauteur max (débordement) [m]
    dt: float = 1.0          # période d'échantillonnage [s]
    n_sub: int = 10          # sous-pas d'Euler par période
    horizon: int = 200       # longueur d'un épisode [pas]
    int_scale: float = 50.0  # normalisation de l'intégrale d'erreur [m.s]
    du_penalty: float = 0.05  # pénalité sur les variations de commande


@dataclass(frozen=True)
class Scenario:
    h0: float        # niveau initial [m]
    h_ref: float     # consigne [m]
    t_dist: int      # pas d'application de la perturbation
    d_mag: float     # amplitude de la perturbation [m^3/s]


def make_scenarios(n: int, seed: int) -> list[Scenario]:
    """Tire n scénarios reproductibles (niveau initial, consigne, perturbation)."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        h_ref = rng.uniform(0.5, 1.5)
        h0 = rng.uniform(0.1, 1.0)
        t_dist = int(rng.integers(90, 140))
        d_mag = rng.uniform(0.0, 0.004)
        out.append(Scenario(h0=h0, h_ref=h_ref, t_dist=t_dist, d_mag=d_mag))
    return out


def tank_step(h: float, u: float, d: float, p: TankParams) -> float:
    """Avance la cuve d'une période d'échantillonnage (Euler, n_sub sous-pas)."""
    u = min(max(u, 0.0), 1.0)
    sub_dt = p.dt / p.n_sub
    for _ in range(p.n_sub):
        dh = (p.q_max * u - p.c_out * np.sqrt(max(h, 0.0)) - d) / p.area
        h = min(max(h + dh * sub_dt, 0.0), p.h_max)
    return float(h)


def equilibrium_command(h_ref: float, d: float, p: TankParams) -> float:
    """Commande qui maintient exactement h_ref (vérité terrain pour les tests)."""
    return (p.c_out * np.sqrt(h_ref) + d) / p.q_max


def make_obs(h: float, h_ref: float, integ: float, u_prev: float, p: TankParams) -> np.ndarray:
    return np.array(
        [h_ref - h, np.clip(integ / p.int_scale, -2.0, 2.0), h / p.h_max, u_prev],
        dtype=np.float32,
    )


class TankEnv(gym.Env):
    """Régulation de niveau : l'agent règle l'ouverture de la vanne d'entrée."""

    metadata = {"render_modes": []}

    def __init__(self, params: TankParams | None = None):
        self.p = params or TankParams()
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32)
        self.sc: Scenario | None = None
        self.h = self.integ = self.u_prev = 0.0
        self.t = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options and "scenario" in options:
            self.sc = options["scenario"]
        else:
            rng = self.np_random
            self.sc = Scenario(
                h0=float(rng.uniform(0.1, 1.0)),
                h_ref=float(rng.uniform(0.5, 1.5)),
                t_dist=int(rng.integers(90, 140)),
                d_mag=float(rng.uniform(0.0, 0.004)),
            )
        self.h, self.integ, self.t = self.sc.h0, 0.0, 0
        self.u_prev = float(equilibrium_command(self.sc.h0, 0.0, self.p))
        self.u_prev = min(max(self.u_prev, 0.0), 1.0)
        return make_obs(self.h, self.sc.h_ref, self.integ, self.u_prev, self.p), {}

    def step(self, action):
        u = float(np.clip(np.asarray(action).reshape(-1)[0], 0.0, 1.0))
        d = self.sc.d_mag if self.t >= self.sc.t_dist else 0.0
        self.h = tank_step(self.h, u, d, self.p)
        e = self.sc.h_ref - self.h
        self.integ += e * self.p.dt
        reward = -abs(e) - self.p.du_penalty * abs(u - self.u_prev)
        self.u_prev = u
        self.t += 1
        obs = make_obs(self.h, self.sc.h_ref, self.integ, self.u_prev, self.p)
        truncated = self.t >= self.p.horizon  # limite de temps, pas un état terminal
        return obs, float(reward), False, truncated, {"h": self.h, "e": e, "u": u}
