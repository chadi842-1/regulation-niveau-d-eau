"""Régulateur PID discret avec anti-windup (intégration conditionnelle)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PID:
    kp: float
    ki: float
    kd: float = 0.0
    dt: float = 1.0
    u_min: float = 0.0
    u_max: float = 1.0
    d_filter: float = 0.8   # filtre du dérivé sur la mesure (0 = pas de filtre)

    def __post_init__(self):
        self.reset()

    def reset(self):
        self.integ = 0.0
        self.prev_h = None
        self.d_state = 0.0

    def step(self, h_ref: float, h: float) -> float:
        e = h_ref - h
        # dérivée sur la mesure (évite le coup de dérivée sur échelon de consigne)
        if self.prev_h is None:
            d_meas = 0.0
        else:
            raw = -(h - self.prev_h) / self.dt
            d_meas = self.d_filter * self.d_state + (1 - self.d_filter) * raw
        self.prev_h, self.d_state = h, d_meas

        u_unsat = self.kp * e + self.ki * self.integ + self.kd * d_meas
        u = min(max(u_unsat, self.u_min), self.u_max)
        # anti-windup : on n'intègre pas quand la commande sature dans le sens de l'erreur
        saturated_hi = u_unsat > self.u_max and e > 0
        saturated_lo = u_unsat < self.u_min and e < 0
        if not (saturated_hi or saturated_lo):
            self.integ += e * self.dt
        return u
