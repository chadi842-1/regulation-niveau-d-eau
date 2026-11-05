"""Adaptateurs : donnent la même interface act(h, h_ref, obs) au PID et à l'agent RL."""

from __future__ import annotations

import numpy as np

from .pid import PID


class PIDController:
    name = "PID"

    def __init__(self, kp: float, ki: float, kd: float = 0.0, dt: float = 1.0):
        self.pid = PID(kp=kp, ki=ki, kd=kd, dt=dt)

    def reset(self):
        self.pid.reset()

    def act(self, h: float, h_ref: float, obs) -> float:
        return self.pid.step(h_ref, h)


class PolicyController:
    """Politique déterministe a = mu(obs) (agent DDPG entraîné)."""

    name = "DDPG"

    def __init__(self, actor):
        self.actor = actor

    def reset(self):
        pass

    def act(self, h: float, h_ref: float, obs) -> float:
        return float(self.actor.forward(np.asarray(obs, dtype=np.float32)[None, :])[0, 0])
