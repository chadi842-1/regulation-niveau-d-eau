import numpy as np

from src.controllers import PIDController
from src.evaluation import metrics, rollout
from src.tank_env import (Scenario, TankEnv, TankParams, equilibrium_command,
                          make_scenarios, tank_step)

P = TankParams()


def test_equilibrium_is_fixed_point():
    h_ref = 1.0
    u = equilibrium_command(h_ref, 0.0, P)
    h = h_ref
    for _ in range(50):
        h = tank_step(h, u, 0.0, P)
    assert abs(h - h_ref) < 1e-6


def test_valve_closed_drains_and_stays_nonnegative():
    h = 1.0
    for _ in range(2000):
        h = tank_step(h, 0.0, 0.0, P)
    assert 0.0 <= h < 1.0


def test_valve_open_fills_and_is_capped():
    h = 0.2
    for _ in range(2000):
        h = tank_step(h, 1.0, 0.0, P)
    assert h <= P.h_max + 1e-9
    assert h > 1.0


def test_env_truncates_at_horizon_without_terminating():
    env = TankEnv()
    env.reset(seed=0)
    for k in range(P.horizon):
        _, _, terminated, truncated, _ = env.step([0.5])
        assert not terminated
        assert truncated == (k == P.horizon - 1)


def test_scenarios_are_reproducible():
    assert make_scenarios(5, 42) == make_scenarios(5, 42)
    assert make_scenarios(5, 42) != make_scenarios(5, 43)


def test_reasonable_pid_tracks_setpoint():
    sc = Scenario(h0=0.3, h_ref=1.0, t_dist=120, d_mag=0.003)
    run = rollout(PIDController(kp=4.0, ki=0.1, kd=0.0), sc)
    m = metrics(run)
    assert m["final_abs_err_m"] < 0.02
    assert run["h"].max() < P.h_max
