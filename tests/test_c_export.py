"""Vérifie que le C embarqué reproduit la politique Python (point par point et en boucle fermée)."""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from src.controllers import PolicyController
from src.ddpg_numpy import MLP
from src.evaluation import rollout
from src.tank_env import equilibrium_command, make_scenarios, TankParams

ROOT = Path(__file__).resolve().parent.parent
C_DIR = ROOT / "c"

pytestmark = pytest.mark.skipif(
    shutil.which("gcc") is None or not (C_DIR / "policy_weights.h").exists(),
    reason="gcc ou c/policy_weights.h absent (lancer scripts/export_c.py)",
)


@pytest.fixture(scope="module")
def binary():
    subprocess.run(["make", "-C", str(C_DIR), "build/test_policy"], check=True, capture_output=True)
    return C_DIR / "build" / "test_policy"


def test_policy_matches_numpy_on_random_observations(binary):
    out = subprocess.run([str(binary), "vectors", str(C_DIR / "test_vectors.csv")],
                         check=True, capture_output=True, text=True).stdout
    max_err = float(out.split("max_abs_err=")[1])
    assert max_err < 1e-5


def test_closed_loop_matches_python(binary):
    actor = MLP([4, 64, 64, 1], True, np.random.default_rng(0), lr=0.0)
    actor.load_state(np.load(ROOT / "results" / "actor.npz"))
    ctrl = PolicyController(actor)
    p = TankParams()
    for sc in make_scenarios(5, seed=99):
        u0 = min(max(equilibrium_command(sc.h0, 0.0, p), 0.0), 1.0)
        out = subprocess.run(
            [str(binary), "sim", str(sc.h0), str(sc.h_ref), str(sc.t_dist), str(sc.d_mag), str(u0)],
            check=True, capture_output=True, text=True).stdout.split()
        h_c = np.array([float(v) for v in out])
        h_py = rollout(ctrl, sc)["h"][1:]
        assert np.max(np.abs(h_c - h_py)) < 1e-3
