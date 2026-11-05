"""Exporte la politique entraînée vers du C embarqué.

Génère :
  c/policy_weights.h   poids de l'acteur (float, const) + constantes d'observation et de la cuve
  c/test_vectors.csv   couples (observation, action attendue) calculés en float32 par numpy

Usage : python scripts/export_c.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ddpg_numpy import MLP  # noqa: E402
from src.tank_env import TankParams  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def c_array(name: str, a: np.ndarray) -> str:
    flat = np.asarray(a, dtype=np.float32).reshape(-1)
    body = ",\n    ".join(", ".join(f"{v:.9e}f" for v in flat[i:i + 4]) for i in range(0, len(flat), 4))
    return f"static const float {name}[{flat.size}] = {{\n    {body}\n}};\n"


def main():
    d = np.load(ROOT / "results" / "actor.npz")
    actor = MLP([4, 64, 64, 1], True, np.random.default_rng(0), lr=0.0)
    actor.load_state(d)
    p = TankParams()

    n_params = sum(w.size + b.size for w, b in zip(actor.W, actor.b))
    header = [
        "/* FICHIER GÉNÉRÉ par scripts/export_c.py — ne pas modifier à la main. */",
        "#ifndef POLICY_WEIGHTS_H",
        "#define POLICY_WEIGHTS_H",
        "",
        f"#define RL_OBS_DIM {actor.W[0].shape[0]}",
        f"#define RL_H1 {actor.W[0].shape[1]}",
        f"#define RL_H2 {actor.W[1].shape[1]}",
        f"#define RL_DT {p.dt}f",
        f"#define RL_INT_SCALE {p.int_scale}f",
        f"#define RL_H_MAX {p.h_max}f",
        "",
        "/* paramètres de la cuve, utilisés uniquement par le banc de test sur PC */",
        f"#define PLANT_AREA {p.area}",
        f"#define PLANT_C_OUT {p.c_out}",
        f"#define PLANT_Q_MAX {p.q_max}",
        f"#define PLANT_H_MAX {p.h_max}",
        f"#define PLANT_DT {p.dt}",
        f"#define PLANT_N_SUB {p.n_sub}",
        f"#define PLANT_HORIZON {p.horizon}",
        "",
        f"/* {n_params} paramètres float = {4 * n_params} octets de flash */",
        c_array("RL_W0", actor.W[0]),
        c_array("RL_B0", actor.b[0]),
        c_array("RL_W1", actor.W[1]),
        c_array("RL_B1", actor.b[1]),
        c_array("RL_W2", actor.W[2]),
        c_array("RL_B2", actor.b[2]),
        "#endif",
        "",
    ]
    (ROOT / "c" / "policy_weights.h").write_text("\n".join(header))

    rng = np.random.default_rng(0)
    obs = np.column_stack([
        rng.uniform(-1.5, 1.5, 300),
        rng.uniform(-2.0, 2.0, 300),
        rng.uniform(0.0, 1.0, 300),
        rng.uniform(0.0, 1.0, 300),
    ]).astype(np.float32)
    expected = actor.forward(obs)[:, 0]
    rows = [",".join(f"{v:.9g}" for v in (*o, e)) for o, e in zip(obs, expected)]
    (ROOT / "c" / "test_vectors.csv").write_text("\n".join(rows) + "\n")
    print(f"c/policy_weights.h écrit ({n_params} paramètres, {4 * n_params} octets) et 300 vecteurs de test")


if __name__ == "__main__":
    main()
