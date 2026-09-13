"""Measure local sensitivity and the legacy lock endpoint discontinuity."""

import argparse

import mujoco
import numpy as np

from gcartpole.config import dump_json, load_config
from gcartpole.env import NLinkCartPoleEnv
from gcartpole.evidence import file_metadata, runtime_metadata, utc_timestamp
from make_lqr_checkpoint import finite_difference_dynamics
from search_swingup_capture import lqr_gain


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="runs/transfer_review_numerics.json")
    args = parser.parse_args()
    linear = []
    for n in (7, 8):
        path = f"configs/swingup{n}_uniform.yaml"
        cfg = load_config(path)
        a, b = finite_difference_dynamics(cfg, 1.0, 1e-7)
        gain = lqr_gain(cfg, progress=1.0, fd_eps=1e-7, control_cost=1000.0)
        linear.append({
            "n_links": n, "config": file_metadata(path),
            "closed_loop_spectral_radius": float(np.max(np.abs(np.linalg.eigvals(a - b @ gain[None, :])))),
            "gain_norm": float(np.linalg.norm(gain)),
            "maximum_gain": float(np.max(np.abs(gain))),
        })
    path = "configs/swingup8_split_unlock_continuation.yaml"
    cfg = load_config(path)
    endpoint = []
    for progress in (0.8, 0.9, 0.99, 0.999999, 1.0):
        env = NLinkCartPoleEnv(cfg, progress=progress)
        try:
            mujoco.mj_resetData(env.model, env.data)
            env.data.qpos[-1] = 0.01
            env.data.qvel[-1] = 0.01
            mujoco.mj_forward(env.model, env.data)
            endpoint.append({
                "progress": progress,
                "equality_count": int(env.model.neq),
                "last_joint_acceleration": float(env.data.qacc[-1]),
                "constraint_force_norm": float(np.linalg.norm(env.data.qfrc_constraint)),
            })
        finally:
            env.close()
    dump_json({
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "claim_status": "development_diagnostic",
        "linearization": {"fd_epsilon": 1e-7, "control_cost": 1000.0, "results": linear},
        "lock_endpoint": {"config": file_metadata(path), "last_relative_angle": 0.01, "last_relative_rate": 0.01, "results": endpoint},
        "runtime": runtime_metadata(),
    }, args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
