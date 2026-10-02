#!/usr/bin/env python
"""Cross-check upright transition derivatives against linearized mechanics."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.linalg import solve_discrete_are

from audit_link_count_frontier import ROOT, WEIGHTS, exact_capture
from gcartpole.config import load_config
from gcartpole.env import NLinkCartPoleEnv
from gcartpole.evidence import file_metadata, runtime_metadata
from gcartpole.generalized_modes import chain_normal_modes
from make_lqr_checkpoint import absolute_angle_cost, finite_difference_dynamics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    path = Path(args.out)
    if path.exists():
        raise FileExistsError(path)
    base = load_config(ROOT / 'configs/swingup11_uniform.yaml')
    records = []
    for n in range(10, 16):
        cfg = copy.deepcopy(base)
        cfg['env']['n_links'] = n
        env = NLinkCartPoleEnv(cfg, progress=1, seed=0)
        env.data.qpos[:] = 0
        env.data.qvel[:] = 0
        mujoco.mj_forward(env.model, env.data)
        d = n + 1
        mass = np.zeros((d, d))
        mujoco.mj_fullM(env.model, mass, env.data.qM)
        modes = chain_normal_modes(env, equilibrium='upright')
        stiffness = np.zeros((d, d))
        stiffness[1:, 1:] = modes.stiffness_matrix
        damping = np.diag(env.model.dof_damping)
        continuous = np.block([[np.zeros((d, d)), np.eye(d)],
                               [-np.linalg.solve(mass, stiffness),
                                -np.linalg.solve(mass, damping)]])
        input_column = np.r_[np.zeros(d), np.linalg.solve(
            mass, np.r_[env.force_limit, np.zeros(n)])][:, None]
        augmented = np.zeros((2*d+1, 2*d+1))
        augmented[:2*d, :2*d] = continuous
        augmented[:2*d, -1:] = input_column
        z = augmented * env.model.opt.timestep
        one_step = np.eye(len(z)) + z + z@z/2 + z@z@z/6 + z@z@z@z/24
        discrete = np.linalg.matrix_power(one_step, env.frame_skip)
        a, b = discrete[:2*d, :2*d], discrete[:2*d, -1:]
        fd_a, fd_b = finite_difference_dynamics(cfg, 1, 1e-7)
        record = dict(n_links=n,
                      relative_a_change=float(np.linalg.norm(a-fd_a)/np.linalg.norm(fd_a)),
                      relative_b_change=float(np.linalg.norm(b-fd_b)/np.linalg.norm(fd_b)))
        try:
            p = solve_discrete_are(a, b, absolute_angle_cost(n, WEIGHTS), np.array([[1000.]]))
            gain = np.linalg.solve(1000+b.T@p@b, b.T@p@a).ravel()
            record.update(gain_norm=float(np.linalg.norm(gain)),
                          spectral_radius=float(np.max(np.abs(np.linalg.eigvals(a-b@gain[None])))),
                          spectral_radius_on_fd=float(np.max(np.abs(np.linalg.eigvals(fd_a-fd_b@gain[None])))))
            if n <= 12:
                directions = np.random.default_rng(20261001+n).uniform(-1, 1, (8, 2*n))
                record['capture_successes_out_of_eight'] = {}
                for amplitude in (1e-7, 1e-8, 1e-9):
                    record['capture_successes_out_of_eight'][str(amplitude)] = sum(
                        exact_capture(cfg, dict(qpos=np.r_[0, amplitude*v[:n]].tolist(),
                                                qvel=np.r_[0, amplitude*v[n:]].tolist()),
                                      gain)['success'] for v in directions)
        except (ValueError, np.linalg.LinAlgError) as error:
            record['error'] = str(error)
        env.close()
        records.append(record)
        print(f"n={n}: relative A difference={record['relative_a_change']:.3e}; rho={record.get('spectral_radius')}", flush=True)
    payload = dict(not_solution=True, runtime=runtime_metadata(), records=records,
                   note='Gravity stiffness is still obtained by finite differences of passive forces; this independently cross-checks the stepping map, not all MuJoCo mechanics.',
                   sources=[file_metadata(Path(__file__)), file_metadata(ROOT/'src/gcartpole/generalized_modes.py')])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
