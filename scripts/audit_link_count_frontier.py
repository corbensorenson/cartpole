#!/usr/bin/env python
"""Reproduce frontier diagnostics without modifying solvers or release artifacts.

Run from the repository root with PYTHONPATH=src:scripts and the existing
.conda-aligator/bin/python. These are component diagnostics, not new releases.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.linalg import eig, solve_discrete_are

from gcartpole.config import load_config
from gcartpole.env import NLinkCartPoleEnv, wrap_angle
from gcartpole.evidence import file_metadata, git_metadata, runtime_metadata
from gcartpole.generalized_modes import chain_normal_modes
from gcartpole.ilqr import MujocoTransition
from gcartpole.linear import modal_input_coupling
from gcartpole.modal import StateScales, dimensionless_absolute_transform
from evaluate_fddp_parked_route import run_episode
from evaluate_fddp_two_expert import load_controller
from gcartpole.generalized_energy import hanging_lqr_gain
from make_lqr_checkpoint import absolute_angle_cost, finite_difference_dynamics
from search_capture_sequence import fixed_state_cfg
from search_fddp_capture import rebuild_feedback_warm_start
from search_ilqr_capture import trajectory_integrity
from search_swingup_capture import lqr_gain
from refine_ilqr_capture_chain import source_trajectory

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = dict(cart_position=0.1, absolute_angle=100.0, cart_velocity=0.1,
               absolute_angular_velocity=1.0, relative_angle=1.0,
               relative_angular_velocity=0.01)


def exact_capture(cfg, state, gain, seconds=8.0):
    env = NLinkCartPoleEnv(fixed_state_cfg(cfg, state, seconds), progress=1.0, seed=0)
    env.reset(seed=0)
    saturated = 0
    raw_max = 0.0
    clean = True
    info = {}
    steps = 0
    for _ in range(env.max_steps):
        physical = np.r_[env.data.qpos.copy(), env.data.qvel.copy()]
        physical[1:env.n + 1] = wrap_angle(physical[1:env.n + 1])
        raw = float(-gain @ physical)
        raw_max = max(raw_max, abs(raw))
        saturated += abs(raw) > 1.0
        before_time = float(env.data.time)
        _, _, terminated, truncated, info = env.step([np.clip(raw, -1.0, 1.0)])
        steps += 1
        clean = bool(clean and np.isclose(env.data.time - before_time, env.dt,
                                         rtol=0.0, atol=1e-9)
                     and not any(w.number for w in env.data.warning)
                     and np.all(np.isfinite(env.data.qpos))
                     and np.all(np.isfinite(env.data.qvel)))
        if not clean or terminated or truncated:
            break
    result = dict(success=bool(info.get("success", False)) and clean,
                  final_hold_seconds=float(info.get("upright_streak_seconds", 0.0)),
                  max_hold_seconds=float(info.get("max_upright_streak_seconds", 0.0)),
                  raw_action_abs_max=raw_max, saturated_steps=int(saturated),
                  steps=steps, trajectory_clean=clean,
                  max_cart_excursion=float(env.max_cart_excursion),
                  termination_reason=info.get("termination_reason"))
    env.close()
    return result


def linear_audit(base):
    records = []
    cached = {}
    for n in range(7, 16):
        cfg = copy.deepcopy(base)
        cfg["env"]["n_links"] = n
        env = NLinkCartPoleEnv(cfg, progress=1.0, seed=0)
        modes = chain_normal_modes(env)
        a, b = finite_difference_dynamics(cfg, 1.0, 1e-7)
        q = absolute_angle_cost(n, WEIGHTS)
        record = dict(n_links=n, minimum_normalized_hanging_coupling=float(
            np.min(np.abs(modes.normalized_coupling))),
            hanging_frequencies_hz=(modes.angular_frequencies / (2 * np.pi)).tolist(),
            joint_mass_condition=float(np.linalg.cond(modes.joint_mass_matrix)))
        a_coarse, b_coarse = finite_difference_dynamics(cfg, 1.0, 1e-6)
        record["fd_relative_matrix_change"] = float(
            np.linalg.norm(a - a_coarse) / np.linalg.norm(a))
        record["fd_relative_input_change"] = float(
            np.linalg.norm(b - b_coarse) / np.linalg.norm(b))
        vals, left = eig(a, left=True, right=False)
        normalized = np.abs(left.conj().T @ b).ravel() / np.linalg.norm(b)
        record["minimum_unstable_left_eigenvector_input_overlap"] = float(
            np.min(normalized[np.abs(vals) > 1.0 + 1e-7]))
        record["riccati_designs"] = []
        transform = dimensionless_absolute_transform(n, StateScales(3, .15, 4, 15))
        inverse = np.linalg.inv(transform)
        for label, aa, bb, qq, map_back in (
            ("physical", a, b, q, np.eye(a.shape[0])),
            ("scaled", transform @ a @ inverse, transform @ b,
             inverse.T @ q @ inverse, transform),
        ):
            for balanced in (True, False):
                item = dict(coordinates=label, balanced=balanced)
                try:
                    p = solve_discrete_are(aa, bb, qq, np.array([[1000.0]]),
                                           balanced=balanced)
                    kz = np.linalg.solve(1000.0 + bb.T @ p @ bb, bb.T @ p @ aa)
                    gain = (kz @ map_back).ravel()
                    radius = float(np.max(np.abs(np.linalg.eigvals(a - b @ gain[None]))))
                    residual = aa.T @ p @ aa - p - aa.T @ p @ bb @ kz + qq
                    item.update(gain_norm=float(np.linalg.norm(gain)),
                                closed_loop_spectral_radius=radius,
                                dare_residual_over_p_norm=float(np.linalg.norm(residual) / np.linalg.norm(p)),
                                initially_nonsaturating_physical_l2_radius=float(1 / np.linalg.norm(gain)))
                    if label == "physical" and balanced:
                        cached[n] = (cfg, gain)
                        record.update(gain_norm=item["gain_norm"],
                                      closed_loop_spectral_radius=radius)
                except (ValueError, np.linalg.LinAlgError) as error:
                    item["error"] = str(error)
                record["riccati_designs"].append(item)
        env.close()
        records.append(record)
        print(f"linear n={n}: K={record.get('gain_norm')} rho={record.get('closed_loop_spectral_radius')}", flush=True)
    return records, cached


def support_audit():
    records = []
    for name in ("swingup11_uniform.yaml", "n11_free_split_stiffness_k40_d08.yaml",
                 "n11_free_split_stiffness_k50_d1.yaml", "n11_free_split_stiffness_k100_d2.yaml"):
        path = ROOT / ("configs" if name.startswith("swingup") else "tmp") / name
        if not path.exists():
            continue
        cfg = load_config(path)
        env = NLinkCartPoleEnv(cfg, progress=1 if name.startswith("swingup") else 0, seed=0)
        env.data.qpos[:] = 0.0
        env.data.qvel[:] = 0.0
        mujoco.mj_forward(env.model, env.data)
        mass = np.zeros((env.model.nv, env.model.nv))
        mujoco.mj_fullM(env.model, mass, env.data.qM)
        damping = np.diag(env.model.dof_damping)
        values = np.linalg.eigvals(-np.linalg.solve(mass, damping))
        z = values * env.model.opt.timestep
        rk4 = 1 + z + z**2/2 + z**3/6 + z**4/24
        record = dict(config=file_metadata(path), fastest_damping_pole=float(np.min(values.real)),
                      timestep=float(env.model.opt.timestep),
                      frozen_damping_rk4_max_amplification=float(np.max(np.abs(rk4))),
                      note="Frozen damping subsystem only; not a full nonlinear stability theorem.")
        # Full held-action transition at hanging: gravity should be stable here.
        env.data.qpos[:] = 0
        env.data.qpos[1] = np.pi
        transition = MujocoTransition(env)
        state = np.r_[env.data.qpos.copy(), np.zeros(env.model.nv)]
        a, _ = transition.linearize(transition.to_coordinates(state), 0,
                                   state_epsilon=1e-6, action_epsilon=1e-4)
        record["hanging_full_discrete_spectral_radius"] = float(np.max(np.abs(np.linalg.eigvals(a))))
        records.append(record)
        env.close()
    return records


def implementation_reproductions():
    transition = lambda state, action: np.asarray(state) + action
    states = rebuild_feedback_warm_start(transition, np.array([1.0]), np.zeros(2),
                                        np.zeros((3, 1)), -np.ones((2, 1)))
    defects = np.array([states[k + 1] - transition(states[k], 0.0) for k in range(2)])
    assert np.max(np.abs(defects)) == 1.0
    a = np.array([[1.0, 10.0], [0.0, 2.0]])
    b = np.array([[0.0], [1.0]])
    assert np.linalg.matrix_rank(np.column_stack((b, a @ b))) == 2
    toy_coupling = modal_input_coupling(a, b)
    assert toy_coupling == 0.0
    n = 11
    transform = dimensionless_absolute_transform(n, StateScales(3, .15, 4, 15))
    physical = np.zeros(2 * (n + 1))
    physical[1] = np.pi - 1e-8
    first = physical.copy()
    second = physical.copy()
    second[1] = -np.pi + 1e-8
    coordinate_first = transform @ first
    coordinate_second = transform @ second
    error = second - first
    error[1:n + 1] = wrap_angle(error[1:n + 1])
    # Small physical motion at the branch cut; StateVector and raw route errors disagree.
    branch = dict(euclidean_coordinate_error_norm=float(np.linalg.norm(coordinate_second-coordinate_first)),
                  periodic_coordinate_error_norm=float(np.linalg.norm(transform @ error)))
    # Existing heuristic permits a collapse after an upright row; it sees no warning/time counters.
    def row(angle):
        return dict(relative_angles=[angle], absolute_angles=[angle], qvel=[0.0, 0.0],
                    x=0.0, is_upright=abs(angle) < .15)
    collapse = [row(3.0), row(.1)] + [row(0.0) for _ in range(10)]
    return dict(feedback_warm_start_states=states.tolist(),
                feedback_warm_start_max_defect=float(np.max(np.abs(defects))),
                controllable_toy_reported_weakest_coupling=toy_coupling,
                controllable_toy_actual_rank=2,
                branch_cut=branch,
                collapse_after_upright_passes_current_integrity_heuristic=bool(trajectory_integrity(collapse)))


def bank_audit(base, gain):
    path = ROOT / "runs/generalized_solver/n11_p39_low_momentum_handoff_states.json"
    bank = json.loads(path.read_text())
    states = bank["states"]
    rows = [exact_capture(base, state, gain) for state in states]
    return dict(source=file_metadata(path), state_count=len(states),
                earliest_source_time=min(s["source_time_seconds"] for s in states),
                latest_source_time=max(s["source_time_seconds"] for s in states),
                max_initial_absolute_angle=max(s["max_abs_angle"] for s in states),
                max_initial_hinge_rms=max(s["hinge_velocity_rms"] for s in states),
                min_already_accumulated_hold=min(s["upright_streak_seconds"] for s in states),
                successes=sum(r["success"] for r in rows), results=rows)


def local_capture_audit(cached):
    records = []
    for n in range(7, 13):
        cfg, gain = cached[n]
        rng = np.random.default_rng(20261001 + n)
        directions = rng.uniform(-1, 1, (8, 2*n))
        for amplitude in (1e-7, 1e-6, 1e-5, 1e-4, 1e-3):
            trials = []
            for index, direction in enumerate(directions):
                qpos = np.r_[0.0, amplitude * direction[:n]]
                qvel = np.r_[0.0, amplitude * direction[n:]]
                result = exact_capture(cfg, dict(qpos=qpos.tolist(), qvel=qvel.tolist()), gain)
                result["direction_index"] = index
                trials.append(result)
            records.append(dict(n_links=n, amplitude=amplitude,
                                successes=sum(row["success"] for row in trials), trials=trials))
        print(f"capture n={n}: " + str([(r['amplitude'], r['successes']) for r in records if r['n_links']==n]), flush=True)
    return dict(note="Eight random relative-angle/rate directions per count, reused across amplitudes; component resets, zero cart state, amplitudes in rad and rad/s. Not a global basin certificate.", records=records)


def tiny_capture_audit(cached):
    records = []
    for n in (10, 11, 12):
        cfg, gain = cached[n]
        a, b = finite_difference_dynamics(cfg, 1.0, 1e-7)
        directions = np.random.default_rng(20261001+n).uniform(-1, 1, (8, 2*n))
        for amplitude in (1e-8, 1e-9, 1e-10, 1e-11, 1e-12):
            trials = []
            for direction in directions:
                state = dict(qpos=np.r_[0, amplitude*direction[:n]].tolist(),
                             qvel=np.r_[0, amplitude*direction[n:]].tolist())
                result = exact_capture(cfg, state, gain)
                x = np.r_[state['qpos'], state['qvel']]
                raw_max = 0.0
                for _ in range(400):
                    raw = float(-gain @ x)
                    raw_max = max(raw_max, abs(raw))
                    x = a @ x + b[:, 0] * np.clip(raw, -1, 1)
                result['linear_raw_action_abs_max'] = raw_max
                trials.append(result)
            records.append(dict(n_links=n, amplitude=amplitude,
                                successes=sum(r['success'] for r in trials), trials=trials))
    return records


def transferred_nominal_audit():
    path = ROOT / 'runs/generalized_solver/n11_from_n10_probe_fddp_warm.json'
    controller = load_controller(path, 11, load_config(ROOT / 'benchmarks/p1_capture_envelope.yaml'))
    controls, states, _ = source_trajectory(json.loads(path.read_text()))
    env = NLinkCartPoleEnv(load_config(ROOT / 'configs/swingup11_uniform.yaml'), progress=1, seed=0)
    transition = MujocoTransition(env, coordinate_transform=controller['transform'])
    defects = []
    clean = True
    for step, action in enumerate(controls):
        prediction = transition(states[step], action)
        clean = bool(clean and not any(w.number for w in transition.data.warning))
        if not clean:
            break
        defects.append(float(np.linalg.norm(transition.difference(prediction, states[step+1]))))
    env.close()
    return dict(source=file_metadata(path), checked_transitions=len(defects), clean=clean,
                max_coordinate_defect=max(defects), median_coordinate_defect=float(np.median(defects)),
                first_coordinate_defect=defects[0],
                note='The subsequent n11_from_n10_probe_fddp.json declares initial_feasible=true without rebuilding these states.')


def artifact_inventory():
    inventory = []
    for path in sorted((ROOT / "runs/generalized_solver").glob("n11*.json")):
        payload = json.loads(path.read_text())
        result = payload.get("result", {})
        search = payload.get("search", {})
        record = dict(path=str(path.relative_to(ROOT)), bytes=path.stat().st_size,
                      summary=payload.get("summary"), progress=search.get("progress", payload.get("progress")),
                      not_solution=payload.get("not_solution"),
                      reported_success=result.get("success"),
                      trajectory_integrity=result.get("trajectory_integrity"),
                      reported_hold=result.get("max_upright_streak_seconds"),
                      termination_reason=result.get("termination_reason"),
                      iterations=search.get("iterations"), converged=search.get("converged"))
        inventory.append(record)
    return inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    path = Path(args.out)
    if path.exists():
        raise FileExistsError(path)
    base = load_config(ROOT / "configs/swingup11_uniform.yaml")
    linear, cached = linear_audit(base)
    evidence = dict(schema_version=1, analysis_date="2026-10-01", not_solution=True,
                    runtime=runtime_metadata(), git=git_metadata(ROOT),
                    base_config=file_metadata(ROOT / "configs/swingup11_uniform.yaml"),
                    linear=linear, support_integration=support_audit(),
                    implementation_reproductions=implementation_reproductions(),
                    capture_bank=bank_audit(base, cached[11][1]),
                    local_capture=local_capture_audit(cached),
                    tiny_capture=tiny_capture_audit(cached),
                    transferred_nominal=transferred_nominal_audit())
    cross = json.loads((ROOT / "runs/generalized_solver/n11_chain_settled_launch_capture_ready_rail12.json").read_text())
    handoff = cross["best"]["metrics"]["best_handoff"]
    physical = np.r_[handoff["qpos"], handoff["qvel"]]
    physical[1:12] = wrap_angle(physical[1:12])
    evidence["settled_crossing"] = dict(max_abs_angle=handoff["max_abs_angle"],
        hinge_rms=handoff["hinge_velocity_rms"],
        absolute_rate_rms=float(np.sqrt(np.mean(np.cumsum(physical[13:])**2))),
        cart_position=physical[0], cart_velocity=physical[12],
        initial_raw_lqr_action=float(-cached[11][1] @ physical),
        exact_canonical_capture=exact_capture(base, handoff, cached[11][1]))
    cfg10 = load_config(ROOT / "configs/swingup10_uniform.yaml")
    spec = load_config(ROOT / "benchmarks/p1_capture_envelope.yaml")
    source = ROOT / "runs/generalized_solver/n10_fddp_refined_route_feedback100.json"
    controller = load_controller(source, 10, spec)
    settle_env = NLinkCartPoleEnv(cfg10, progress=1, seed=0)
    settle_gain = hanging_lqr_gain(settle_env, control_cost=1000)
    settle_env.close()
    result = run_episode(cfg10, controller, cached[10][1], settle_gain, seed=20261001,
                         park_seconds=16, cart_target=-.05, tracking_gain_scale=1,
                         phase_adaptive=False, phase_window=0, include_trace=True)
    evidence["fresh_ten_link_replay"] = result
    inventory = artifact_inventory()
    evidence["artifact_inventory"] = inventory
    evidence["artifact_count"] = len(inventory)
    evidence["sources"] = [file_metadata(ROOT / p) for p in (
        "scripts/audit_link_count_frontier.py", "src/gcartpole/ilqr.py",
        "src/gcartpole/fddp.py", "src/gcartpole/linear.py", "src/gcartpole/mjxml.py",
        "src/gcartpole/env.py", "scripts/search_fddp_capture.py",
        "scripts/search_swingup_chain.py", "scripts/evaluate_fddp_parked_route.py",
        "scripts/search_ilqr_capture.py", "docs/levers_and_pitfalls.md")]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n")
    print(f"saved {path}; bank={evidence['capture_bank']['successes']}/24; n10 replay={result['success']}", flush=True)


if __name__ == "__main__":
    main()
