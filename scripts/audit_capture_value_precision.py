#!/usr/bin/env python
"""Compare capture-value arithmetic on identical binary64 plant/gain/state inputs."""
import argparse
import json
from pathlib import Path

import mpmath
import numpy as np

from gcartpole.config import dump_json, load_config
from gcartpole.evidence import file_metadata, runtime_metadata, utc_timestamp
from gcartpole.modal import dimensionless_wrapped_state
try:
    from scripts.evaluate_fddp_two_expert import load_controller
    from scripts.evaluate_fddp_parked_route import capture_metric
    from scripts.make_lqr_checkpoint import finite_difference_dynamics
    from scripts.search_swingup_capture import lqr_gain
except ModuleNotFoundError:
    from evaluate_fddp_two_expert import load_controller
    from evaluate_fddp_parked_route import capture_metric
    from make_lqr_checkpoint import finite_difference_dynamics
    from search_swingup_capture import lqr_gain


def precise_value_matrix(a, b, gain, transform, scale, digits):
    ctx = mpmath.mp.clone(); ctx.dps = digits
    convert = lambda array: ctx.matrix([[ctx.mpf(float(v)) for v in row] for row in np.atleast_2d(array)])
    t = convert(transform)
    closed = t*(convert(a)-convert(b)*(ctx.mpf(float(scale))*convert(gain)))*(t**-1)
    radius = max(abs(v) for v in ctx.eig(closed, left=False, right=False))
    if radius >= 1:
        raise ValueError(f"promoted-input closed loop is unstable: {radius}")
    power, value = closed.copy(), ctx.eye(closed.rows)
    for iteration in range(40):
        value = value+power.T*value*power
        power = power*power
        if max(abs(v) for v in power) < ctx.mpf("1e-40"):
            break
    else:
        raise ValueError("Lyapunov doubling series did not converge")
    residual = value-closed.T*value*closed-ctx.eye(closed.rows)
    root = ctx.cholesky(value).T
    return ctx, value, root, dict(spectral_radius=str(radius), doubling_iterations=iteration+1,
        absolute_equation_residual=str(max(abs(v) for v in residual)),
        maximum_matrix_entry=str(max(abs(v) for v in value)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--controller", required=True)
    parser.add_argument("--spec", default="benchmarks/p1_capture_envelope.yaml")
    parser.add_argument("--decimal-digits", type=int, default=100)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if Path(args.out).exists():
        raise FileExistsError(args.out)
    cfg, spec = load_config(args.config), load_config(args.spec)
    payload = json.loads(Path(args.controller).read_text())
    n = cfg["env"]["n_links"]
    controller = load_controller(Path(args.controller), n, spec)
    gain = lqr_gain(cfg, progress=1., fd_eps=1e-7,
                    control_cost=controller["lqr_control_cost"], q_weights=controller["lqr_weights"])
    a, b = finite_difference_dynamics(cfg, 1., 1e-7)
    legacy = dict(controller, defer_handoff_until_horizon=False)
    original = capture_metric(cfg, legacy, gain)
    ctx, accurate, root, diagnostics = precise_value_matrix(
        a, b, gain, controller["transform"], controller["lqr_scale"], args.decimal_digits)
    rounded_root = np.array(root.tolist(), dtype=float)
    promoted_original = ctx.matrix([[ctx.mpf(float(v)) for v in row] for row in original])
    samples = []
    nominal = controller["nominal_states"]
    physical_end = np.linalg.solve(controller["transform"], nominal[-1])
    selected = [("nominal_endpoint", physical_end)]
    rows = payload["result"]["trajectory"]
    selected += [("actual_minimum", np.r_[r["qpos"], r["qvel"]]) for r in
                 [min(rows, key=lambda r:r["dimensionless_lyapunov_value"])]]
    handoff = payload["result"].get("first_handoff_step")
    if handoff:
        r = next(r for r in rows if r["step"] == handoff)
        selected.append(("actual_capture_decision", np.r_[r["qpos"], r["qvel"]]))
    for name, state in selected:
        z = dimensionless_wrapped_state(state[:n+1], state[n+1:], controller["transform"])
        mp_z = ctx.matrix([ctx.mpf(float(v)) for v in z])
        samples.append(dict(name=name, state=state.tolist(), coordinate_state=z.tolist(),
            original_binary64_value=float(z@original@z),
            original_matrix_precise_dot=str((mp_z.T*promoted_original*mp_z)[0]),
            precise_matrix_precise_dot=str((mp_z.T*accurate*mp_z)[0]),
            precise_root_binary64_value=float(np.linalg.norm(rounded_root@z)**2)))
    dump_json(dict(schema_version=1, not_solution=True, generated_at=utc_timestamp(),
        parameters=vars(args), source=file_metadata(args.controller), config=file_metadata(args.config),
        runtime=runtime_metadata(), diagnostics=diagnostics, samples=samples,
        relative_matrix_difference=str(max(abs(accurate[i,j]-promoted_original[i,j])
            for i in range(accurate.rows) for j in range(accurate.cols))/max(abs(v) for v in accurate)),
        input_matrices=dict(a=a.tolist(), b=b.tolist(), gain=gain.tolist(), transform=controller["transform"].tolist()),
        precise_matrix=[[str(accurate[i,j]) for j in range(accurate.cols)] for i in range(accurate.rows)],
        rounded_value_factor=rounded_root.tolist(),
        note="Same binary64 identification, rounded gain and wrapped state inputs promoted exactly; this isolates Lyapunov solve/evaluation arithmetic. No improved plant identification, controller change, trajectory replay or release claim."), args.out)
    print(json.dumps(dict(diagnostics=diagnostics, samples=[{k:v for k,v in r.items()
        if k not in ("state", "coordinate_state")} for r in samples])), flush=True)


if __name__ == "__main__":
    main()
