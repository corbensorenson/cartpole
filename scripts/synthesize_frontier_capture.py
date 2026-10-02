#!/usr/bin/env python
"""Run the shared full-horizon physical capture repair on a declared initializer.

The generated candidate is development evidence. Canonical release gates and
reserved seeds are separate; no successful nominal solve advances a count.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from gcartpole.config import dump_json, load_config
from gcartpole.evidence import file_metadata


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--initializer', required=True)
    p.add_argument('--out-directory', required=True)
    p.add_argument('--iterations', type=int, default=20)
    p.add_argument('--lqr-decimal-digits', type=int, default=100)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--qp-solver', choices=('osqp', 'clarabel'), default='osqp')
    p.add_argument('--qp-max-iterations', type=int, default=10000)
    p.add_argument('--qp-initial-tolerance', type=float, default=.001)
    p.add_argument('--qp-tolerance', type=float, default=1e-8)
    p.add_argument('--qp-inexact-dual-tolerance', type=float, default=.001)
    p.add_argument('--trust-policy', choices=('legacy', 'agreement'), default='legacy')
    p.add_argument('--state-trust', type=float, default=.05)
    p.add_argument('--control-trust', type=float, default=.1)
    p.add_argument('--initial-regularization', type=float, default=10.)
    p.add_argument('--defect-penalty', type=float, default=1e5)
    args = p.parse_args()
    cfg = load_config(args.config)
    seed_base = 200000+1000*cfg['env']['n_links']
    reserved = set(range(seed_base, seed_base+20)) | set(range(seed_base+100, seed_base+220))
    reserved.add(seed_base+500)
    if args.seed in reserved:
        raise ValueError('reserved canonical seeds cannot be used for development synthesis')
    initial = json.loads(Path(args.initializer).read_text())
    capture = initial['controller']['capture_start_seconds']
    dt = cfg['env']['timestep']*cfg['env']['frame_skip']
    if capture is None or not 0 < capture < len(initial['controller']['controls'])*dt:
        raise ValueError('initializer requires an explicit positive capture interval inside the horizon')
    out = Path(args.out_directory)
    if (out/'synthesis').exists() or (out/'result.json').exists():
        raise FileExistsError(out)
    # This is the successful twelve-link residual-form full-horizon recipe.
    # Every change is a named CLI parameter recorded in the frozen experiment.
    values = {
        '--config': args.config, '--state-json': args.initializer, '--state-index': 'selected',
        '--initial-controller': args.initializer, '--iterations': args.iterations,
        '--optimizer': 'sparse-scvx', '--derivative-order': 4,
        '--state-epsilon': .0002, '--action-epsilon': .0005,
        '--tracking-gain-scale': 1, '--lqr-scale': 1, '--lqr-control-cost': 1000,
        '--lqr-decimal-digits': args.lqr_decimal_digits, '--terminal-weight': 0,
        '--terminal-state-weight': 1000, '--terminal-angle-factor': 20,
        '--terminal-hinge-velocity-factor': 4, '--initial-regularization': args.initial_regularization,
        '--defect-penalty': args.defect_penalty, '--stage-weight': .01, '--control-cost': .01,
        '--rail-soft-limit': 2.85, '--rail-weight': 5000000,
        '--handoff-lyapunov': 10, '--handoff-cart-abs': 1.5, '--handoff-angle-abs': .15,
        '--handoff-cart-velocity-abs': .5, '--handoff-hinge-velocity-rms': .75,
        '--capture-start-seconds': capture, '--capture-stage-weight': 1000, '--capture-value-weight': 0,
        '--seed': args.seed, '--out': out/'synthesis/result.json', '--optimize-suffix-start-seconds': 0,
        '--sparse-capture-angle-limit': .14, '--sparse-state-trust': args.state_trust,
        '--sparse-control-trust': args.control_trust, '--sparse-qp-solver': args.qp_solver,
        '--sparse-qp-max-iterations': args.qp_max_iterations, '--sparse-qp-tolerance': args.qp_tolerance,
        '--sparse-qp-initial-tolerance': args.qp_initial_tolerance,
        '--sparse-qp-inexact-dual-tolerance': args.qp_inexact_dual_tolerance,
        '--sparse-trust-policy': args.trust_policy,
    }
    command = [sys.executable, 'scripts/search_fddp_capture.py']
    for key, value in values.items():
        command.extend([key, str(value)])
    command.extend(['--solver-verbose', '--continuous-angles', '--physical-shooting',
                    '--defer-handoff-until-horizon', '--enforce-rail-during-search'])
    subprocess.run([sys.executable, 'scripts/run_frontier_experiment.py', '--directory', str(out/'synthesis'),
        '--input', args.initializer, '--input', args.config, '--', *command], check=True)
    result = json.loads((out/'synthesis/result.json').read_text())
    result['shared_capture_recipe'] = dict(initialization=file_metadata(args.initializer),
        parameters=vars(args), entire_horizon_free=True, no_runtime_state_projection=True,
        release_promotion=False)
    dump_json(result, out/'result.json')
    print(json.dumps({k:v for k,v in result['result'].items() if not isinstance(v,(dict,list))}))


if __name__ == '__main__':
    main()
