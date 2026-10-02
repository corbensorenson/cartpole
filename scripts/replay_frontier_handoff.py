"""Trim only a route's duration and physically test the unchanged LQR handoff."""
import argparse
import json
import subprocess
from pathlib import Path
import numpy as np
from gcartpole.config import dump_json, load_config
from gcartpole.evidence import file_metadata

p = argparse.ArgumentParser()
p.add_argument('--directory', required=True)
p.add_argument('--source', required=True)
p.add_argument('--config', required=True)
p.add_argument('--handoff-seconds', type=float, required=True)
args = p.parse_args()
if Path(args.directory, 'result.json').exists():
    raise FileExistsError(Path(args.directory, 'result.json'))
out, source = Path(args.directory), Path(args.source)
prior = json.loads(source.read_text())
cfg = load_config(args.config)
dt = cfg['env']['timestep']*cfg['env']['frame_skip']
if not np.isfinite(args.handoff_seconds) or not np.isclose(args.handoff_seconds/dt, round(args.handoff_seconds/dt), rtol=0., atol=1e-10):
    raise ValueError('handoff must be a finite tick-aligned time')
steps = int(round(args.handoff_seconds/dt))
if not 1 < steps < len(prior['controller']['controls']):
    raise ValueError('handoff must be inside the inherited route')
if len(prior['selected_state']['qpos']) != cfg['env']['n_links']+1:
    raise ValueError('source and config dimensions differ')
c = dict(prior['controller'])
c['controls'] = c['controls'][:steps]
c['feedback_gains'] = c['feedback_gains'][:steps]
if 'solver_feedback_gains' in c:
    c['solver_feedback_gains'] = c['solver_feedback_gains'][:steps]
trimmed = dict(not_solution=True, selected_state=prior['selected_state'], seed=prior['seed'],
               controller=c, search=dict(nominal_coordinate_states=prior['search']['nominal_coordinate_states'][:steps+1]),
               source=file_metadata(source), diagnostic='Unchanged prefix truncated to test earlier physical LQR handoff')
warm_path = out/'truncated_controller.json'
dump_json(trimmed, warm_path)
reference = source.parent/'synthesis/execution.json'
command = json.loads(reference.read_text())['command']
command[command.index('--config')+1] = args.config
command[command.index('--state-json')+1] = str(source)
command[command.index('--initial-controller')+1] = str(warm_path)
command[command.index('--optimizer')+1] = 'sqrt-ilqr'
command[command.index('--out')+1] = str(out/'synthesis/result.json')
command[command.index('--optimize-suffix-start-seconds')+1] = '0'
if '--sparse-capture-angle-limit' in command:
    i = command.index('--sparse-capture-angle-limit')
    del command[i:i+2]
if '--enforce-rail-during-search' in command:
    command.remove('--enforce-rail-during-search')
for flag in ('--capture-start-seconds', '--capture-stage-weight', '--capture-value-weight'):
    if flag not in command:
        continue
    i = command.index(flag)
    del command[i:i+2]  # Objective is unused when replaying inherited gains.
command.append('--replay-only')
subprocess.run(['.conda-aligator/bin/python', 'scripts/run_frontier_experiment.py',
                '--directory', str(out/'synthesis'), '--input', str(warm_path), '--input', str(source),
                '--input', str(reference), '--', *command], check=True)
data = json.loads((out/'synthesis/result.json').read_text())
assert np.array_equal(data['controller']['controls'], prior['controller']['controls'][:steps])
assert np.array_equal(data['controller']['feedback_gains'], prior['controller']['feedback_gains'][:steps])
assert np.array_equal(data['search']['nominal_coordinate_states'], prior['search']['nominal_coordinate_states'][:steps+1])
physical_prefix_identical = all(np.array_equal(data['result']['trajectory'][i]['qpos'], prior['result']['trajectory'][i]['qpos'])
                                and np.array_equal(data['result']['trajectory'][i]['qvel'], prior['result']['trajectory'][i]['qvel'])
                                for i in range(min(steps, data['result']['length'])))
if not physical_prefix_identical:
    raise ValueError('physical prefix changed; this is not a handoff-only comparison')
data['handoff_only_comparison'] = dict(source=file_metadata(source), handoff_seconds=args.handoff_seconds,
    prefix_controls_identical=True, prefix_feedback_identical=True, nominal_prefix_identical=True,
    physical_prefix_identical=physical_prefix_identical, reserved_seeds_used=False,
    no_runtime_state_projection=True, lqr_gain_unchanged=True)
dump_json(data, out/'result.json')
print(json.dumps({k:v for k,v in data['result'].items() if not isinstance(v, (list, dict))}))
