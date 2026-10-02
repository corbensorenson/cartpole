"""Save delivered physical prefix nodes/actions, then independently replay it."""
import argparse
import json
import subprocess
from pathlib import Path
import numpy as np
from gcartpole.config import dump_json, load_config
from gcartpole.env import NLinkCartPoleEnv
from gcartpole.evidence import file_metadata
from gcartpole.ilqr import MujocoTransition, data_state
from scripts.search_capture_sequence import fixed_state_cfg
from scripts.search_fddp_capture import rebuild_feedback_warm_start, warm_start_diagnostics

p = argparse.ArgumentParser()
p.add_argument('--directory', required=True)
p.add_argument('--source', required=True)
p.add_argument('--config', required=True)
args = p.parse_args()
if Path(args.directory, 'result.json').exists():
    raise FileExistsError(Path(args.directory, 'result.json'))
out, source = Path(args.directory), Path(args.source)
prior = json.loads(source.read_text())
cfg = fixed_state_cfg(load_config(args.config), prior['selected_state'], 30.)
env = NLinkCartPoleEnv(cfg, progress=1., seed=prior['seed'])
env.reset(seed=prior['seed'])
transition = MujocoTransition(env, coordinate_transform=np.asarray(prior['controller']['coordinate_transform'], dtype=float), continuous_angles=True)
c = dict(prior['controller'])
controls, states = rebuild_feedback_warm_start(transition, transition.to_coordinates(data_state(env.data)),
    np.asarray(c['controls']), np.asarray(prior['search']['nominal_coordinate_states']), np.asarray(c['feedback_gains']))
controls = controls.astype(np.float32).astype(float)  # Actual action interface precision.
diagnostics = warm_start_diagnostics(transition, states[0], controls, states, require_feasible=True)
assert diagnostics['is_exactly_feasible']
actual = prior['result']['trajectory'][:len(controls)]
physical_states = np.asarray([row['qpos']+row['qvel'] for row in actual])
assert np.array_equal(states[1:], physical_states @ transition.coordinate_transform.T)
assert np.array_equal(controls, np.asarray([row['action'] for row in actual], dtype=np.float32).astype(float))
c.update(controls=controls.tolist(), final_trajectory_diagnostics=diagnostics,
         warm_start_diagnostics=diagnostics, physical_rebuild_source=file_metadata(source))
warm_path = out/'physical_controller.json'
dump_json(dict(not_solution=True, selected_state=prior['selected_state'], seed=prior['seed'], controller=c,
               search=dict(nominal_coordinate_states=states.tolist()), source=file_metadata(source)), warm_path)
env.close()
reference = source.parent/'synthesis/execution.json'
command = json.loads(reference.read_text())['command']
command[command.index('--config')+1] = args.config
command[command.index('--state-json')+1] = str(warm_path)
command[command.index('--initial-controller')+1] = str(warm_path)
command[command.index('--out')+1] = str(out/'synthesis/result.json')
subprocess.run(['.conda-aligator/bin/python', 'scripts/run_frontier_experiment.py', '--directory', str(out/'synthesis'),
                '--input', str(warm_path), '--input', str(source), '--input', str(reference), '--', *command], check=True)
d = json.loads((out/'synthesis/result.json').read_text())
assert d['controller']['final_trajectory_diagnostics']['is_exactly_feasible']
assert d['result']['length'] == prior['result']['length']
assert all(np.array_equal(row['qpos'], before['qpos']) and np.array_equal(row['qvel'], before['qvel'])
           for row,before in zip(d['result']['trajectory'],prior['result']['trajectory']))
d['physical_rebuild_comparison'] = dict(source=file_metadata(source), all_physical_states_identical=True,
    delivered_controls_saved=True, no_runtime_state_projection=True, reserved_seeds_used=False)
dump_json(d, out/'result.json')
print(json.dumps({k:v for k,v in d['result'].items() if not isinstance(v, (dict,list))}))
