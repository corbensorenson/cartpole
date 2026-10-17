"""Verify causal launch forecast tapes against complete actual suffixes."""
import argparse
import json
from pathlib import Path
import numpy as np
from spatialpole.experiment import write_json

p = argparse.ArgumentParser()
p.add_argument('--folder',required=True)
p.add_argument('--budget',type=int,required=True)
a = p.parse_args()
folder = Path(a.folder)
assert (folder/'completed.json').exists()
summary = json.loads((folder/'summary.json').read_text())
counts,query_steps = {},0
for row in summary['results']:
    tape = np.load(folder/('nominal.npz' if row['angle_noise']==0 else f"seed{row['seed']}.npz"))
    m = json.loads(str(tape['policy_metadata']))
    assert m==row['policy_metadata']
    assert m['launch_admitted'] and len(m['launch_forecasts'])==2
    chosen,start = m['selected_launch_profile'],m['launch_forecast_start_step']
    np.testing.assert_array_equal(tape['states'][start:],tape[f'forecast_{chosen}_states'])
    for name in ('actions','raw_actions','tilts'):
        np.testing.assert_array_equal(tape[name][start:],tape[f'forecast_{chosen}_{name}'])
    forecasts = m['launch_forecasts']
    admitted = [r['profile'] for r in forecasts if r['success'] and r['steps']==1500
                and not r['physics_warning'] and r['plane_margin']>=0]
    assert admitted and chosen==admitted[0]
    for r in forecasts:
        assert r['native_steps']==4*len(tape[f"forecast_{r['profile']}_actions"])
    steps = sum(r['native_steps'] for r in forecasts)
    assert steps==m['predictor_native_steps']==m['launch_predictor_native_steps']
    assert steps<=2*(1500-start)*4
    assert row['success'] and row['steps']==1500 and row['replay_exact']
    counts[str(chosen)] = counts.get(str(chosen),0)+1
    query_steps += steps
total = sum(r['native_steps']+r['proof_native_steps'] for r in summary['results'])+query_steps+summary.get('design_native_steps',0)
assert total<=a.budget
report = dict(episodes=len(summary['results']),selected_profiles=counts,
    all_selected_forecasts_byte_exact=True,predictor_native_steps=query_steps,
    inclusive_native_steps=total,budget=a.budget,
    scope='causal model forecasts; actual full hanging episodes and independent replay separately retained')
write_json(folder/'forecast_actual_parity.json',report)
print(json.dumps(report))
