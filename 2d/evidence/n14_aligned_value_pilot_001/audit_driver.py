"""Audit retained inner native predictions, including the final successor."""
import argparse
import json
import hashlib
from pathlib import Path
import numpy as np
from spatialpole import Config,SpatialCartpole
from spatialpole.aligned_value import aligned_value_policy_from_payload
from spatialpole.experiment import episode,write_json

p = argparse.ArgumentParser()
p.add_argument('--run',required=True)
a = p.parse_args()
out = Path(a.run)
summary = json.loads((out/'summary.json').read_text())
payload = np.load(out/'route.npz')
config = Config(**json.loads(str(payload['config'])))
oracle = SpatialCartpole(config)
identities,records = {},[]
for row in summary['results']:
    path = out/('nominal.npz' if row['angle_noise']==0 else f"seed{row['seed']}.npz")
    identities[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    t = np.load(path)
    steps,offsets = t['native_search_step'],t['native_search_offsets']
    assert len(steps)==len(t['actions'])-960
    assert row['policy_metadata']['predictor_native_steps']==4*int(offsets[-1])
    selected_costs,baseline_costs = [],[]
    for i,step in enumerate(steps):
        lo,hi = offsets[i:i+2]
        actions = t['native_search_action'][lo:hi]
        assert 1<=len(actions)<=16 and len({u.tobytes() for u in actions})==len(actions)
        assert np.max(np.linalg.norm(actions.astype(float),axis=1))<=80
        index = int(t['native_search_selected_index'][i])
        fallback = bool(t['native_search_baseline_fallback'][i])
        chosen = lo if fallback else lo+index
        assert fallback==(index<0)
        if fallback:
            assert not np.any(t['native_search_valid'][lo:hi])
        else:
            assert t['native_search_valid'][chosen]
            valid_costs = t['native_search_cost'][lo:hi][t['native_search_valid'][lo:hi]]
            assert t['native_search_cost'][chosen]==min(valid_costs)
        actual = 960+int(step)
        np.testing.assert_array_equal(t['actions'][actual],t['native_search_action'][chosen])
        np.testing.assert_array_equal(t['states'][actual+1],t['native_search_successor'][chosen])
        if actual in (960,1060,1160,1260,1360,1460):
            for k in range(lo,hi):
                oracle.min_plane_margin,oracle.warning_seen = config.plane_limit,False
                successor = oracle.transition(t['states'][actual],t['native_search_action'][k])
                np.testing.assert_array_equal(successor,t['native_search_successor'][k])
                assert oracle.min_plane_margin==t['native_search_plane_margin'][k]
                assert oracle.warning_seen==t['native_search_physics_warning'][k]
    records.append(dict(seed=row['seed'],commands=len(steps),distinct_queries=int(offsets[-1]),
        selected_successors_exact=True,final_successor_exact=True,candidate_accounting_exact=True))
assert oracle.native_steps<=2304
native_spot_steps = oracle.native_steps
oracle.close()
f = lambda e: aligned_value_policy_from_payload(e,payload)
reexecution = episode(config,f,20261016,'hanging',0.,out/'nominal_reexecution.npz')
old,new = np.load(out/'nominal.npz'),np.load(out/'nominal_reexecution.npz')
parity = {k:bool(np.array_equal(old[k],new[k])) for k in old.files if k not in ('config','policy_metadata')}
assert all(parity.values()),parity
reexecution_steps = reexecution['native_steps']+reexecution['proof_native_steps']+reexecution['policy_metadata']['predictor_native_steps']
inclusive = summary['inclusive_native_steps']+native_spot_steps+reexecution_steps
assert inclusive<=285000
write_json(out/'native_integrity_audit.json',dict(input_tape_sha256=identities,episodes=records,
    nominal_reexecution=reexecution,nominal_reexecution_array_parity=parity,
    native_spot_steps=native_spot_steps,reexecution_inclusive_steps=reexecution_steps,
    pilot_inclusive_native_steps=summary['inclusive_native_steps'],total_inclusive_native_steps=inclusive,
    scope='integrity on reused pilot data; not independent qualification'))
print(json.dumps(dict(audit='passed',total_inclusive_native_steps=inclusive)),flush=True)
