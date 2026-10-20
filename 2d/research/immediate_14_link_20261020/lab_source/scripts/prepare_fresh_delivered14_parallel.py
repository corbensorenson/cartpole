"""Original reset and unchanged common prepare, with an exclusive case claim."""
import argparse
import json
import os
from pathlib import Path
import shutil
import numpy as np
from spatialpole import Config,SpatialCartpole
from spatialpole.experiment import freeze_inputs,write_json
from spatialpole.native_mpc import exact_bytes
from delivered_planner14 import prepare
from delivered_fresh_context14_parallel import registered,serialized_args
from stage_six_source_synthesis14 import helper_closure
from eleven_synthesis_protocol import read,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--registration',required=True,type=Path);p.add_argument('--out',required=True,type=Path);a=p.parse_args()
    bank,runtime,cm,m,expected,prior=registered(a.registration);assert not a.out.exists()
    with (a.registration/'activation.json').open('x',encoding='utf-8') as f:
        json.dump(dict(pid=os.getpid(),run=str(a.out.resolve()),registration_manifest_sha256=sha(a.registration/'frozen_manifest.json')),f,indent=2)
    freeze_inputs(a.out,serialized_args(a))
    for name in helper_closure({'prepare_fresh_delivered14_parallel.py'}):shutil.copyfile(Path(__file__).with_name(name),a.out/name)
    write_json(a.out/'frozen_manifest.json',dict(protocol=cm['protocol'],config=cm['config'],seed=m['seed'],noise=.02,conditioning_steps=0,
        registration=str(a.registration.resolve()),registration_manifest_sha256=sha(a.registration/'frozen_manifest.json'),
        candidate=m['candidate'],candidate_manifest_sha256=m['candidate_manifest_sha256'],controller_identity=cm['controller_identity'],
        registered_initial_sha256=m['registered_initial_sha256'],cohort=m['cohort'],cohort_manifest_sha256=m['cohort_manifest_sha256'],qualification_attempt=True,max_native_steps=9072324,max_wall_seconds=15000,qualified=False))
    write_json(a.out/'worker.json',dict(pid=os.getpid()));geometry=SpatialCartpole(Config(n_links=14))
    try:
        observed=expected.copy()  # Already sampled once by fresh registration; no seed enters prepare.
        assert geometry.native_steps==0;np.savez_compressed(a.out/'observed_state.npz',state=observed)
        report=prepare(observed,bank,runtime,a.out/'planning')
        write_json(a.out/'summary.json',dict(**report,controller_identity=cm['controller_identity'],seed=m['seed'],
            preparation_only=True,actual_benchmark_episode_executed=False,source_manifest_sha256=sha(a.out/'frozen_manifest.json')))
        write_json(a.out/'completed.json',dict(status='completed'));print(read(a.out/'summary.json'),flush=True)
    finally:geometry.close()


if __name__=='__main__':main()
