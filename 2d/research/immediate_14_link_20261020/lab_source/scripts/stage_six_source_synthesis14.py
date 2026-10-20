"""Freeze six-source synthesis only after audited known22 success; no reruns."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from spatialpole import Config, SpatialCartpole
from spatialpole.experiment import freeze_inputs, write_json
from spatialpole.native_mpc import exact_bytes
from spatial_motion_library14 import reference
from spatial_synthesis_cohort_protocol import ALGORITHM
from eleven_synthesis_protocol import read, sha

NUMERICAL_HELPERS = tuple(sorted(set(ALGORITHM) | {
    'spatial_motion_library14.py', 'prepare_spatial_motion_blend14.py',
    'spatial_source_bank14.py', 'working_source14_support.py',
    'evaluate_spatial_observed_case.py', 'eleven_synthesis_protocol.py'}))
IDENTITY_KEYS = ('protocol', 'config', 'algorithm_sha256', 'source_sha256',
    'template_sha256', 'solver_receipt_sha256', 'max_models',
    'max_refinement_native_steps', 'max_case_native_steps', 'proposal_rule')


def helper_closure(names):
    """Include local lazy audit imports so the frozen directory is standalone."""
    directory = Path(__file__).parent
    found, pending = set(), set(names)
    while pending:
        name = pending.pop()
        if name in found:
            continue
        source = directory / name
        assert source.is_file()
        found.add(name)
        for node in ast.walk(ast.parse(source.read_text(encoding='utf-8-sig'))):
            modules = ([node.module] if isinstance(node, ast.ImportFrom) and node.module
                else [alias.name for alias in node.names] if isinstance(node, ast.Import) else [])
            for module in modules:
                local = module.split('.')[0] + '.py'
                if (directory / local).is_file() and local not in found:
                    pending.add(local)
    return sorted(found)


def main():
    p = argparse.ArgumentParser()
    for name in ('preparation-inventory', 'known-success-inventory', 'runtime', 'out'):
        p.add_argument('--' + name, required=True)
    a = p.parse_args()
    prep_path, known_path, runtime, out = map(Path, (a.preparation_inventory, a.known_success_inventory, a.runtime, a.out))
    prep, known = read(prep_path), read(known_path)
    assert prep['passed'] and prep['ready_for_refinement'] and known['passed'] and known['task_success']
    case, audit, preparation = Path(known['run']), Path(known['audit']), Path(prep['run'])
    cm, pm, g = read(case / 'frozen_manifest.json'), read(preparation / 'frozen_manifest.json'), read(audit / 'summary.json')
    assert cm['protocol'] == 'observed-state-six-source-native14-v1' and cm['seed'] == 20261122
    assert pm['protocol'] == 'six-source-common-azimuth-motion14-v1' and pm['seed'] == 20261122
    assert g['passed'] and g['task_success'] and g['all_forecasts_cold_byte_exact']
    assert g['actual_cold_byte_exact_via_identical_forecast']
    assert sha(case / 'summary.json') == known['summary_sha256'] == g['source_summary_sha256']
    assert sha(audit / 'summary.json') == known['audit_summary_sha256']
    assert cm['parent_inventory_sha256'] == sha(prep_path)
    assert sha(preparation / 'summary.json') == prep['summary_sha256']
    assert sha(Path(prep['audit']) / 'summary.json') == prep['audit_summary_sha256']
    assert sha(runtime / 'receipt.json') == cm['solver_receipt_sha256']
    assert known['result']['steps'] == 1500 and known['result']['best_hold_seconds'] >= 5
    assert known['result']['policy_metadata']['park_seconds'] == 0
    assert known['result']['policy_metadata']['physical_velocities'] == 44
    freeze_inputs(out, a)
    assert read(out / 'inputs.json')['source_sha256'] == cm['source_sha256']
    (out / 'helpers').mkdir()
    (out / 'templates').mkdir()
    shutil.copytree(out / 'source', out / 'frozen_package/spatialpole')
    templates = []
    for name, digest in pm['template_sha256'].items():
        source = preparation / 'templates' / name
        assert sha(source) == digest
        shutil.copyfile(source, out / 'templates' / name)
        with np.load(source) as z:
            templates.append({key: z[key].copy() for key in ('states', 'actions')})
    assert len(templates) == 6
    geometry = SpatialCartpole(Config(n_links=14))
    try:
        with np.load(case / 'observed_state.npz') as z:
            initial = z['state'].copy()
        v, info = reference(geometry, initial, templates)
        with np.load(case / 'virtual_reference.npz') as z:
            for key in ('states', 'actions'):
                exact_bytes(v[key], z[key])
        assert geometry.native_steps == 0
        write_json(out / 'known22_constructor_parity.json', dict(passed=True, native_steps=0,
            all1501_states_and1500_forces_byte_exact=True, known_success_inventory_sha256=sha(known_path),
            source_reference_sha256=sha(case / 'virtual_reference.npz'), preparation=info,
            scope='known22 mean reconstructed only, actual/planning evidence inherited; not a new sample or six old-policy equivalence claim'))
    finally:
        geometry.close()
    helpers = helper_closure(set(ALGORITHM) | {'stage_six_source_synthesis14.py', 'evaluate_six_source14_case.py',
        'audit_six_source14_case.py', 'spatial_motion_library14.py', 'prepare_spatial_motion_blend14.py',
        'spatial_source_bank14.py', 'working_source14_support.py', 'evaluate_spatial_observed_case.py',
        'eleven_synthesis_protocol.py'})
    for name in helpers:
        shutil.copyfile(Path(__file__).with_name(name), out / 'helpers' / name)
    data = dict(protocol='observed-state-six-source-native14-v4', config=Config(n_links=14).__dict__,
        helper_sha256={name: sha(out / 'helpers' / name) for name in helpers}, source_sha256=cm['source_sha256'],
        algorithm_sha256={name: sha(out / 'helpers' / name) for name in NUMERICAL_HELPERS},
        template_sha256=pm['template_sha256'], source_lineage=pm['source_lineage'],
        source21_canonical_rotation=pm['source21_canonical_rotation'],
        solver_receipt_sha256=sha(runtime / 'receipt.json'), runtime=str(runtime.resolve()),
        max_models=4, max_refinement_native_steps=8766320, max_case_native_steps=8778320,
        proposal_rule='state-only global common-azimuth/simplex minimum over six fixed successful motions, fit improvement diagnostic only; original physical gates unchanged',
        physical_episode_commands=1500, initial_noise=.02, planning_velocities=44, park_seconds=0,
        registered_next_development_seed=20261123,
        known22_constructor_parity_sha256=sha(out / 'known22_constructor_parity.json'))
    data['controller_identity'] = hashlib.sha256(json.dumps({k: data[k] for k in IDENTITY_KEYS}, sort_keys=True).encode()).hexdigest()
    write_json(out / 'frozen_manifest.json', data)
    write_json(out / 'summary.json', dict(passed=True, links=14, qualified=False, native_steps=0,
        controller_identity=data['controller_identity'], known22_proposal_only_reused=True,
        new_task_successes=0, new_noisy_samples=0, reserved_cases_consumed=0,
        scope='six-source general constructor frozen after audited22, before next candidate execution on retained development23; not broad fourteen qualification'))
    write_json(out / 'completed.json', dict(status='completed'))
    print(read(out / 'summary.json'), flush=True)


if __name__ == '__main__':
    main()
