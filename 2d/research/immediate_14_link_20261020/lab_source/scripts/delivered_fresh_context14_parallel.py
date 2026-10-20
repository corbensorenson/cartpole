"""Fresh cohort bookkeeping; observed-state numerical preparation is unchanged."""
import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT.parent / 'experiments/2d/fourteen-link-immediate'
FRESH = list(range(20263816, 20263848))
PROTOCOL = 'common-delivered14-fresh-qualification-parallel-v2'
REGISTRATION = 'common-delivered14-fresh-case-registration-parallel-v2'
RESERVATIONS = ('reserved_14_immediate_20263816.json', 'reserved_working14_immediate_20263816.json')
CASE_NATIVE_CAP = 9072324 + 384324 + 24000
MAX_COHORT_WALL = 86400
REQUIRED = ('freeze_fresh_delivered14_parallel.py', 'delivered_fresh_context14_parallel.py',
    'register_fresh_delivered14_parallel.py', 'prepare_fresh_delivered14_parallel.py',
    'audit_fresh_delivered14_parallel.py', 'witness_fresh_delivered14_parallel.py',
    'inventory_fresh_delivered14_parallel.py', 'inventory_fresh_delivered14_witness_parallel.py',
    'run_fresh_delivered14_parallel.py', 'audit_fresh_delivered14_cohort_parallel.py')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def serialized_args(args):
    return argparse.Namespace(**{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()})


def require_runtime(expected):
    import importlib.metadata
    import os
    import platform
    import sys
    import numpy
    import mujoco
    current = dict(python=sys.version,platform=platform.platform(),numpy=numpy.__version__,mujoco=mujoco.__version__,
        packages={name:importlib.metadata.version(name) for name in expected['packages']},
        thread_environment={name:os.environ[name] for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS') if name in os.environ})
    assert current == expected and current['thread_environment'] == {}, 'Original numerical runtime changed'
    return current


def cohort(folder):
    """Validate a previously admitted freeze; never reserve or sample here."""
    folder = Path(folder).resolve()
    m = read(folder / 'frozen_manifest.json')
    assert folder.parent == EXPERIMENTS.resolve() and (folder / 'completed.json').exists()
    assert m['protocol'] == PROTOCOL and m['seeds'] == FRESH and not m['qualified']
    assert m['fresh_qualification_activated'] and m['development_states'] == 45
    assert m['per_case_native_cap'] == CASE_NATIVE_CAP and m['total_native_cap'] == 32 * CASE_NATIVE_CAP
    assert m['max_wall_seconds'] == MAX_COHORT_WALL
    for name in RESERVATIONS:
        reservation = read(folder.parent / name)
        assert reservation['seeds'] == FRESH and Path(reservation['run']).resolve() == folder
        assert reservation['controller_identity'] == m['controller_identity']
        assert reservation['frozen_manifest_sha256'] == sha(folder / 'frozen_manifest.json')
    for name, digest in m['helper_sha256'].items():
        assert Path(name).name == name and sha(folder / 'helpers' / name) == digest == sha(ROOT / 'scripts' / name)
    assert sha(folder / 'prospective_protocol.md') == m['prospective_protocol_sha256']
    proof = Path(m['development_proof'])
    assert sha(proof / 'summary.json') == m['development_proof_summary_sha256']
    assert sha(proof / 'frozen_manifest.json') == m['development_proof_manifest_sha256']
    development = read(proof / 'summary.json')
    assert development['passed'] and development['all45_development_passed'] and development['fresh_gate_satisfied']
    assert development['verified_development_states'] == 45
    from delivered_candidate14 import candidate
    bank, runtime, cm = candidate(m['candidate'])
    assert sha(Path(m['candidate']) / 'frozen_manifest.json') == m['candidate_manifest_sha256']
    assert cm['controller_identity'] == m['controller_identity'] == development['controller_identity']
    assert cm['source_sha256'] == m['source_sha256'] and cm['fresh_qualification_seeds'] == FRESH
    assert len(bank) == 15 and not cm['fresh_qualification_activated']
    assert m['original_runtime'] == read(Path(m['candidate']) / 'inputs.json')['runtime']
    require_runtime(m['original_runtime'])
    return bank, runtime, cm, m, development


def previous(path, cm, next_seed):
    """Require the preceding fresh case's immutable canonical/reproduction proof."""
    p = read(path)
    assert p['passed'] and p['task_success'] and not p['qualified'] and p['links'] == 14
    assert p['seed'] == next_seed - 1 and p['seed'] in FRESH
    assert p['controller_identity'] == cm['controller_identity']
    assert p['fresh_case_ordinal'] == FRESH.index(p['seed']) + 1
    assert p['original_reset_episodes'] == p['independent_reproduction_episodes'] == 1
    r = p['result']
    assert r['steps'] == 1500 and r['success'] and r['best_hold_seconds'] >= 5.
    assert r['angle_noise'] == .02 and r['park_seconds'] == 0. and r['physical_velocities'] == 44
    assert r['plane_margin'] >= 0 and not r['physics_warning'] and r['max_force_norm'] <= 80.
    for folder in (Path(p['run']), Path(p['audit'])):
        assert (folder / 'completed.json').exists()
        for name, item in p['studies'][str(folder)].items():
            assert sha(folder / name) == item['sha256']
    assert sha(Path(p['run']) / 'summary.json') == p['summary_sha256']
    assert sha(Path(p['audit']) / 'summary.json') == p['audit_summary_sha256']
    parent = read(p['parent_inventory'])
    assert sha(p['parent_inventory']) == p['parent_inventory_sha256']
    assert parent['cohort_manifest_sha256'] == p['cohort_manifest_sha256']
    canonical, reproduced = read(Path(p['run']) / 'summary.json'), read(Path(p['audit']) / 'summary.json')
    assert reproduced['canonical_reset_actual_controls_states_scores_cache_and_cold_independently_byte_exact']
    assert reproduced['source_summary_sha256'] == p['summary_sha256']
    assert reproduced['source_actual_sha256'] == p['actual_sha256'] == sha(Path(p['run']) / 'actual.npz')
    for key, value in canonical.items():
        assert reproduced[key] == value
    from audit_spatial_synthesis_models import load
    from spatialpole.native_mpc import exact_bytes
    actual, replica = load(Path(p['run']) / 'actual.npz'), load(Path(p['audit']) / 'actual.npz')
    assert actual.keys() == replica.keys()
    for key in actual:
        exact_bytes(actual[key], replica[key])
    return p


def prefix_prior(folder, cm, m, development, index, path):
    # Independent case ledgers share the same historical development baseline.
    # Only the final cohort auditor sums their increments; ordinal is not coverage.
    assert path is None and 0 <= index < 32
    return dict(total_new_target_preparation_native_steps=development['total_new_target_preparation_native_steps'],
        total_new_target_preparation_native_steps_bounds=development['total_new_target_preparation_native_steps_bounds'])


def registered(folder):
    """Read a consumed original observation, without another reset or draw."""
    folder = Path(folder).resolve()
    rm = read(folder / 'frozen_manifest.json')
    assert (folder / 'completed.json').exists() and rm['registration_protocol'] == REGISTRATION
    run = Path(rm['cohort'])
    bank, runtime, cm, m, development = cohort(run)
    assert sha(run / 'frozen_manifest.json') == rm['cohort_manifest_sha256']
    index, seed = rm['fresh_index'], rm['seed']
    assert 0 <= index < 32 and FRESH[index] == seed and rm['controller_identity'] == cm['controller_identity']
    consumed = read(run / 'consumed_cases.json')['seeds']
    assert consumed == FRESH[:len(consumed)] and consumed[:index + 1] == FRESH[:index + 1]
    claim = run / 'case_claims' / f'{seed}.json'
    assert sha(claim) == rm['case_claim_sha256']
    claimed = read(claim)
    assert claimed['seed'] == seed and Path(claimed['registration']).resolve() == folder
    assert claimed['cohort_manifest_sha256'] == rm['cohort_manifest_sha256']
    prior = prefix_prior(run, cm, m, development, index, rm['previous_witness_inventory'])
    assert prior['total_new_target_preparation_native_steps'] == rm['previous_closed_native_steps']
    assert prior['total_new_target_preparation_native_steps_bounds'] == rm['previous_closed_native_bounds']
    assert rm['previous_witness_inventory'] is None and rm['previous_witness_inventory_sha256'] is None
    assert rm['noise'] == .02 and rm['conditioning_steps'] == 0 and rm['new_noisy_samples'] == 1
    assert sha(folder / 'registered_initial.npz') == rm['registered_initial_sha256']
    import numpy as np
    with np.load(folder / 'registered_initial.npz') as z:
        initial = z['state']
    assert initial.shape == (102,) and initial.dtype == np.float64 and np.isfinite(initial).all()
    assert hashlib.sha256(initial.tobytes()).hexdigest() == rm['observed_initial_bytes_sha256']
    return bank, runtime, cm, rm, initial, prior
