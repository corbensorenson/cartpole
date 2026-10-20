"""Admission for post-qualification media; never accepts development evidence."""
import hashlib
import json
from pathlib import Path


FRESH = list(range(20263816, 20263848))


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class NotQualified(RuntimeError):
    pass


def first_qualified(proof):
    """Bind the whole fresh cohort, then return its fixed first reserved case."""
    proof = Path(proof).resolve()
    summary = read(proof / 'summary.json')
    if not ((proof / 'completed.json').exists() and summary.get('passed') is True
            and summary.get('qualified') is True and summary.get('task_success') is True
            and summary.get('links') == 14 and summary.get('cohort') == 'qualification32'
            and summary.get('noisy_passes') == 32 and summary.get('passes') == 32):
        raise NotQualified('A completed original-noise fresh32 qualification proof is required; development evidence cannot produce qualified media')
    assert proof.name == 'qualification-proof'
    run = proof.parent
    execution = read(run / 'execution/summary.json')
    assert (run / 'execution/completed.json').exists() and execution['qualified'] and execution['task_success']
    assert execution['status'] == 'all32_fresh_cases_and_cohort_proof_passed' and execution['all32_cases_completed']
    assert summary['case_seeds'] == FRESH and [row['seed'] for row in summary['results']] == FRESH
    assert summary['source_reserved_cases_consumed'] == 32 and summary['native_steps'] == 0
    assert summary['new_noisy_samples'] == summary['reserved_cases_consumed'] == 0
    from delivered_fresh_context14_parallel import cohort, previous
    bank, runtime, cm, manifest, development = cohort(run)
    identity = cm['controller_identity']
    assert summary['controller_identity'] == identity
    assert execution['links'] == 14 and execution['max_worker_processes'] <= 6
    assert sha(run / 'frozen_manifest.json') == summary['source_cohort_manifest_sha256'] == execution['source_cohort_manifest_sha256']
    assert sha(proof / 'cohort-manifest.json') == summary['source_cohort_manifest_sha256']
    assert sha(proof / 'driver.py') == manifest['helper_sha256']['audit_fresh_delivered14_cohort_parallel.py']
    assert read(proof / 'inputs.json')['source_sha256'] == cm['source_sha256']
    assert read(proof / 'inputs.json')['runtime'] == manifest['original_runtime']
    assert read(run / 'consumed_cases.json')['seeds'] == FRESH
    assert [row['seed'] for row in execution['results']] == FRESH
    assert execution['known_closed_native_ledger'] == summary['total_new_target_preparation_native_steps']
    assert execution['known_closed_native_bounds'] == summary['total_new_target_preparation_native_steps_bounds']
    first = None
    for index, (row, owned) in enumerate(zip(summary['results'], execution['results'])):
        seed = FRESH[index]
        inventory = run / 'cases' / str(seed) / 'witness_inventory.json'
        assert Path(row['witness_inventory']).resolve() == Path(owned['witness_inventory']).resolve() == inventory
        assert sha(inventory) == row['witness_inventory_sha256'] == owned['witness_inventory_sha256']
        witness = previous(inventory, cm, seed + 1)
        assert witness['actual_sha256'] == row['actual_sha256']
        assert sha(witness['parent_inventory']) == row['preparation_inventory_sha256']
        assert witness['fresh_case_ordinal'] == index + 1 and witness['result'] == owned['result']
        result = witness['result']
        assert row['commands'] == result['steps'] == 1500 and row['seconds'] == 30.
        assert row['best_hold_seconds'] == result['best_hold_seconds'] >= 5.
        assert row['minimum_native_substep_margin'] == result['plane_margin'] >= 0.
        assert row['peak_delivered_force'] == result['max_force_norm'] <= 80.
        assert witness['cohort_manifest_sha256'] == summary['source_cohort_manifest_sha256']
        if index == 0:
            first = witness
    assert first is not None and first['seed'] == FRESH[0]
    return dict(cohort=str(run),proof=str(proof),proof_summary_sha256=sha(proof / 'summary.json'),
        cohort_manifest_sha256=summary['source_cohort_manifest_sha256'],controller_identity=identity,
        first_seed=FRESH[0],witness=first,qualified=True,new_noisy_samples=0,
        first_case_cost=execution['results'][0],
        per_episode_costs=[dict(seed=row['seed'],actor_native_steps=row['planning_actor_native_steps'],
            audit_native_steps=row['planning_audit_native_steps'],total_native_steps=row['incremental_native_steps'],
            total_prepare_audit_witness_wall_seconds=row['wall_seconds']) for row in execution['results']],
        runtime_class='offline per-episode planning with simulation paused')
