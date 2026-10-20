"""Admit all45, freeze adapters, then claim both legacy fresh namespaces."""
import argparse
import json
from pathlib import Path
import shutil
from delivered_fresh_gate14 import require_all45
from delivered_fresh_context14_parallel import (ROOT, EXPERIMENTS, FRESH, PROTOCOL, RESERVATIONS,
    REQUIRED, CASE_NATIVE_CAP, MAX_COHORT_WALL, read, sha, serialized_args, require_runtime)


def main():
    p = argparse.ArgumentParser()
    for name in ('proof', 'candidate', 'out'):
        p.add_argument('--' + name, required=True, type=Path)
    a = p.parse_args()
    # Scheduling is separate from scientific admission. Completing all45 does
    # not lift the October 19 margin-review hold. Missing policy fails closed.
    policy_path = ROOT / 'research/management/launch_policy.json'
    if not policy_path.is_file():
        raise SystemExit('Fresh qualification held: local step-back launch policy is required')
    policy = json.loads(policy_path.read_text(encoding='utf-8'))
    gate = policy['fresh_qualification']
    if gate['status'] != 'active' or not gate.get('review_note'):
        raise SystemExit('Fresh qualification held pending a recorded step-back decision: ' + gate['reason'])
    review_note = (ROOT / gate['review_note']).resolve()
    review_note.relative_to(ROOT.resolve())
    if not review_note.is_file() or sha(review_note) != gate.get('review_note_sha256'):
        raise SystemExit('Fresh qualification held: review note is missing or its digest differs')
    # No output directory, reservation, numerical imports or observation exists
    # before a real, fully bound 45-state proof has passed this admission.
    admission = require_all45(a.proof, a.candidate)
    assert a.out.resolve().parent == EXPERIMENTS.resolve() and not a.out.exists()
    assert all(not (EXPERIMENTS / name).exists() for name in RESERVATIONS)
    from spatialpole.experiment import freeze_inputs, write_json
    from delivered_candidate14 import candidate
    from stage_six_source_synthesis14 import helper_closure
    bank, runtime, cm = candidate(a.candidate)
    original_runtime = read(a.candidate / 'inputs.json')['runtime']
    require_runtime(original_runtime)
    helpers = helper_closure(set(REQUIRED))
    freeze_inputs(a.out, serialized_args(a))
    assert read(a.out / 'inputs.json')['source_sha256'] == cm['source_sha256']
    assert read(a.out / 'inputs.json')['runtime']['thread_environment'] == {}
    assert read(a.out / 'inputs.json')['runtime'] == original_runtime
    (a.out / 'helpers').mkdir()
    for name in helpers:
        shutil.copyfile(ROOT / 'scripts' / name, a.out / 'helpers' / name)
    shutil.copyfile(ROOT / 'research/delivered_fresh14_parallel_protocol.md', a.out / 'prospective_protocol.md')
    manifest = dict(protocol=PROTOCOL, config=cm['config'], controller_identity=cm['controller_identity'],
        candidate=str(a.candidate.resolve()), candidate_manifest_sha256=admission['candidate_manifest_sha256'],
        development_proof=str(a.proof.resolve()), development_states=45, original_runtime=original_runtime,
        development_proof_summary_sha256=admission['development_proof_summary_sha256'],
        development_proof_manifest_sha256=admission['development_proof_manifest_sha256'],
        helper_sha256={name: sha(a.out / 'helpers' / name) for name in helpers}, source_sha256=cm['source_sha256'],
        prospective_protocol_sha256=sha(a.out / 'prospective_protocol.md'), seeds=FRESH,
        per_case_native_cap=CASE_NATIVE_CAP, total_native_cap=32 * CASE_NATIVE_CAP,
        max_wall_seconds=MAX_COHORT_WALL, max_actor_native_steps=9072324, max_audit_native_steps=384324,
        max_witness_native_steps=12000, max_reproduction_native_steps=12000,
        core_actor_wall_cap=15000, core_audit_wall_cap=7200, core_witness_wall_cap=240,
        fresh_qualification_activated=True, qualified=False, source_bank_expands=False,
        stop_on_any_failed_gate=True, max_worker_processes=6, independent_case_ledgers=True, claim_before_reset=True, prepare_receives_observed_state_not_seed=True)
    write_json(a.out / 'frozen_manifest.json', manifest)
    reservation = dict(run=str(a.out.resolve()), seeds=FRESH, controller_identity=cm['controller_identity'],
        frozen_manifest_sha256=sha(a.out / 'frozen_manifest.json'), protocol=PROTOCOL)
    # A partial claim is retained and blocks activation after any interruption;
    # no case can sample until both files and the completed freeze agree.
    for name in RESERVATIONS:
        with (EXPERIMENTS / name).open('x', encoding='utf-8') as stream:
            json.dump(reservation, stream, indent=2)
    (a.out / 'case_claims').mkdir()
    write_json(a.out / 'consumed_cases.json',dict(seeds=[],controller_identity=cm['controller_identity']))
    write_json(a.out / 'summary.json',dict(passed=True,qualified=False,task_success=False,native_steps=0,
        controller_identity=cm['controller_identity'],fresh_labels_opened=0,reserved_cases_consumed=0,
        cohort_namespace_claimed=True,scope='prospective qualification freeze; no fresh reset or task result'))
    write_json(a.out / 'completed.json',dict(status='completed'))


if __name__ == '__main__':
    main()
