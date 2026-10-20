"""Read-only qualification/media/reproduction gate, before packaging or network writes."""
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from spatialpole import Config
from eleven_synthesis_protocol import read, sha, seeds, controller_identity, validate_parent


def planning_reproduction_proof(qualification, reproduction):
    """Bind the separate full planning parity receipt to the actual first case."""
    q, repro = Path(qualification), Path(reproduction); case = q/'seed20263516'
    inventory = Path(__file__).resolve().parents[1]/'output/eleven_first_reserved_reproduction_inventory_001.json'
    proof = read(inventory)
    assert proof['passed'] and not proof['qualified']
    assert proof['fresh_noisy_samples'] == proof['reserved_cases_consumed'] == 0
    assert proof['controller_identity'] == controller_identity(read(q/'frozen_manifest.json'))
    assert proof['source_qualification_manifest_sha256'] == sha(q/'frozen_manifest.json')
    assert proof['original_case_summary_sha256'] == sha(case/'summary.json')
    assert proof['reproduction_summary_sha256'] == sha(repro/'summary.json')
    assert proof['reproduction_summary']['seed'] == 20263516
    assert proof['new_native_steps'] == read(repro/'summary.json')['inclusive_native_steps']
    assert proof['actual_all_arrays_byte_exact']
    rows = proof['planning_file_parity']; names = {row['path'] for row in rows}
    assert len(names) == len(rows) and names
    assert names == {p.relative_to(case/'planning').as_posix() for p in (case/'planning').rglob('*.npz')}
    assert names == {p.relative_to(repro/'planning').as_posix() for p in (repro/'planning').rglob('*.npz')}
    for row in rows:
        old, new = case/'planning'/row['path'], repro/'planning'/row['path']
        assert sha(old) == row['original_sha256'] and sha(new) == row['reproduction_sha256']
        same = row['original_sha256'] == row['reproduction_sha256']
        assert row['all_arrays_byte_exact'] and same == row['file_byte_exact']
        if not same:
            with np.load(old) as x, np.load(new) as y:
                assert set(x.files) == set(y.files)
                for name in x.files:
                    xx, yy = x[name], y[name]
                    assert xx.dtype == yy.dtype and xx.shape == yy.shape and xx.tobytes() == yy.tobytes()
    return dict(inventory_sha256=sha(inventory), planning_files=len(rows),
        all_arrays_byte_exact=True, all_files_byte_exact=all(row['file_byte_exact'] for row in rows))


def qualified_record(qualification, cohort_audit, runtime_receipt, reproduction, video, video_qa):
    q, gate, rr, repro, video, quality = map(Path,
        (qualification, cohort_audit, runtime_receipt, reproduction, video, video_qa))
    assert (q/'completed.json').exists(), 'Qualification is incomplete'
    s = read(q/'summary.json')
    assert s['cohort'] == 'qualification32' and s['qualification_attempt']
    assert s['qualification_gate_passed'] and s['cohort_finished'] and not s['stopped_on_failure']
    assert s['passes'] == len(s['results']) == 32
    assert [r['result']['seed'] for r in s['results']] == seeds('qualification32')
    assert s['reserved_cases_consumed'] == 32 and s['reserved_start'] == 20263516
    assert not s['inherited_cases'] and s['actual_benchmark_episodes'] == 32
    m = read(q/'frozen_manifest.json'); identity = controller_identity(m)
    assert s['controller_identity'] == m['controller_identity'] == identity
    reservation = read(q.parent/'reserved_11_immediate_20263516.json')
    assert Path(reservation['run']).resolve() == q.resolve()
    assert reservation['seeds'] == seeds('qualification32') and reservation['controller_identity'] == identity
    assert reservation['frozen_manifest_sha256'] == sha(q/'frozen_manifest.json')
    assert (gate/'completed.json').exists(); g = read(gate/'summary.json')
    assert g['passed'] and g['qualified'] and g['qualification'] and g['cohort'] == 'qualification32'
    assert g['source_summary_sha256'] == sha(q/'summary.json')
    assert g['frozen_manifest_sha256'] == sha(q/'frozen_manifest.json') and g['controller_identity'] == identity
    assert g['case_seeds'] == seeds('qualification32')
    args = read(q/'inputs.json')['args']
    development = Path(args['parent_run']); development_audit = Path(args['parent_audit'])
    ds, dg, dm = validate_parent(development, development_audit, 'qualification32')
    assert controller_identity(dm) == identity
    assert m['source_audit_sha256'] == sha(development_audit/'summary.json')
    assert sha(q/'parent_summary.json') == sha(development/'summary.json')
    assert sha(q/'parent_integrity_audit.json') == sha(development_audit/'summary.json')
    for name, digest in m['source_sha256'].items():
        assert sha(q/'frozen_package/spatialpole'/name) == digest
    for name, digest in m['helper_sha256'].items():
        assert sha(q/'helpers'/name) == digest
    for name, digest in m['bundle_sha256'].items():
        assert sha(q/'reference'/name) == digest
    cases = []; audit_native = 0
    for row in s['results']:
        r = row['result']; seed = r['seed']; case = q/f'seed{seed}'
        assert (case/'completed.json').exists()
        assert r['success'] and r['replay_exact'] and r['steps'] == 1500
        assert r['initialization'] == 'hanging' and r['angle_noise'] == .02
        assert r['native_steps'] == r['proof_native_steps'] == 6000
        assert r['best_hold_seconds'] >= 5. and r['plane_margin'] >= 0. and not r['physics_warning']
        assert r['max_force_norm'] <= 80.
        policy = r['policy_metadata']
        assert policy['park_seconds'] == 0. and policy['physical_velocities'] == 35
        assert not policy['hardware_realtime_claim']
        assert row['inclusive_native_steps'] == row['planning_native_steps']+12000
        assert policy['predictor_native_steps'] == row['planning_native_steps']
        assert row['frozen_manifest_sha256'] == sha(q/'frozen_manifest.json')
        audit_ref = g['case_audits'][str(seed)]; audit = Path(audit_ref['path'])
        assert sha(audit/'summary.json') == audit_ref['summary_sha256']
        proof = read(audit/'summary.json')
        assert proof['passed'] and proof['actual_cold_byte_exact']
        assert proof['case_summary_sha256'] == sha(case/'summary.json')
        assert proof['actual_sha256'] == sha(case/'actual.npz')
        assert proof['source_native_steps'] == row['inclusive_native_steps']
        audit_native += proof['native_steps']
        cases.append(dict(seed=seed, tape_sha256=sha(case/'actual.npz'),
            case_summary_sha256=sha(case/'summary.json'), audit_summary_sha256=sha(audit/'summary.json'),
            best_hold_seconds=r['best_hold_seconds'], plane_margin=r['plane_margin'],
            max_force_norm=r['max_force_norm'], planning_native_steps=row['planning_native_steps'],
            inclusive_native_steps=row['inclusive_native_steps'], audit_native_steps=proof['native_steps']))
    assert sum(r['inclusive_native_steps'] for r in s['results']) == s['inclusive_native_steps']
    runtime = read(rr); archive = Path(runtime['archive'])
    assert runtime['passed'] and runtime['all_zip_crc_verified'] and runtime['all_payload_hashes_verified']
    assert runtime['controller_identity'] == identity and sha(archive) == runtime['sha256']
    extracted = Path(runtime['extracted_copy']); rm = read(extracted/'manifest.json')
    assert rm['controller_identity'] == identity and rm['solver_receipt_sha256'] == m['solver_receipt_sha256']
    assert rm['study_manifest_sha256'] == sha(extracted/'study/frozen_manifest.json')
    for name, description in rm['files'].items():
        assert sha(extracted/name) == description['sha256']
    assert (repro/'completed.json').exists(); rp = read(repro/'summary.json')
    assert rp['passed'] and not rp['query_parity_only'] and rp['seed'] == 20263516
    assert rp['arrays_byte_exact'] and rp['expected_case']['all_arrays_byte_exact']
    assert rp['fresh_noisy_samples'] == rp['reserved_cases_consumed'] == 0
    assert rp['study_manifest_sha256'] == rm['study_manifest_sha256']
    assert rp['expected_case']['actual_tape_sha256'] == cases[0]['tape_sha256']
    assert rp['expected_case']['case_summary_sha256'] == cases[0]['case_summary_sha256']
    assert rp['planning_native_steps'] == cases[0]['planning_native_steps']
    assert rp['inclusive_native_steps'] == cases[0]['inclusive_native_steps']
    planning_integrity = planning_reproduction_proof(q, repro)
    assert rp['result']['success'] and rp['result']['replay_exact'] and rp['result']['steps'] == 1500
    assert Path(rp['imported_package']).is_relative_to(extracted/'study/frozen_package')
    assert Path(rp['imported_synthesis']).is_relative_to(extracted/'study/helpers')
    assert Path(rp['imported_solver']).is_relative_to(extracted/'numerical-runtime/site')
    vm, vq = read(video.with_suffix('.json')), read(quality)
    assert vm['qualified'] and not vm['development_preview'] and vm['full_video']
    assert vm['frames'] == 1501 and vm['fps'] == 50 and vm['simulation_seconds'] == 30.
    assert vm['park_seconds'] == 0. and vm['seed'] == 20263516
    assert vm['source_tape_sha256'] == cases[0]['tape_sha256']
    assert vm['source_case_audit_sha256'] == cases[0]['audit_summary_sha256']
    assert vm['frozen_manifest_sha256'] == sha(q/'frozen_manifest.json')
    assert vm['controller_identity'] == identity
    assert vq['passed'] and vq['video_sha256'] == vm['sha256'] == sha(video)
    assert vq['frames'] == 1501 and vq['fps'] == 50 and vq['simulation_seconds'] == 30.
    assert set((0, 170, 325, 1500)).issubset(vq['inspected_frames'])
    assert all(vq['visual_checks'][key] for key in
        ('complete_chain_visible', 'legible_labels', 'explicit_qualification_status', 'xy_inset_visible'))
    assert not vq.get('development_preview', False)
    with np.load(q/'seed20263516/actual.npz') as first:
        config = json.loads(str(first['config']))
        assert first['states'].shape == (1501, 81) and first['actions'].shape == (1500, 2)
    assert config == asdict(Config(n_links=11))
    planning_times = [r['planning_wall_seconds'] for r in s['results']]
    return dict(schema=1, tag='2d-cartpole-11-link-immediate-v1', links=11,
        benchmark='2D XY cart / unrestricted passive spatial ball joints',
        variant='observed-state full-horizon native synthesis and feedback; zero hanging preparation',
        qualified=True, reserved_passes=32, development_passes=45, park_seconds=0.,
        minimum_hold_seconds=min(c['best_hold_seconds'] for c in cases),
        minimum_plane_margin=min(c['plane_margin'] for c in cases),
        maximum_force_norm=max(c['max_force_norm'] for c in cases),
        config=config, velocity_coordinates=35,
        native_tangent_coordinates=70, planning_coordinates=59,
        controller_identity=identity, hardware_realtime_claim=False, learned_model=False,
        force_gust_qualified=False, highest_overall_qualified_links=14,
        planning_wall_seconds=dict(minimum=min(planning_times), maximum=max(planning_times),
                                   mean=sum(planning_times)/len(planning_times)),
        qualification_cases=cases, qualification_native_steps=s['inclusive_native_steps'],
        independent_case_audit_native_steps=audit_native, first_reserved_reproduction_native_steps=rp['inclusive_native_steps'],
        source_sha256=m['source_sha256'], solver_receipt_sha256=m['solver_receipt_sha256'],
        qualification_summary_sha256=sha(q/'summary.json'), qualification_audit_sha256=sha(gate/'summary.json'),
        development_summary_sha256=sha(development/'summary.json'), development_audit_sha256=sha(development_audit/'summary.json'),
        runtime_archive=dict(name=archive.name, path=str(archive.resolve()), sha256=sha(archive), bytes=archive.stat().st_size),
        video=dict(name=video.name, path=str(video.resolve()), sha256=sha(video), bytes=video.stat().st_size,
                   frames=1501, fps=50, simulation_seconds=30.),
        video_quality_sha256=sha(quality), reproduction_summary_sha256=sha(repro/'summary.json'),
        reproduction_planning_integrity=planning_integrity,
        runtime_receipt_sha256=sha(rr))
