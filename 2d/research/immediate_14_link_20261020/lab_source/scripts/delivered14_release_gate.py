"""Join real fresh32, extracted-runtime, dev margin and inspected-video evidence.

No output, Git mutation, sampling, dynamics, controller or optimization here.
"""
import json
from pathlib import Path
from qualified_delivered_source14_parallel import first_qualified,read,sha


ROOT=Path(__file__).resolve().parents[1]
TAG='2d-cartpole-14-link-immediate-v1'


def asset(path,name=None):
    path=Path(path).resolve()
    return dict(name=name or path.name,path=str(path),bytes=path.stat().st_size,sha256=sha(path))


def qualified_record(proof,archive_run,reproduction,margin,video,video_qa):
    # This first check refuses incomplete/development evidence before any other
    # path needs to exist, and before expensive hashing or publication work.
    admission=first_qualified(proof)
    archive_run,reproduction,margin,video,video_qa=map(lambda p:Path(p).resolve(),
        (archive_run,reproduction,margin,video,video_qa))
    identity=admission['controller_identity'];cohort=Path(admission['cohort'])
    am=read(archive_run/'summary.json');members=read(archive_run/'member-manifest.json')
    assert (archive_run/'completed.json').exists() and am['passed']
    assert am['controller_identity']==members['controller_identity']==identity
    assert am['all_member_crc_and_sha256_checked'] and am['source_hardlinks_not_used_for_extraction']
    assert am['original_scientific_and_staged_view_paths_blocked'] and am['extracted_identity_and_relative_materialization_passed']
    assert sha(archive_run/'member-manifest.json')==am['member_manifest_sha256']
    assert len(members['archives'])==5
    runtime=[]
    for index,row in enumerate(members['archives']):
        path=archive_run/'archives'/row['name']
        assert path.stat().st_size==row['bytes']<2**31 and sha(path)==row['sha256']
        name='2d-cartpole-14-link-immediate-runtime'+('' if index==0 else f'-data-part{index}')+'.zip'
        runtime.append(asset(path,name))
    rr=read(reproduction/'summary.json');job=read(reproduction/'job.json')
    assert (reproduction/'completed.json').exists() and rr['passed'] and rr['source_qualified']
    assert rr['controller_identity']==job['controller_identity']==identity
    assert rr['seed']==job['first_seed']==admission['first_seed']==20263816
    assert rr['source_job_sha256']==sha(reproduction/'job.json')
    assert rr['source_worker_summary_sha256']==sha(reproduction/'execution/summary.json')
    assert rr['source_qualification_summary_sha256']==job['qualification_summary_sha256']==admission['proof_summary_sha256']
    assert job['source_archive_member_manifest_sha256']==am['member_manifest_sha256']
    assert job['worker_sha256']==sha(reproduction/'driver.py')
    assert rr['source_actual_sha256']==admission['witness']['actual_sha256']
    assert rr['actual_sha256']==sha(reproduction/'execution/actual.npz')
    for key in ('all_regenerated_planning_npz_arrays_byte_exact','source_numerical_audit_inherited_after_literal_packet_parity',
                'full_native_episode_and_independent_cold_byte_exact','all_raw_delivered_controls_states_tilts_byte_exact',
                'original_and_staged_scientific_paths_blocked','expected_results_blocked_until_prepare_finished',
                'callbacks_use_real_native_predictions_without_cache'):
        assert rr[key] is True,key
    assert rr['park_seconds']==0 and rr['physical_velocities']==44 and rr['best_hold_seconds']>=5.
    assert rr['preparation_native_steps']<=9072324 and rr['actual_native_steps']==rr['cold_native_steps']==6000
    assert rr['new_noisy_samples']==rr['reserved_cases_consumed']==0
    assert rr['planning_wall_seconds']>0 and rr['total_worker_wall_seconds']>=rr['planning_wall_seconds']
    ms=read(margin/'summary.json');mi=read(margin/'inputs.json')
    assert ms['completed'] and not ms['qualified'] and ms['controller_identity']==identity
    assert ms['referenced_episodes']==480 and ms['native_steps']==ms['new_labels']==0
    grid=Path(mi['run']);gs=read(grid/'summary.json')
    assert gs['passed'] and not gs['smoke'] and gs['episodes']==480
    assert gs['controller_identity']==identity and sha(grid/'summary.json')==mi['source_summary_sha256']==ms['source_summary_sha256']
    assert sha(grid/'rows.jsonl')==mi['source_rows_sha256']
    assert mi['driver_sha256']==sha(ROOT/'scripts/analyze_delivered14_margin.py')
    preserved=read(ROOT/'research/sources/immediate-disturbance-margin-2026-10-19/manifest.json')
    assert mi['estimator_sha256']==preserved['files']['tools/analyze_immediate_margin.py']['sha256']
    assert ms['sigma50_bracketed'] and ms['sigma50_N'] is not None
    assert ms['force']['firth']['usable'] and ms['force']['firth']['sigma50_N']==ms['sigma50_N']
    rows=[json.loads(line) for line in (grid/'rows.jsonl').read_text().splitlines()]
    assert len(rows)==480 and len({(r['seed'],r['sigma'],r['mass_delta']) for r in rows})==480
    assert set(ms['force']['seeds'])==set(range(20261416,20261436))
    for cell in ms['force']['cells']:
        sub=[r for r in rows if r['sigma']==cell['sigma_N'] and r['mass_delta']==0]
        assert len(sub)==cell['trials']==20 and sum(r['success'] for r in sub)==cell['successes']
    for cell in ms['mass']['cells']:
        sub=[r for r in rows if r['sigma']==0 and r['mass_delta']==cell['mass_delta']]
        assert len(sub)==cell['trials']==20 and sum(r['success'] for r in sub)==cell['successes']
    vm,vq=read(video.with_suffix('.json')),read(video_qa)
    assert vm['qualified'] and vm['full_video'] and not vm['development_preview']
    assert vm['links']==14 and vm['seed']==admission['first_seed'] and vm['park_seconds']==0
    assert vm['frames']==1501 and vm['fps']==50 and vm['simulation_seconds']==30
    assert vm['controller_identity']==identity and vm['source_actual_sha256']==admission['witness']['actual_sha256']
    assert vm['source_cohort_audit_sha256']==admission['proof_summary_sha256']
    assert vm['source_cohort_manifest_sha256']==admission['cohort_manifest_sha256']
    assert vm['sha256']==vq['video_sha256']==sha(video)
    assert vq['passed'] and vq['manual_visual_qa_complete'] and vq['all1501_encoded_frames_decoded']
    for key in ('complete_chain_visible','legible_labels','explicit_qualification_status','xy_inset_visible'):
        assert vq['visual_checks'][key] is True
    for frame in (0,150,257,350,750,1500):
        inspected=vq['inspected_encoded_frames'][str(frame)]
        assert Path(inspected['path']).resolve()==video.with_name(video.stem+f'-frame{frame:04d}.png')
        assert sha(inspected['path'])==inspected['sha256']
    framing=video.parent/'framing_proof.json';fp=read(framing)
    assert sha(framing)==vm['framing_proof_sha256'] and fp['passed'] and fp['native_steps']==0
    assert vq['contact_sheet_inspected'] and vq['contact_sheet_sha256']==sha(video.parent/'contact-sheet.png')
    for frame in (fp['highest_frame'],fp['lowest_frame']):
        inspected=vq['inspected_encoded_frames'][str(frame)]
        assert Path(inspected['path']).resolve()==video.with_name(video.stem+f'-frame{frame:04d}.png')
        assert sha(inspected['path'])==inspected['sha256']
    assert fp['frames']==1501 and fp['cart_all8vertices_included'] and fp['overlay_clearance_pixels']>=8
    assert fp['capsule_and_joint_radius_bound_m']>=.035
    import numpy as np
    with np.load(video.parent/'projected_body_bounds.npz') as z:bounds=z['bounds']
    assert bounds.shape==(1501,4) and np.isfinite(bounds).all()
    assert bounds[:,0].min()>=25 and bounds[:,1].max()<=1015
    assert bounds[:,2].min()>=153 and bounds[:,3].max()<=629
    assert float(bounds[:,2].min())==fp['minimum_projected_y'] and float(bounds[:,3].max())==fp['maximum_projected_y']
    assert sha(video.parent/'projected_body_bounds.npz')==vq['projected_body_bounds_sha256']
    assert sha(video.parent/'render_driver.py')==vm['render_driver_sha256']==sha(ROOT/'scripts/render_qualified_delivered14_parallel.py')
    costs=admission['per_episode_costs'];assert len(costs)==32
    qualified=read(Path(admission['proof'])/'summary.json');cm=read(cohort/'frozen_manifest.json')
    dev=read(Path(cm['development_proof'])/'summary.json')
    return dict(tag=TAG,links=14,highest_immediate_links=14,qualified=True,controller_identity=identity,
        runtime_class='offline per-episode planning with simulation paused; full-state simulator feedback',
        fresh_passes=32,fresh_cases=32,development_passes=45,conditioning_steps=0,physical_velocities=44,
        minimum_hold_seconds=qualified['minimum_hold_seconds'],minimum_plane_margin=qualified['minimum_plane_margin'],
        maximum_force_norm=qualified['maximum_force_norm'],first_seed=admission['first_seed'],
        source_cohort_manifest_sha256=admission['cohort_manifest_sha256'],
        qualification_proof_sha256=admission['proof_summary_sha256'],
        extracted_resynthesis_summary_sha256=sha(reproduction/'summary.json'),
        source_archive_member_manifest_sha256=am['member_manifest_sha256'],
        measured_sigma50_N=ms['sigma50_N'],sigma50_bootstrap95_N=ms['force']['bootstrap']['interval95_N'],
        mass_tolerance=ms['mass_tolerance'],mass_passing_magnitudes=ms['mass']['passing_magnitudes'],
        minimum_tested_mass=min(abs(cell['mass_delta']) for cell in ms['mass']['cells']),
        margin_mode='development20 guard-bypass diagnostic; iid Gaussian per axis on every command, both-sign uniform mass',
        margin_analysis_sha256=sha(margin/'summary.json'),development_margin_tiers_met=ms['development_margin_tiers_met'],
        per_episode_costs=costs,first_extracted_planning_native_steps=rr['preparation_native_steps'],
        first_extracted_planning_wall_seconds=rr['planning_wall_seconds'],
        referenced_development_preparation_native_bounds=dev['total_new_target_preparation_native_steps_bounds'],
        fresh_inclusive_native_steps=qualified['referenced_fresh_native_steps'],
        runtime_archives=runtime,video=asset(video),video_qa_sha256=sha(video_qa),
        scope='fresh32 original-task qualification plus extracted resynthesis and separate measured development margins; no hardware real-time claim')
