"""Read-only gates for reused development evidence and fresh spatial cohorts."""
import hashlib
import json
from pathlib import Path
import numpy as np
from spatialpole import Config
from spatial_observed_synthesis import planning_budget
from eleven_synthesis_protocol import read, sha

ALGORITHM=('spatial_observed_synthesis.py','spatial_synthesis_geometry.py',
    'eleven_value_sequence.py','donor_conic_step.py','verify_eleven_sequence_refinement.py',
    'eleven_sequence_qp.py','trajectory_expert_support.py')
PILOT=list(range(20261116,20261128))
ADDITIONAL=list(range(20261416,20261448))


def reserved(links):
    start=20262416+100*links
    return list(range(start,start+32))


def manifest_check(run):
    run=Path(run);m=read(run/'frozen_manifest.json');config=Config(n_links=m['links'])
    budget=planning_budget(config)
    assert m.get('max_planning_native',budget)==budget
    assert m.get('max_case_native',budget+12000)==budget+12000
    payload=dict(algorithm_sha256={name:m['helper_sha256'][name] for name in ALGORITHM},
        source_sha256=m['source_sha256'],bundle_sha256=m['source_bundle_sha256'],
        solver_receipt_sha256=m['solver_receipt_sha256'],max_planning_native=budget,max_case_native=budget+12000)
    digest=lambda p:hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest()
    allowed={digest(dict(links=m['links'],**payload))}
    if m['links']==12:allowed.add(digest(payload))  # Original frozen pilot's identity format.
    assert m['controller_identity'] in allowed
    for name,value in m['helper_sha256'].items():assert sha(run/'helpers'/name)==value
    for name,value in m['source_sha256'].items():assert sha(run/'frozen_package/spatialpole'/name)==value
    for name,value in m['inventory_sha256'].items():assert sha(run/'inventories'/name)==value
    assert sha(Path(m['runtime'])/'receipt.json')==m['solver_receipt_sha256']
    return m,config


def nominal_equivalence(run,m):
    """Reuse the existing physical source law and its complete planner query proof."""
    source=read(run/'inventories/source.json');port=read(run/'inventories/port.json')
    assert source['passed'] and port['passed']
    s,p=source['summary'],port['summary'];r=s['result']
    assert s['nominal_source_law_admitted'] and s['zero_residual_controls']
    assert s['source_model_inventory_sha256']==sha(run/'inventories/model.json')
    assert p['source_inventory_sha256']==sha(run/'inventories/source.json')
    assert r['seed']==20261016 and r['angle_noise']==0. and r['initialization']=='hanging'
    assert r['success'] and r['steps']==1500 and r['replay_exact'] and r['best_hold_seconds']>=5.
    assert r['native_steps']==r['proof_native_steps']==6000
    assert r['policy_metadata']['park_seconds']==0. and r['policy_metadata']['physical_velocities']==2+3*m['links']
    assert p['passed'] and p['native_steps']==0 and p['callbacks_byte_exact']==1500
    assert p['source_reused_without_design'] and source['source_sha256']==m['source_sha256']
    for name,value in p['helper_sha256'].items():assert m['helper_sha256'][name]==value
    physical=Path(source['run']);query=Path(port['run'])
    assert sha(physical/'summary.json')==source['summary_sha256']
    assert sha(query/'summary.json')==port['summary_sha256']
    for name in ('input_nominal.npz','coordinates.npz','actual.npz','frames.npz'):
        assert sha(physical/name)==source['files'][name]['sha256']
    assert sha(physical/'actual.npz')==m['source_bundle_sha256']['actual.npz']
    return dict(passed=True,new_native_steps=0,source=str(physical),query=str(query),
        source_summary_sha256=source['summary_sha256'],query_summary_sha256=port['summary_sha256'],
        scope='existing full physical source-law episode and cold replay plus 1500 exact planner callbacks; no repeated nominal integration')


def case_check(row,m,qualification=False):
    case,audit=Path(row['run']),Path(row['audit'])
    assert (case/'completed.json').exists() and (audit/'completed.json').exists()
    s,g,cm=read(case/'summary.json'),read(audit/'summary.json'),read(case/'frozen_manifest.json')
    r=s['result'];seed=row['seed'];nv=2+3*m['links']
    assert row['success'] and g['passed'] and r['success'] and r['seed']==g['seed']==seed
    assert cm['config']==Config(n_links=m['links']).__dict__
    assert cm['source_sha256']==m['source_sha256'] and cm['bundle_sha256']==m['source_bundle_sha256']
    assert cm['solver_receipt_sha256']==m['solver_receipt_sha256']
    missing=[]
    for name in ALGORITHM:
        if name not in cm['helper_sha256']:missing.append(name)
        else:assert cm['helper_sha256'][name]==m['helper_sha256'][name]
    if missing:
        # The original N12 first case never imported its three lazy solve-only modules.
        first=read(Path(row['origin_inventories'] if 'origin_inventories' in row else m['first_inventory_directory'])/'first.json')
        assert m['links']==12 and seed==20261116 and not qualification
        assert case.resolve()==Path(first['run']).resolve() and row['inherited']
        assert set(missing)==set(ALGORITHM[4:])
        certs=list((case/'planning').rglob('certificate.npz'));assert certs
        for path in certs:
            with np.load(path) as z:assert str(z['kind'])=='zero_residual'
    assert s['frozen_manifest_sha256']==g['frozen_manifest_sha256']==sha(case/'frozen_manifest.json')
    assert g['case_summary_sha256']==row['source_case_summary_sha256']==sha(case/'summary.json')
    assert g['actual_sha256']==sha(case/'actual.npz')
    assert row['source_audit_summary_sha256']==sha(audit/'summary.json')
    assert g['actual_cold_byte_exact'] and g['physical_velocities']==nv
    assert r['steps']==1500 and r['replay_exact'] and r['native_steps']==r['proof_native_steps']==6000
    assert r['angle_noise']==.02 and r['initialization']=='hanging'
    assert r['best_hold_seconds']==g['best_hold_seconds']>=5.
    assert r['plane_margin']>=0. and not r['physics_warning'] and r['max_force_norm']<=80.
    assert r['policy_metadata']['park_seconds']==0. and r['policy_metadata']['physical_velocities']==nv
    assert r['policy_metadata']['predictor_native_steps']==s['planning_native_steps']
    assert not r['policy_metadata']['hardware_realtime_claim']
    assert g['source_native_steps']==s['inclusive_native_steps']==s['planning_native_steps']+12000
    assert s['inclusive_native_steps']<=planning_budget(Config(n_links=m['links']))+12000
    assert bool(cm.get('qualification_attempt',False))==qualification
    assert s.get('reserved_cases_consumed',0)==int(qualification)
    if row['inherited']:
        assert not qualification and row['new_actor_native_steps']==row['new_audit_native_steps']==0
    else:
        assert row['new_actor_native_steps']==s['inclusive_native_steps']
        assert row['new_audit_native_steps']==g['native_steps']
    return dict(seed=seed,path=str(audit.resolve()),summary_sha256=sha(audit/'summary.json'),
        actual_sha256=g['actual_sha256'],actor_native_steps=s['inclusive_native_steps'],
        audit_native_steps=g['native_steps'],best_hold_seconds=r['best_hold_seconds'],
        plane_margin=r['plane_margin'],max_force_norm=r['max_force_norm'])


def complete(run):
    run=Path(run);assert (run/'completed.json').exists(), 'Parent/cohort is incomplete'
    s=read(run/'summary.json');m,config=manifest_check(run)
    # Supply the first-inventory location to validate the original inherited N12 case.
    m=dict(m,first_inventory_directory=str(run/'inventories'))
    stage=m.get('cohort','pilot12');fresh=stage=='qualification32'
    expected=reserved(m['links']) if fresh else PILOT+(ADDITIONAL if stage=='development45' else [])
    assert stage in ('pilot12','development45','qualification32')
    assert s['cohort_finished'] and not s['stopped_on_failure']
    assert m['seeds']==expected and [r['seed'] for r in s['results']]==expected
    nominal=None if fresh else nominal_equivalence(run,m)
    proofs=[case_check(row,m,fresh) for row in s['results']]
    noisy=len(proofs);passes=noisy+int(nominal is not None)
    assert s['passes']==(noisy if stage=='pilot12' else passes)
    own_actor=sum(r['new_actor_native_steps'] for r in s['results'])
    own_audit=sum(r['new_audit_native_steps'] for r in s['results'])
    assert s['inclusive_native_steps']==own_actor+own_audit
    if stage=='development45':
        parent=Path(m['parent_run']);ps=read(parent/'summary.json')
        assert sha(parent/'summary.json')==m['parent_summary_sha256']
        for old,row in zip(ps['results'],s['results'][:12]):
            assert row['inherited'] and row['run']==old['run'] and row['audit']==old['audit']
            assert row['source_case_summary_sha256']==old['source_case_summary_sha256']
            assert row['source_audit_summary_sha256']==old['source_audit_summary_sha256']
        assert all(not row['inherited'] for row in s['results'][12:])
    if fresh:
        reservation=read(run.parent/f'reserved_{m["links"]}_immediate_{expected[0]}.json')
        assert reservation['seeds']==expected and Path(reservation['run']).resolve()==run.resolve()
        assert reservation['frozen_manifest_sha256']==sha(run/'frozen_manifest.json')
        assert reservation['controller_identity']==m['controller_identity']
        assert read(run/'consumed_cases.json')['seeds']==expected
        assert s['reserved_cases_consumed']==32 and not any(r['inherited'] for r in s['results'])
    return dict(passed=True,cohort=stage,links=m['links'],passes=passes,noisy_passes=noisy,
        qualified=fresh,native_steps=0,reserved_cases_consumed=0,source_reserved_cases_consumed=32 if fresh else 0,
        controller_identity=m['controller_identity'],source_summary_sha256=sha(run/'summary.json'),
        frozen_manifest_sha256=sha(run/'frozen_manifest.json'),case_seeds=expected,case_audits=proofs,
        nominal_equivalence=nominal,referenced_actor_native_steps=own_actor,
        referenced_case_audit_native_steps=own_audit,
        minimum_hold_seconds=min(p['best_hold_seconds'] for p in proofs),
        minimum_plane_margin=min(p['plane_margin'] for p in proofs),
        maximum_force_norm=max(p['max_force_norm'] for p in proofs),
        scope='complete frozen-candidate cohort, original resets, task/cold/model case proofs, inherited lineage and one separately proven nominal; no new physics')
