"""Create-only scientific bundle; runtime data remain unchanged multipart assets."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import zipfile
from delivered14_release_gate import qualified_record,TAG,ROOT,asset
from qualified_delivered_source14_parallel import first_qualified,read,sha
from release_archive import create_evidence_archives,verify_archive
from stage_six_source_synthesis14 import helper_closure


def put(source,target):
    source,target=Path(source),Path(target)
    assert source.is_file() and not target.exists()
    target.parent.mkdir(parents=True,exist_ok=True)
    try:os.link(source,target)
    except OSError:shutil.copyfile(source,target)
    assert sha(source)==sha(target)


def scientific_tree(source,target,*,keep_models=False,omit_runtime_view=False):
    """Retain motion, proposals, certificates and failures; declare every omission."""
    source,target=Path(source),Path(target);assert source.is_dir() and not target.exists()
    target.mkdir(parents=True);omitted={};retained=0
    for path in sorted(source.rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts or path.suffix=='.pyc':continue
        name=path.relative_to(source).as_posix();reason=None
        if path.name=='plan.md':reason='private_historical_research_notebook_not_a_numerical_input'
        elif omit_runtime_view and 'runtime-view' in Path(name).parts:
            reason='materializable_unchanged_shared_runtime_from_five_external_assets'
        elif not keep_models and re.search(r'/(native_models|coordinates)\.npz$',name):
            reason='regenerable_native_model_or_value_coordinate_cache'
        if reason:
            omitted[name]=dict(sha256=sha(path),bytes=path.stat().st_size,reason=reason)
        else:put(path,target/name);retained+=1
    (target/'declared_omissions.json').write_text(json.dumps(dict(files=omitted,
        scope='Original omitted bytes stay local; all actual and negative/positive forecast tapes retained. Full first-fresh expected planning is retained for isolated resynthesis.'),indent=2)+'\n')
    return dict(source=str(source),retained_files=retained,omitted_files=len(omitted))


def main():
    p=argparse.ArgumentParser()
    for name in ('proof','archive-run','reproduction','margin','video','video-qa','out'):
        p.add_argument('--'+name,required=True,type=Path)
    a=p.parse_args()
    record=qualified_record(a.proof,a.archive_run,a.reproduction,a.margin,a.video,a.video_qa)
    admission=first_qualified(a.proof);cohort=Path(admission['cohort'])
    out=a.out.resolve();out.relative_to((ROOT.parent/'artifacts/releases/2d').resolve())
    assert not out.exists();out.mkdir(parents=True)
    stage=out/'content';stage.mkdir();rows=[]
    rows.append(scientific_tree(cohort,stage/'evidence/qualification',keep_models=False))
    # Keep all first-fresh planning fields even when other cases omit large caches.
    first=cohort/'cases'/str(admission['first_seed'])/'prepare/planning'
    for path in first.rglob('*.npz'):
        rel=path.relative_to(cohort).as_posix();target=stage/'evidence/qualification'/rel
        if not target.exists():put(path,target)
    # Explicitly refine that tree's omission list for the retained first case.
    omitted_path=stage/'evidence/qualification/declared_omissions.json'
    omission=read(omitted_path)
    for name in list(omission['files']):
        if (stage/'evidence/qualification'/name).exists():del omission['files'][name]
    omitted_path.write_text(json.dumps(omission,indent=2)+'\n')
    cm=read(cohort/'frozen_manifest.json');development=Path(cm['development_proof'])
    rows.append(scientific_tree(development,stage/'evidence/development-proof'))
    ds=read(development/'summary.json')
    for row in ds['results']:
        if row['proof_kind']=='unchanged_exact_source_constructor':continue  # Exact source packets/proofs are in immutable runtime data.
        inventory=Path(row['witness_inventory'])
        prep_path=inventory.with_name(inventory.name.replace('_witness_inventory_','_inventory_'))
        witness,prep=read(inventory),read(prep_path)
        target=stage/'evidence/development'/str(row['seed'])
        put(inventory,target/'witness_inventory.json');put(prep_path,target/'preparation_inventory.json')
        for name,folder in [('prepare',prep['run']),('audit',prep['audit']),('witness',witness['run']),('reproduction',witness['audit'])]:
            rows.append(scientific_tree(folder,target/name))
        if prep.get('failed_audit'):
            rows.append(scientific_tree(prep['failed_audit'],target/'preserved_failed_audit'))
    rows.append(scientific_tree(a.reproduction,stage/'evidence/extracted-resynthesis',keep_models=False,omit_runtime_view=True))
    # Original expected planning is the complete first-case reconstruction oracle,
    # blocked during actual re-synthesis by the isolated worker's audit hook.
    expected=Path(a.reproduction)/'expected-planning'
    for path in expected.rglob('*.npz'):
        target=stage/'evidence/extracted-resynthesis/expected-planning'/path.relative_to(expected)
        if not target.exists():put(path,target)
    rows.append(scientific_tree(a.margin,stage/'evidence/development-margin-analysis'))
    mi=read(Path(a.margin)/'inputs.json');grid=Path(mi['run'])
    for name in ('inputs.json','summary.json','rows.jsonl','exploration.json'):
        put(grid/name,stage/'evidence/development-margin-grid'/name)
    (stage/'evidence/development-margin-grid/tapes_retained_locally.json').write_text(json.dumps(dict(
        scope='All480 actual/cold-verified tapes remain local and are hash-listed in rows.jsonl; reproduce the frozen development diagnostic to regenerate them.',
        source_rows_sha256=sha(grid/'rows.jsonl')),indent=2)+'\n')
    # Only frozen scientific closure and explicit public reconstruction adapters.
    for name,digest in cm['helper_sha256'].items():
        source=cohort/'helpers'/name;assert sha(source)==digest;put(source,stage/'lab_source/scripts'/name)
    package=Path(cm['candidate'])/'source'
    if not package.is_dir():package=ROOT/'src/spatialpole'
    for name,digest in cm['source_sha256'].items():
        source=package/name;assert sha(source)==digest;put(source,stage/'lab_source/src/spatialpole'/name)
    extras=('qualified_delivered_source14_parallel.py','reproduce_qualified_delivered14_timed.py',
        'render_qualified_delivered14_parallel.py','delivered14_release_gate.py','package_delivered14_release.py',
        'stage_delivered14_publication.py','publish_delivered14_release.py','release_archive.py',
        'analyze_delivered14_margin.py','measure_delivered14_margin_v2.py','publish_eleven_synthesis_release.py')
    tracked=set(subprocess.check_output(['git','ls-files','--','scripts/*.py'],cwd=ROOT,text=True).splitlines())
    for name in helper_closure(set(extras)):
        assert 'scripts/'+name in tracked,'Only committed scientific sources may enter the release'
        target=stage/'lab_source/scripts'/name
        if not target.exists():put(ROOT/'scripts'/name,target)
    for name in ('pyproject.toml','requirements-lock.txt'):put(ROOT/name,stage/'lab_source'/name)
    for name in ('relative-profile-template.json','layout.json','relocation.json','archive-provenance.json'):
        put(Path(a.archive_run)/'extracted-payload'/name,stage/'runtime-metadata'/name)
    put(Path(a.archive_run)/'member-manifest.json',stage/'runtime-metadata/member-manifest.json')
    for source in (a.video.with_suffix('.json'),a.video_qa,*a.video.parent.glob(a.video.stem+'-frame*.png'),
                   a.video.parent/'contact-sheet.png',a.video.parent/'framing_proof.json',
                   a.video.parent/'projected_body_bounds.npz',a.video.parent/'render_driver.py'):
        put(source,stage/'video-proof'/source.name)
    public_assets=[]
    for item in record['runtime_archives']:
        target=out/item['name'];put(item['path'],target);public_assets.append(asset(target))
    target=out/(TAG+'.mp4');put(a.video,target);public_video=asset(target)
    record=dict(record,runtime_archives=public_assets,video=public_video)
    (stage/'copy_index.json').write_text(json.dumps(rows,indent=2)+'\n')
    mass=record['mass_tolerance']
    mass_text=str(mass) if mass is not None else f"below tested floor {record['minimum_tested_mass']} (not located)"
    (stage/'README.md').write_text(f'''# 14-link 2D immediate noisy swing-up

32/32 fresh original noisy starts; 45/45 development states. No hanging
conditioning. Original full30s task, at least 5-second world-tilt hold,44physical velocities,
80N radial float32 force,3m square,50 Hz commands / 200 Hz float64 MuJoCo RK4.

Runtime: **offline per-episode planning with simulation paused**, followed by
full-state simulator feedback. Cost for every qualification episode is in
manifest.json/per_episode_costs (actor native, independent audit native, complete
case wall INCLUDING registration/audit/witness). A separately timed extracted
first-case re-synthesis measures pure planning: {record['first_extracted_planning_native_steps']:,}
native steps/{record['first_extracted_planning_wall_seconds']:.3f}s. This is not
a hardware real-time controller. The conditioned14 comparator is separate.

Development guard-bypass force sigma50={record['measured_sigma50_N']:.9g}N per axis;
both-sign mass tolerance={mass_text}. Noise acts on EVERY command before the
original clip/quantizer; true mass labels never enter policy. All20starts/cell,
Firth fit/paired2000bootstrap. These are measured margins, not a gust-robustness
claim beyond the reported levels. Exact-model qualification remains separate.

Download all five runtime ZIP assets, all evidence ZIP parts, manifest and MP4.
Runtime ZIPs preserve the byte-identical development payload; inner development
metadata describe its original freeze stage, while this companion manifest
binds its later qualification. Extract runtime ZIPs together into one payload
directory. Pinned original Windows/Python/packages and empty thread environment
are required for the recorded byte proof; see requirements-lock.txt.

The isolated extracted reconstruction driver is retained in
evidence/extracted-resynthesis/driver.py. With a NEW worker output on the same
volume as the extracted runtime payload, invoke its --job job.json --payload
PAYLOAD --worker-out NEW_OUTPUT. It regenerates first-fresh planning and actual/
cold motion with original scientific paths and expected arrays blocked during
planning. It reuses a qualified observation, not another fresh sample.

Every qualification and newly planned development actual tape, failed/successful
forecast, correction and certificate survives. Other native-model/value caches
are explicitly omitted as regenerable; full first-case expected planning is
retained. Private historical notebook copies and materializable runtime-view
aliases are also explicitly hash-listed omissions. All480 disturbance tapes stay
local with hashes and frozen regeneration source. No private collaboration tree,
user-owned untracked code or native installation is swept into this bundle.
''',encoding='utf-8')
    for omission_file in stage.rglob('declared_omissions.json'):
        declaration=read(omission_file)
        for name in list(declaration['files']):
            if (omission_file.parent/name).exists():del declaration['files'][name]
        omission_file.write_text(json.dumps(declaration,indent=2)+'\n')
    files={p.relative_to(stage).as_posix():sha(p) for p in sorted(stage.rglob('*')) if p.is_file()}
    manifest=dict(record,accepted_swingup_links=14,archive_name_stem='2d-cartpole-14-link-immediate',files_sha256=files)
    (stage/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    create_evidence_archives(stage,out/(TAG+'.zip'),manifest,compression=zipfile.ZIP_STORED)
    verify_archive(out/(TAG+'.zip'),manifest)
    archives=[asset(out/(TAG+'.zip'))]+[asset(out/item['name']) for item in manifest.get('supplementary_evidence_bundles',[])]
    put(stage/'manifest.json',out/(TAG+'-manifest.json'))
    receipt=dict(passed=True,qualified=True,all_zip_hashes_and_crc_verified=True,
        manifest_sha256=sha(out/(TAG+'-manifest.json')),archives=archives,
        runtime_archives=public_assets,video=public_video,controller_identity=record['controller_identity'],
        source_record=record,native_steps=0,new_labels=0)
    (out/'bundle-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(dict(passed=True,qualified=True,out=str(out),evidence_archives=len(archives),native_steps=0),flush=True)


if __name__=='__main__':main()
