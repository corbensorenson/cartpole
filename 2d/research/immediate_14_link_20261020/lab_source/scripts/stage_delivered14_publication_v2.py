"""Stage an additive, reviewable source result only after every release gate."""
import argparse
from datetime import datetime
import json
import re
from pathlib import Path
import shutil
import subprocess
from delivered14_release_gate import qualified_record,TAG,ROOT
from qualified_delivered_source14_parallel import read,sha


def git(repo,*args):return subprocess.check_output(['git','-C',str(repo),*args],text=True).strip()


def main():
    p=argparse.ArgumentParser()
    for key in ('proof','archive-run','reproduction','margin','video','video-qa','bundle','publication-repo','out'):
        p.add_argument('--'+key,required=True,type=Path)
    p.add_argument('--study-date',required=True);a=p.parse_args()
    record=qualified_record(a.proof,a.archive_run,a.reproduction,a.margin,a.video,a.video_qa)
    assert re.fullmatch('[0-9]{8}',a.study_date)
    datetime.strptime(a.study_date,'%Y%m%d')
    manifest=read(a.bundle/(TAG+'-manifest.json'));bundle=read(a.bundle/'bundle-receipt.json')
    assert bundle['passed'] and bundle['qualified'] and bundle['all_zip_hashes_and_crc_verified']
    assert bundle['manifest_sha256']==sha(a.bundle/(TAG+'-manifest.json'))
    for key,value in record.items():
        if key not in ('runtime_archives','video'):assert manifest[key]==value,key
    for expected,public in zip(record['runtime_archives'],manifest['runtime_archives']):
        assert all(expected[key]==public[key] for key in ('name','sha256','bytes'))
    assert len(record['runtime_archives'])==len(manifest['runtime_archives'])==5
    assert record['video']['sha256']==manifest['video']['sha256']
    repo=a.publication_repo.resolve()
    assert Path(git(repo,'rev-parse','--show-toplevel')).resolve()==repo
    assert git(repo,'remote','get-url','origin')=='https://github.com/corbensorenson/cartpole.git'
    assert git(repo,'status','--porcelain')=='','Preserve existing checkout changes'
    branch='codex/2d-immediate-14';assert subprocess.run(['git','-C',str(repo),'show-ref','--verify','--quiet','refs/heads/'+branch]).returncode==1
    study=repo/f'2d/research/immediate_14_link_{a.study_date}';assert not study.exists()
    documents=[repo/'README.md',repo/'2d/README.md'];heading='## 14-link immediate noisy swing-up'
    assert all(path.is_file() and heading.encode() not in path.read_bytes() for path in documents)
    parent=git(repo,'rev-parse','HEAD');a.out.mkdir(parents=True,exist_ok=False)
    git(repo,'switch','-c',branch)
    study.mkdir(parents=True)
    shutil.copytree(a.bundle/'content/lab_source',study/'lab_source')
    # Packaging/evidence bytes remain immutable. Retain the mechanical Windows
    # staging amendment as an additional public source alongside the old driver.
    amended=study/'lab_source/scripts/stage_delivered14_publication_v2.py'
    assert not amended.exists();shutil.copyfile(__file__,amended)
    assert sha(amended)==sha(__file__)
    (study/'publication-staging-amendment.json').write_text(json.dumps(dict(
        reason='Normalize resolved filesystem paths instead of comparing Git forward slashes to Windows backslashes',
        original_driver_sha256=sha(ROOT/'scripts/stage_delivered14_publication.py'),
        amended_driver_sha256=sha(amended),scientific_inputs_or_results_changed=False,
        packaged_assets_changed=False,native_steps=0,new_labels=0),indent=2)+'\n')
    for source,name in [(a.bundle/(TAG+'-manifest.json'),'manifest.json'),
                        (Path(a.proof)/'summary.json','fresh32-proof.json'),
                        (Path(a.reproduction)/'summary.json','extracted-resynthesis.json'),
                        (Path(a.margin)/'summary.json','development-margins.json'),
                        (a.video_qa,'video-qa.json')]:
        shutil.copyfile(source,study/name);assert sha(source)==sha(study/name)
    shutil.copyfile(a.video.parent/'contact-sheet.png',study/'contact-sheet.png')
    mass=record['mass_tolerance'];mass_text=f'{mass:.9g}' if mass is not None else f"below tested floor {record['minimum_tested_mass']:.1e} (not located)"
    base='https://github.com/corbensorenson/cartpole/releases/download/'+TAG
    costs=record['per_episode_costs']
    table='\n'.join(f"| {row['seed']} | {row['actor_native_steps']:,} | {row['audit_native_steps']:,} | {row['total_native_steps']:,} | {row['total_prepare_audit_witness_wall_seconds']:.3f} |" for row in costs)
    text=f'''# 14-link 2D immediate noisy swing-up

**32/32 fresh starts pass**, following 45/45 development states. Zero hanging
conditioning; original 30-second task andat least 5-second upright hold. Minimum observed hold:
{record['minimum_hold_seconds']:.2f}s. Free XY cart, passive unrestricted ball
joints, all 44 physical velocities,80 N radial float32 force,3 m square,200 Hz float64 RK4.
The conditioned14 result remains a separate comparator.

[Full 50 fps video]({base}/{manifest['video']['name']}) ·
[Release with all runtime/evidence parts](https://github.com/corbensorenson/cartpole/releases/tag/{TAG})

![Frames from the qualified first fresh start](contact-sheet.png)

## Runtime and compute

**Offline per-episode planning with simulation paused**, then full-state
simulator feedback. This is not a hardware real-time controller. Planning takes
place before physical motion; no stationary physical conditioning is used.
The separately timed first-case extracted-runtime re-synthesis used
{record['first_extracted_planning_native_steps']:,}planning native steps and
{record['first_extracted_planning_wall_seconds']:.3f}s of pure planning.

Every qualification episode's cost is listed below. Wall includes registration,
planning, independent numerical audit and both actual/cold witnesses; it is
not mislabeled as pure planning. Native validation work is explicit.

| Fresh seed | Actor planning native | Numerical audit native | Total native incl. witnesses | Total case wall(s) |
|---|---:|---:|---:|---:|
{table}

## Measured disturbance margins

Development 20 starts/cell, unchanged frozen common candidate, parity-guard bypass
diagnostic. Force sigma50 **{record['measured_sigma50_N']:.9g}N per axis**,
95% paired-bootstrap interval {record['sigma50_bootstrap95_N']}. Both-sign
uniform relative chain-mass tolerance: **{mass_text}** (>=10/20eachsign, all
smaller sampled magnitudes also pass). Gaussian force is added on EVERY command
before original clipping/float32 delivery. True mass labels do not enter policy.
Full tables, failures and estimator provenance are in development-margins.json.
TierE qualification and these finite measured margins are distinct claims.

## Reconstruction and evidence

Download all five runtime ZIPs and every evidence ZIP part from the release.
Runtime bytes are unchanged from their development payload freeze; the external
qualification manifest binds that payload to this result. Follow the evidence
README and the isolated reconstruction driver for the first qualified seed
20263816 using pinned original Windows/Python/package versions and empty thread
environment. Re-synthesis blocks original scientific paths and expected results
until planning ends, then checks all planning arrays and actual/cold motion.

All actual/failed/successful forecast tapes for this candidate's qualification
and planned development cases survive in evidence. Regenerable model/value
caches, private notebook copies and shared runtime aliases are declared by hash;
the full first-case expected planning is retained. The 480 disturbance tapes remain
local, hash-listed and regenerable. No extra force-gust or hardware claim beyond
the measured levels is implied. Existing qualified lower counts remain intact.
'''
    (study/'README.md').write_text(text,encoding='utf-8')
    snippet=f'\n\n{heading}\n\n32/32 fresh original noisy starts, zero hanging conditioning; full30-second / 5-second hold gates. '
    snippet+=f"Offline per-episode planning with simulation paused. Measured dev20 sigma50={record['measured_sigma50_N']:.9g}N/axis, both-sign mass={mass_text}. "
    snippet+=f'[Result, all per-episode costs and full video](https://github.com/corbensorenson/cartpole/tree/{TAG}/2d/research/immediate_14_link_{a.study_date}).\n'
    for path in documents:
        with path.open('ab') as f:f.write(snippet.encode())
    result=dict(passed=True,publication_parent=parent,branch=branch,study_path=study.relative_to(repo).as_posix(),
        source_manifest_sha256=sha(study/'manifest.json'),native_steps=0,new_labels=0,
        scope='reviewable additive staging only; commit/push/tag/release are subsequent authorized steps')
    (a.out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result,flush=True)


if __name__=='__main__':main()
