"""Publish only a fully gated 14-link 2D immediate result and immutable assets."""
import argparse
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import quote
from delivered14_release_gate import qualified_record,TAG
from qualified_delivered_source14_parallel import read,sha
from publish_eleven_synthesis_release import Github,API,find_release


def main():
    p=argparse.ArgumentParser()
    for key in ('proof','archive-run','reproduction','margin','video','video-qa','bundle','publication-repo','out'):
        p.add_argument('--'+key,required=True,type=Path)
    p.add_argument('--commit',required=True);p.add_argument('--study-path',required=True)
    p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    record=qualified_record(a.proof,a.archive_run,a.reproduction,a.margin,a.video,a.video_qa)
    assert re.fullmatch('[0-9a-f]{40}',a.commit)
    assert re.fullmatch(r'2d/research/immediate_14_link_[0-9]{8}',a.study_path)
    manifest_path=a.bundle/(TAG+'-manifest.json');m=read(manifest_path);b=read(a.bundle/'bundle-receipt.json')
    assert b['qualified'] and b['all_zip_hashes_and_crc_verified'] and b['manifest_sha256']==sha(manifest_path)
    for key,value in record.items():
        if key not in ('runtime_archives','video'):assert m[key]==value,key
    assert len(m['runtime_archives'])==len(record['runtime_archives'])==5
    for expected,public in zip(record['runtime_archives'],m['runtime_archives']):
        assert all(expected[k]==public[k] for k in ('name','sha256','bytes'))
    assert m['video']['sha256']==record['video']['sha256'] and m['video']['name']==TAG+'.mp4'
    assets=[m['video'],*m['runtime_archives'],*b['archives'],
            dict(name=manifest_path.name,path=str(manifest_path.resolve()),sha256=sha(manifest_path),bytes=manifest_path.stat().st_size)]
    assert len({row['name'] for row in assets})==len(assets)
    for row in assets:
        path=Path(row['path']);assert path.name==row['name'] and path.stat().st_size==row['bytes']<2**31
        assert sha(path)==row['sha256']
    public=subprocess.check_output(['git','-C',str(a.publication_repo),'show',a.commit+':'+a.study_path+'/manifest.json'])
    assert json.loads(public)==m
    assert subprocess.check_output(['git','-C',str(a.publication_repo),'status','--porcelain'],text=True).strip()==''
    blob=subprocess.check_output(['git','-C',str(a.publication_repo),'rev-parse',a.commit+':'+a.study_path+'/manifest.json'],text=True).strip()
    mass=m['mass_tolerance'];mass_text=f'{mass:.9g}' if mass is not None else f"below tested floor {m['minimum_tested_mass']:.1e} (not located)"
    study='https://github.com/corbensorenson/cartpole/tree/'+a.commit+'/'+a.study_path
    tiers=[key for key in ('M3','M2','M1') if m['development_margin_tiers_met'][key]]
    tier=tiers[0] if tiers else 'E'
    body=f'''Qualified **14-link 2D immediate noisy swing-up**: **32/32 fresh starts**
after 45/45 development states, with zero hanging conditioning. Every episode
completes the original 30-second task; minimum upright hold is
{m['minimum_hold_seconds']:.2f} seconds. Full first-fresh 50 fps video attached.

Runtime class: **offline per-episode planning with simulation paused**, then
full-state simulator feedback. The separately timed extracted first-case plan
used **{m['first_extracted_planning_native_steps']:,} native steps** and
**{m['first_extracted_planning_wall_seconds']:.3f} seconds** of planning.
[All 32 per-episode costs and source]({study}) distinguish planning native work,
independent numerical audit, actual/cold witnesses and complete case wall time.
Hardware real-time operation is not claimed.

Measured development margins, same frozen candidate and 20 starts per cell:
**sigma50={m['measured_sigma50_N']:.9g} N per axis**, 95% paired-bootstrap
interval {m['sigma50_bootstrap95_N']}; **both-sign relative mass tolerance
{mass_text}**. Force noise is independent Gaussian on EVERY command before
the original clip/float32 delivery. Parity checks are diagnostic counters in
this margin measurement. Both mass signs need at least 10/20; the preserved
estimator also requires all smaller sampled levels to pass. Qualified tier: **{tier}**.

The cart moves freely in XY; every joint is a passive unrestricted ball joint.
All 44 physical velocities remain. Original 80 N radial force, 3 m square,
50 Hz commands and 200 Hz float64 MuJoCo RK4 are unchanged. The conditioned
14-link comparator and all existing lower-count results are preserved.

Download **all five runtime ZIPs**, every evidence ZIP part, manifest and MP4.
The runtime bytes preserve their development payload freeze; the companion
qualification manifest binds them to this result. The first fresh case was
fully re-synthesized from extracted code/data with original scientific paths
and expected results blocked during planning; all planning and actual/cold
arrays match. See the evidence README for pinned reconstruction instructions.
Regenerable caches/private notebook omissions are declared by hash. Disturbance
tapes remain local and hash-listed with frozen regeneration code. Measurements
are in simulation and do not establish hardware behavior.
'''
    if a.dry_run:
        print(json.dumps(dict(dry_run=True,tag=TAG,body=body,assets=[{k:r[k] for k in ('name','bytes','sha256')} for r in assets]),indent=2));return
    assert not a.out.exists()
    github=Github()
    ref=github.request(API+'/git/ref/tags/'+quote(TAG,safe=''))['object']
    for _ in range(4):
        if ref['type']=='commit':break
        assert ref['type']=='tag';ref=github.request(API+'/git/tags/'+ref['sha'])['object']
    assert ref['type']=='commit' and ref['sha']==a.commit
    remote=github.request(API+'/contents/'+a.study_path+'/manifest.json?ref='+a.commit)
    assert remote['type']=='file' and remote['sha']==blob
    release=find_release(github,TAG)
    if release is None:
        release=github.request(API+'/releases','POST',dict(tag_name=TAG,target_commitish=a.commit,
            name='2D Cartpole — 14 Links: Immediate Noisy Swing-Up',body=body,draft=True,prerelease=False,make_latest='false'))
    assert release['tag_name']==TAG and not release['prerelease']
    existing=github.assets(release['assets_url']);verified=[]
    for row in assets:
        path=Path(row['path'])
        if row['name'] not in existing:
            content='video/mp4' if path.suffix=='.mp4' else 'application/zip' if path.suffix=='.zip' else 'application/json'
            print(dict(uploading=row['name'],bytes=row['bytes']),flush=True)
            existing[row['name']]=github.request(release['upload_url'].split('{')[0]+'?name='+quote(row['name'],safe=''),
                'POST',file=path,content_type=content)
        remote=existing[row['name']]
        assert remote['state']=='uploaded' and remote['size']==row['bytes'] and remote.get('digest')=='sha256:'+row['sha256']
        verified.append({k:row[k] for k in ('name','bytes','sha256')})
    if release['draft'] or release.get('body')!=body:
        release=github.request(API+'/releases/'+str(release['id']),'PATCH',dict(draft=False,body=body,make_latest='false'))
    assert not release['draft'] and not release['prerelease']
    final=github.assets(release['assets_url'])
    for row in verified:
        assert final[row['name']].get('digest')=='sha256:'+row['sha256']
        row['url']=final[row['name']]['browser_download_url']
    receipt=dict(passed=True,qualified=True,tag=TAG,commit=a.commit,url=release['html_url'],assets=verified,
        controller_identity=m['controller_identity'],server_digests_verified=True,all_assets_complete=True,
        links=14,tier=tier,source_manifest_sha256=sha(manifest_path))
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('x') as f:json.dump(receipt,f,indent=2)
    print(receipt,flush=True)


if __name__=='__main__':main()
