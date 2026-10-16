"""Publish an authorized, prepared count-specific 2D GitHub release.

Uses the existing Git credential helper in memory. Never prints credentials.
Run only after the isolated additive source commit is pushed and reviewed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from spatialpole import Config
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

p = argparse.ArgumentParser()
p.add_argument('--commit',required=True)
p.add_argument('--links',type=int,default=1)
p.add_argument('--qualification')
a = p.parse_args()
root = Path(__file__).resolve().parents[1]
credentials = subprocess.run(['git','credential','fill'],input='protocol=https\nhost=github.com\n\n',text=True,capture_output=True,check=True)
fields = dict(line.split('=',1) for line in credentials.stdout.splitlines() if '=' in line)
token = fields['password']
repo = 'corbensorenson/cartpole'
base = 'https://api.github.com/repos/'+repo
tag = f'2d-cartpole-{a.links}-link-v1'

def request(url,method='GET',payload=None,content_type='application/json'):
    if isinstance(payload,dict):
        payload = json.dumps(payload).encode()
    req = Request(url,data=payload,method=method,headers={
        'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json',
        'X-GitHub-Api-Version':'2022-11-28','User-Agent':'2d-cartpole-research-release',
        'Content-Type':content_type})
    with urlopen(req,timeout=120) as r:
        return json.load(r)

qualification = root/a.qualification if a.qualification else root/f'output/qualify_n{a.links}_001'
summary = json.loads((qualification/'summary.json').read_text())
assert summary['passes'] == 32 and len(summary['results']) == 32
assert all(row['success'] and row['replay_exact'] for row in summary['results'])
hold = min(row['best_hold_seconds'] for row in summary['results'])
body = f'''## 2D cart / spatial ball-joint chain — {a.links} links

The cart moves freely in X and Y. The passive ball joints permit both tilt
directions and twist. This is a separate benchmark from the 1D hinge records.

**Qualified hanging-start swing-up: 32/32 fresh noisy 3D starts**, plus nominal
and a separate development cohort. Every episode lasts 30s with at least {hold:.2f}s upright.
All recorded float32 force tapes replay exactly in native MuJoCo 3.3.5.

3m / 1kg chain, 1kg cart, 80N total XY force norm, +/-3m square boundary,
50Hz actions, 200Hz RK4; all world link tilts <0.15rad for >=5s required.
No joint motors, locks, hidden supports or ground contacts. Full-state classical
trajectory feedback followed by upright LQR; no learned-policy claim.

The attached full video shows an entire held-out qualification episode.
The evidence ZIP contains source, frozen synthesis and qualification snapshots,
controller/gains, all final state/action tapes, failed two-link baseline and
its optimizer history, tests, runtime, research plans and SHA256 manifest.
See the count-specific manifest and PROGRESS.md for the current frontier and failures.

This is empirical reset-noise evidence, not arbitrary disturbance rejection,
indefinite stability or hardware real-time validation.
'''
frozen = np.load(qualification/'frozen_route.npz')
assert Config(**json.loads(str(frozen['config']))) == Config(n_links=a.links),'cannot publish an altered-plant result as this benchmark'
if 'park_steps' in frozen:
    config = json.loads(str(frozen['config']))
    park_seconds = int(frozen['park_steps'])*config['timestep']*config['frame_skip']
    route_seconds = len(frozen['actions'])*config['timestep']*config['frame_skip']
    body += f'\nThe frozen policy first applies {park_seconds:.2f}s of hanging LQR, then uses observed axial-frame alignment before its {route_seconds:.2f}s swing. All phases fit inside the same 30s episode; there are no state resets or joint locks.\n'
if a.links == 3:
    body += '\nThe earlier unconditioned three-link policy failed qualification at 25/32. Its failures were retained and reused only as development data; this release passes a genuinely new 32-start cohort.\n'
if a.links == 4:
    body += '\nFour-link synthesis uses a historical altered-mechanics motion only as a geometric target. Native affine tracking repair restores the exact benchmark dynamics. The first repaired checkpoint was frozen before actual outcomes and passed nominal plus 12 development starts, followed by the fresh 32-start cohort. Its 23.04s motion starts immediately; no extra conditioning time or state reset is used. Altered-plant diagnostics and failed optimizer alternatives are retained separately.\n'
if a.links == 5:
    body += '\nFive-link synthesis uses a lifted historical motion only as a geometric proposal. The source reached 3.33m on a wider rail and is not accepted as this benchmark. Native affine tracking repair produces an 8.42s swing on the exact spatial model inside the unchanged 3m boundary. The frozen final controller passes nominal plus 12 development starts and the fresh 32-start cohort. No preliminary conditioning, gauge alignment or extra episode time is used. Maximum final-cohort cart coordinate is 2.731121m.\n'
if a.links == 6:
    body += '\nThe raw six-link native repair passed nominal but failed four of 12 noisy development starts; those failures are retained. The conditioned frozen policy passes nominal plus all 44 noisy development starts and a fresh 32-start qualification cohort. The historical wider-rail motion is only a geometric proposal. All final episodes use the unchanged spatial model and 3m boundary; maximum final-cohort cart coordinate is 2.729448m.\n'
try:
    release = request(base+'/releases/tags/'+tag)
except HTTPError as exc:
    if exc.code != 404:
        raise
    release = request(base+'/releases','POST',{'tag_name':tag,'target_commitish':a.commit,
        'name':f'2D Cartpole — {a.links} '+('Link' if a.links == 1 else 'Links')+': Spatial Swing-Up and Hold', 'body':body,
        'draft':False,'prerelease':False,'make_latest':'false'})
assert release['tag_name'] == tag and not release['draft']
assets = {x['name']:x for x in request(release['assets_url'])}
receipts = []
for name,kind in [(f'2d-cartpole-{a.links}-link-swingup.mp4','video/mp4'),(f'2d-cartpole-{a.links}-link-evidence.zip','application/zip')]:
    file = root/'output'/name
    digest = hashlib.sha256(file.read_bytes()).hexdigest()
    if name in assets:
        asset = assets[name]
        assert asset['size'] == file.stat().st_size
        assert asset.get('digest') in (None,'sha256:'+digest)
    else:
        upload = release['upload_url'].split('{')[0]+'?name='+quote(name)
        asset = request(upload,'POST',file.read_bytes(),kind)
    assert asset['state'] == 'uploaded' and asset['size'] == file.stat().st_size
    receipts.append({'name':name,'url':asset['browser_download_url'],'sha256':digest,'size':asset['size']})
receipt = {'url':release['html_url'],'tag':tag,'commit':a.commit,'assets':receipts}
(root/f'output/github_2d_{a.links}_link_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
print(json.dumps(receipt,indent=2))
