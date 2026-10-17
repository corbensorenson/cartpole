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
trajectory feedback; no learned-policy claim.

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
    swing_steps = int(frozen['swing_steps']) if 'swing_steps' in frozen else len(frozen['actions'])
    route_seconds = swing_steps*config['timestep']*config['frame_skip']
    body += f'\nThe frozen policy first applies {park_seconds:.2f}s of hanging LQR, then uses observed axial-frame alignment before its {route_seconds:.2f}s swing. All phases fit inside the same 30s episode; there are no state resets or joint locks.\n'
    if 'affine_value' in frozen and bool(frozen['affine_value']):
        design_seconds = len(frozen['actions'])*config['timestep']*config['frame_skip']
        body += f'\nIts full affine QR feedback was designed over {design_seconds:.2f}s, including an upright continuation; execution uses only the prefix remaining after preparation. The policy aligns reference axial frames at each observed feedback step. Measured twist/spin stays physical and free. Finite design time is not added to episode time.\n'
if a.links == 3:
    body += '\nThe earlier unconditioned three-link policy failed qualification at 25/32. Its failures were retained and reused only as development data; this release passes a genuinely new 32-start cohort.\n'
if a.links == 4:
    body += '\nFour-link synthesis uses a historical altered-mechanics motion only as a geometric target. Native affine tracking repair restores the exact benchmark dynamics. The first repaired checkpoint was frozen before actual outcomes and passed nominal plus 12 development starts, followed by the fresh 32-start cohort. Its 23.04s motion starts immediately; no extra conditioning time or state reset is used. Altered-plant diagnostics and failed optimizer alternatives are retained separately.\n'
if a.links == 5:
    body += '\nFive-link synthesis uses a lifted historical motion only as a geometric proposal. The source reached 3.33m on a wider rail and is not accepted as this benchmark. Native affine tracking repair produces an 8.42s swing on the exact spatial model inside the unchanged 3m boundary. The frozen final controller passes nominal plus 12 development starts and the fresh 32-start cohort. No preliminary conditioning, gauge alignment or extra episode time is used. Maximum final-cohort cart coordinate is 2.731121m.\n'
if a.links == 6:
    body += '\nThe raw six-link native repair passed nominal but failed four of 12 noisy development starts; those failures are retained. The conditioned frozen policy passes nominal plus all 44 noisy development starts and a fresh 32-start qualification cohort. The historical wider-rail motion is only a geometric proposal. All final episodes use the unchanged spatial model and 3m boundary; maximum final-cohort cart coordinate is 2.729448m.\n'
if a.links == 7:
    body += '\nThe earlier raw seven-link controller failed reserved qualification at 31/32. Those seeds were reused only as development. The conditioned frozen policy passes nominal plus all 76 noisy development starts, then a NEW untouched 32-start cohort (20271816..20271847). Both failures and the independent spatial-freedom audit are preserved. The nominal swing reference remains essentially planar in unrestricted 3D dynamics; this is not a claim of a deliberately nonplanar swing maneuver. A separate controlled six-link XY-loop demonstration verifies deliberate two-axis motion.\n'
if a.links == 8:
    body += '\nThe raw eight-link native repair passed nominal but failed all 12 noisy development starts. Its conditioned frozen policy passes nominal plus all 44 development starts and a fresh 32-start qualification cohort. Maximum final-cohort cart coordinate is 2.726041m. The near-planar nominal reference is tracked in unrestricted spatial dynamics; compiled eight-link freedoms, independent persistent native XY replay, axis symmetry and every joint tilt direction/spin are covered by the 48-test suite. The spatial audit and controlled six-link XY demonstration are included.\n'
if a.links == 9:
    body += '\nThe raw nine-link native repair passed nominal but failed all 12 noisy development starts; all failures are retained. The preplanned 18s hanging/gauge wrapper passes nominal plus all 44 development starts before a fresh 32-start qualification cohort. Preparation, the 3.94s swing and upright feedback all run inside the same 30s episode. The near-planar nominal reference is tracked in unrestricted spatial dynamics; nine-link compiled freedoms, persistent native XY replay, axis symmetry and each joint tilt direction/spin are included in the 52-test suite. The independent spatial audit and controlled six-link XY demonstration remain included.\n'
if a.links == 10:
    body += '\nThe raw ten-link native repair passed nominal but failed all 12 noisy development starts; those failures are retained. Its preplanned 16s hanging/gauge wrapper passes nominal plus all 44 development starts before the fresh 32-start qualification cohort. The 16s preparation, 8s swing and upright feedback fit inside the unchanged 30s episode, without state resets or joint locks. The nominal reference remains near-planar in unrestricted spatial dynamics. Ten-link compiled freedoms, persistent native XY replay, axis symmetry and every joint tilt direction/spin are included in the 56-test suite. The independent spatial audit and controlled six-link XY demonstration remain included. Native instability warnings occurred in rejected optimizer search trials; no accepted actual episode reports a physics warning.\n'
if a.links == 11:
    body += '\nEleven-link synthesis uses a historical planar motion only as a geometric proposal for the unchanged spatial model. Full spatial simultaneous sparse shooting repairs native defects, then full affine QR offsets and a finite upright continuation produce a nominal pass. The raw controller fails all 12 noisy development starts. Static launch alignment with 16s, 18s and 18.4s preparation reaches only 27/45, 40/45 and 43/45 development passes. The selected policy uses 18s preparation plus causal per-command axial reference alignment, passes nominal plus all 44 shared noisy development starts, then the NEW 32-start cohort (20262216..20262247). The 30s episode includes its entire preparation, 8s swing and finite upright tracking. Failed optimization, action-search, timing and controller variants are retained. The nominal reference remains near-planar in unrestricted spatial physics; no spatial joint or twist is locked. The 85-test suite includes eleven-link freedoms, native replay, independent affine optima and axial-frame feedback invariance. The separate six-link controlled XY-loop audit remains included. This adapts affine-value and finite-horizon ideas reviewed in corben-marc-hybrid.md; it is not a reproduction of the scalar 1D B17 controller.\n'
if a.links == 12:
    body += '\nTwelve-link synthesis decodes a historical physical-relative donor through its explicit coordinate matrix, verifies exact stored physical-tape parity and hanging initialization, and repairs its proposal on the unchanged full spatial plant. An earlier incorrect cumulative-coordinate import and its interrupted repair are retained with correction notes. Native full-state sparse shooting uses augmented ridge solves within registered bounds; a factored finite-QR upright design avoids a numerically indefinite stationary Riccati value. The repaired baseline holds 4.24s and later fails. Full affine finite feedback passes nominal23.50s but all12 raw noisy starts fail. The selected18s hanging/online-axial policy passes nominal plus all44 noisy development starts, then all32 NEW reserved starts20262316..20262347, each completing30s with5.50s hold and exact replay. Its9s swing and final3s upright tracking execute a600-command prefix of a1500-command design. Physical XYZ joint freedom and passive spin remain intact. The96-test suite includes12-link native freedoms, independent affine optima, gauge invariance, factored design checks and typed donor-coordinate regression. The nominal reference remains near-planar; the separate six-link controlled XY-loop audit remains included.\n'
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
