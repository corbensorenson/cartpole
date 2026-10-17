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
manifest = json.loads((root/'publication/2d/manifest.json').read_text())
assert manifest['accepted_swingup_links']==a.links
if manifest.get('archive_schema',1)>=2:
    body = body.replace('controller/gains, all final state/action tapes, failed two-link baseline and\nits optimizer history, tests, runtime, research plans and SHA256 manifest.',
        'controller/gains, all current-count state/action tapes and failed variants,\ntests, runtime, research plans and SHA256 manifest. Unchanged historical raw\nevidence stays in earlier immutable release bundles and in the Git checkout.')
    prior = manifest['prior_evidence_bundle']
    current_scope='current release archives (primary and supplements)' if manifest.get('supplementary_evidence_bundles') else 'current ZIP'
    body += f"\nHistorical evidence dependency: [{prior['links']}-link evidence bundle]({prior['url']}), SHA256 `{prior['sha256']}`. The manifest binds its exact bytes and prior inventory. Current-count reproduction uses the {current_scope}. Historical manifests bind any additional parts required for complete prior evidence.\n"
if manifest.get('supplementary_evidence_bundles'):
    body += '\nThe retained current-count evidence spans multiple bounded ZIP assets. Download the primary and every listed supplement for the complete record; their exact file inventories, SHA256 digests and byte counts are bound in the primary manifest. Runtime/source and the current qualification stay in the primary archive.\n'
    for part in manifest['supplementary_evidence_bundles']:
        body += f"\n- [{part['name']}]({part['url']}), SHA256 `{part['sha256']}`, {part['bytes']} bytes.\n"
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
if a.links == 13:
    assert 'launch_reference_profiles' in frozen
    assert (qualification/'forecast_actual_parity.json').exists()
    body += '\nThirteen transfers an unwrapped material angle/rate field from a historical fourteen-link geometric proposal and repairs it on native thirteen-link mechanics. The first thirteen donor reaches only30/45 development passes; the new earlier-arrival donor reaches44/45. A fixed -1mm Y reference offset also reaches44/45, with a different failure. Both complete failed gates are retained. The selected causal controller prepares for19s, then forecasts BOTH frozen references from the observed launch state in owned native environments and selects the first complete passing continuation. There is no seed lookup, future actual state or plant reset. Every forecast tape and physics-query count is saved. Its9s swing plus2s tracking executes550 commands of the original1500-command finite affine QR design. Nominal plus44 noisy development starts pass, followed by32 NEW reserved starts20262416..20262447. Every chosen forecast matches actual suffix states, delivered/raw forces and tilts byte-for-byte. All actual episodes complete30s with5.16s hold and exact independent cold replay. The131-test suite includes native spatial freedoms, predictor isolation/admission/fallback and archive integrity. The nominal motion remains nearly planar in fully unrestricted spatial physics. Prediction cost is reported separately; no hardware real-time claim is made.\n'
if a.links == 14:
    assert bool(frozen['controller_bank']) and len(frozen['bank_states'])==5
    assert (qualification/'bank_audit.json').is_file()
    body += '\nFourteen uses a moving reference transferred as geometry from a planar donor and repaired under the exact spatial mechanics. A quotient value design retains every physical velocity, including spin. The original controller passes43/45 development starts. Two full-spatial suffix repairs from measured development entries rescue the difficult starts individually while preserving the first22.2s physical prefix exactly. A three-controller forecast bank passes45/45 development but fails its first reserved cohort31/32; all failures are retained. The revised fixed family adds reflected suffix laws, passes46 distinct development starts including that explicitly reused former failure, then passes a NEW32-start cohort20262616..20262647. At the observed19.2s launch, complete owned native forecasts in fixed order select the first full passing continuation; runtime receives no episode seed and never resets the plant. The9s bridge and finite upright tracking execute540 commands of the original700-command design. Every selected raw/delivered force, state and tilt matches its full forecast byte-for-byte, including the final transition. The nominal motion remains nearly planar in unrestricted spatial physics; spatial repairs do not lock or actuate joints. No hardware real-time claim is made.\n'
if manifest.get('git_lfs_files'):
    body += '\nLarge Git checkpoints use Git LFS (run git lfs pull after cloning). Every release ZIP contains materialized original bytes rather than LFS pointers; the manifest binds logical file hashes and the staged verifier checks the corresponding LFS objects.\n'
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
asset_specs=[(f'2d-cartpole-{a.links}-link-swingup.mp4','video/mp4'),(f'2d-cartpole-{a.links}-link-evidence.zip','application/zip')]
for part in manifest.get('supplementary_evidence_bundles',[]):
    name=part['name']
    assert Path(name).name==name and name.startswith(f'2d-cartpole-{a.links}-link-evidence-part') and name.endswith('.zip')
    file=root/'output'/name
    assert file.stat().st_size==part['bytes']
    with file.open('rb') as stream:
        assert hashlib.file_digest(stream,'sha256').hexdigest()==part['sha256']
    asset_specs.append((name,'application/zip'))
for name,kind in asset_specs:
    file = root/'output'/name
    assert file.stat().st_size<2**31,'release asset exceeds GitHub per-file limit'
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
