"""Prepare an additive 2d/ publication subtree; no GitHub mutation here."""
from dataclasses import asdict
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
import numpy as np
from spatialpole import Config, SpatialCartpole

root = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument('--links',type=int,default=1)
p.add_argument('--qualification')
a = p.parse_args()
target = root / 'publication' / '2d'
qualification = root/a.qualification if a.qualification else root/f'output/qualify_n{a.links}_001'
q = json.loads((qualification/'summary.json').read_text())
assert q['passes'] == 32 and len(q['results']) == 32
assert all(r['success'] and r['replay_exact'] for r in q['results'])
frozen = np.load(qualification/'frozen_route.npz')
config = Config(**json.loads(str(frozen['config'])))
assert config.n_links == a.links
assert config == Config(n_links=a.links),'altered-plant evidence cannot promote the unchanged benchmark'
target.mkdir(parents=True,exist_ok=True)
if (target/'manifest.json').exists():
    old = json.loads((target/'manifest.json').read_text())
    if old['accepted_swingup_links'] < a.links:
        (target/'manifests').mkdir(exist_ok=True)
        shutil.copy2(target/'manifest.json',target/'manifests'/f"2d-cartpole-{old['accepted_swingup_links']}-link-v1.json")

for name in ('pyproject.toml','requirements-lock.txt','README.md','PROGRESS.md','.gitattributes','.gitignore'):
    shutil.copy2(root/name,target/name)
for name in ('src','tests','research'):
    shutil.copytree(root/name,target/name,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','.pytest_cache','*.egg-info'))
(target/'scripts').mkdir(exist_ok=True)
for driver in sorted((root/'scripts').glob('*.py')):
    shutil.copy2(driver,target/'scripts'/driver.name)
evidence = target/'evidence'
for name in ('swing_n1_001','qualify_n1_001','swing_n2_001','swing_n2_002'):
    shutil.copytree(root/'output'/name,evidence/name,dirs_exist_ok=True)
if a.links >= 2:
    for name in ('n2_world_handoff_001','n2_world_nominal_001','qualify_n2_001'):
        shutil.copytree(root/'output'/name,evidence/name,dirs_exist_ok=True)
    (target/'publication').mkdir(exist_ok=True)
    shutil.copy2(root/'output/github_2d_one_link_receipt.json',target/'publication/one_link_receipt.json')
for count in range(3,a.links+1):
    for source in sorted((root/'output').iterdir()):
        prefixes = (f'swing_n{count}_',f'planar_lift_n{count}_',f'qualify_n{count}_',
                    f'conditioned_n{count}_',f'fddp_n{count}_',f'n{count}_',
                    f'tracking_repair_n{count}_',f'checkpoint_n{count}_',f'nonplanar_demo_n{count}_',
                    f'split_n{count-1}_to_n{count}_')
        if source.is_dir() and source.name.startswith(prefixes):
            shutil.copytree(source,evidence/source.name,dirs_exist_ok=True)
for receipt in sorted((root/'output').glob('github_2d_*_link_receipt.json')):
    (target/'publication').mkdir(exist_ok=True)
    shutil.copy2(receipt,target/'publication'/receipt.name)
for name in ('balance_001','balance_world_001','energy_n1_001'):
    dst = evidence/name
    dst.mkdir(parents=True,exist_ok=True)
    shutil.copy2(root/'output'/name/'summary.json',dst/'summary.json')
videos = target/'videos'
videos.mkdir(exist_ok=True)
video = root/f'output/2d-cartpole-{a.links}-link-swingup.mp4'
shutil.copy2(video,videos/video.name)
preview = root/f'output/2d-cartpole-{a.links}-link-preview.gif'
if preview.exists():
    shutil.copy2(preview,videos/preview.name)
(evidence/f'{a.links}_link_model.xml').write_text(SpatialCartpole(config).xml,encoding='utf-8')
def release_files():
    allowed = {'src','tests','research','scripts','evidence','videos','manifests','publication','audits'}
    for path in sorted(target.rglob('*')):
        relative = path.relative_to(target)
        if not path.is_file() or len(relative.parts)>1 and relative.parts[0] not in allowed:
            continue
        if any(part in ('__pycache__','.pytest_cache','.venv') or part.endswith('.egg-info') for part in relative.parts):
            continue
        yield path
hashes = {str(p.relative_to(target)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest()
          for p in release_files() if p != target/'manifest.json'}
seed = q['results'][0]['seed']
manifest = {'benchmark':'2D cart / spatial unrestricted ball-joint chain', 'accepted_swingup_links':a.links,
    'config':asdict(config),'qualification':f'evidence/{qualification.name}/summary.json',
    'controller':f'evidence/{qualification.name}/frozen_route.npz',
    'video':str(video.name),'video_source':f'evidence/{qualification.name}/seed{seed}.npz',
    'video_frames':1501,'video_fps':50,'qualified_seeds':[r['seed'] for r in q['results']],
    'final_qualification_passes':32,'development_gate':'at least 12 disjoint noisy development passes plus nominal',
    'scope':'finite simulation reset-noise evidence, no general disturbance or hardware guarantee',
    'files_sha256':hashes}
(target/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
archive = root/f'output/2d-cartpole-{a.links}-link-evidence.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for path in release_files():
        z.write(path,'2d/'+str(path.relative_to(target)))
print(json.dumps({'files':len(hashes),'evidence_zip':str(archive),'zip_bytes':archive.stat().st_size,
                  'video_sha256':hashlib.sha256(video.read_bytes()).hexdigest()}))
