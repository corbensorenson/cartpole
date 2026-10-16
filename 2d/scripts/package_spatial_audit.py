"""Publish a self-contained audit without changing count-specific release files."""
import hashlib
import json
from pathlib import Path
import shutil

root = Path(__file__).resolve().parents[1]
target = root/'publication/2d/audits/spatial_freedom_001'
target.mkdir(parents=True,exist_ok=False)
ignore = shutil.ignore_patterns('__pycache__','.pytest_cache','*.egg-info')
for name in ('src','tests'):
    shutil.copytree(root/name,target/name,ignore=ignore)
for name in ('pyproject.toml','requirements-lock.txt'):
    shutil.copy2(root/name,target/name)
(target/'research').mkdir()
shutil.copy2(root/'research/plan.md',target/'research/plan.md')
(target/'scripts').mkdir()
for name in ('audit_spatial.py','nonplanar_demo.py'):
    shutil.copy2(root/'scripts'/name,target/'scripts'/name)
shutil.copytree(root/'output/spatial_audit_001',target/'physics_audit',ignore=ignore)
shutil.copytree(root/'output/nonplanar_demo_n6_001',target/'controlled_demo',ignore=ignore)
shutil.copytree(root/'output/qualify_n7_001',target/'evidence/qualify_n7_001',ignore=ignore)
shutil.copy2(root/'output/2d-six-link-controlled-xy-orbit.mp4',target/'controlled_xy_orbit.mp4')
shutil.copy2(root/'output/2d-six-link-controlled-xy-orbit-preview.gif',target/'controlled_xy_orbit_preview.gif')
shutil.copy2(root/'research/spatial_freedom_audit.md',target/'README.md')
files = {p.relative_to(target).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
         for p in sorted(target.rglob('*')) if p.is_file()}
manifest = {'kind':'spatial physics and controlled XY demonstration audit',
    'accepted_count_at_audit_checkpoint':6,'raw_seven_link_candidate_status':'31/32 failed first qualification; not accepted by this audit',
    'physics_audit':'physics_audit/summary.json','controlled_demo':'controlled_demo/summary.json',
    'video':'controlled_xy_orbit.mp4','video_source':'controlled_demo/seed20261116.npz',
    'files_sha256':files}
(target/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps({'target':str(target),'files':len(files)}))
