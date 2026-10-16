"""Verify every staged and archived release file against the frozen SHA256 list."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

p = argparse.ArgumentParser()
p.add_argument('--links',type=int,required=True)
a = p.parse_args()
root = Path(__file__).resolve().parents[1]
checkout = root/'publication'
manifest = json.loads((checkout/'2d/manifest.json').read_text())
assert manifest['accepted_swingup_links'] == a.links
files = list(manifest['files_sha256'])
queries = ('\n'.join(':2d/'+name for name in files)+'\n').encode()
data = subprocess.check_output(['git','-C',str(checkout),'cat-file','--batch'],input=queries)
position = 0
for name in files:
    end = data.index(b'\n',position)
    header = data[position:end].decode().split()
    assert len(header)==3 and header[1]=='blob',(name,header)
    size = int(header[2])
    blob = data[end+1:end+1+size]
    position = end+size+2
    assert hashlib.sha256(blob).hexdigest()==manifest['files_sha256'][name],name
archive = root/f'output/2d-cartpole-{a.links}-link-evidence.zip'
with zipfile.ZipFile(archive) as z:
    bundled = json.loads(z.read('2d/manifest.json'))
    assert bundled == manifest
    for name,digest in manifest['files_sha256'].items():
        assert hashlib.sha256(z.read('2d/'+name)).hexdigest()==digest,name
print(f'Verified {len(files)} staged blobs and archived files; no mismatches.')
