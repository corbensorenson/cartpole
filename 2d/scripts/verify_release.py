"""Verify every staged and archived release file against the frozen SHA256 list."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile
from release_archive import verify_archive
from staged_files import verify_staged_files

p = argparse.ArgumentParser()
p.add_argument('--links',type=int,required=True)
a = p.parse_args()
root = Path(__file__).resolve().parents[1]
checkout = root/'publication'
manifest = json.loads((checkout/'2d/manifest.json').read_text())
assert manifest['accepted_swingup_links'] == a.links
files = manifest['files_sha256']
staged = verify_staged_files(checkout,{'2d/'+name:digest for name,digest in files.items()})
assert staged['lfs_files']==len(manifest.get('git_lfs_files',{})),'large evidence must use the declared LFS representation'
archive = root/f'output/2d-cartpole-{a.links}-link-evidence.zip'
coverage = verify_archive(archive,manifest)
print(f'Verified {staged} and complete archive coverage: {coverage}; no mismatches.')
