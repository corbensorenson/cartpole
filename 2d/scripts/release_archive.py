"""Hash-covered incremental evidence archives with immutable prior bundles."""
import hashlib
import json
from pathlib import Path
import zipfile


def select_archive_files(current,previous):
    old = previous.get('files_sha256',{})
    return {name:digest for name,digest in current.items()
            if not name.startswith('evidence/') or old.get(name)!=digest}


def verify_archive(archive,expected):
    """Verify all current files directly or through hash-bound prior bundles."""
    archive = Path(archive)
    handles,manifests,prior_hashes = {},{},{}

    def open_bundle(path):
        path = Path(path)
        if path not in handles:
            z = zipfile.ZipFile(path)
            names = z.namelist()
            assert len(names)==len(set(names)),'duplicate archive member'
            manifest = json.loads(z.read('2d/manifest.json'))
            direct = manifest.get('archive_files_sha256',manifest['files_sha256'])
            assert set(names)=={'2d/'+n for n in direct}|{'2d/manifest.json'},'archive coverage mismatch'
            assert all(manifest['files_sha256'].get(n)==h for n,h in direct.items())
            handles[path],manifests[path] = z,manifest
        return handles[path],manifests[path]

    def verify_file(path,name,digest):
        z,m = open_bundle(path)
        assert m['files_sha256'].get(name)==digest,(name,'manifest mismatch')
        direct = m.get('archive_files_sha256',m['files_sha256'])
        if name in direct:
            assert hashlib.sha256(z.read('2d/'+name)).hexdigest()==digest,name
            return
        assert name.startswith('evidence/'),'runtime/source file cannot be inherited'
        prior = m['prior_evidence_bundle']
        assert 0<prior['links']<m['accepted_swingup_links'],'invalid prior-bundle order'
        prior_path = archive.parent/f"2d-cartpole-{prior['links']}-link-evidence.zip"
        if prior_path not in prior_hashes:
            assert prior_path.stat().st_size==prior['bytes']
            with prior_path.open('rb') as stream:
                actual = hashlib.file_digest(stream,'sha256').hexdigest()
            assert actual==prior['sha256'],'prior bundle digest mismatch'
            prior_hashes[prior_path] = actual
        pz,pm = open_bundle(prior_path)
        saved_manifest = z.read('2d/'+prior['manifest'])
        assert hashlib.sha256(saved_manifest).hexdigest()==prior['manifest_sha256']
        assert saved_manifest==pz.read('2d/manifest.json'),'prior manifest bytes differ'
        assert pm['files_sha256'].get(name)==digest,'inherited file absent from prior inventory'
        verify_file(prior_path,name,digest)

    try:
        _,manifest = open_bundle(archive)
        assert manifest==expected
        for name,digest in manifest['files_sha256'].items():
            verify_file(archive,name,digest)
        direct = manifest.get('archive_files_sha256',manifest['files_sha256'])
        return dict(direct_files=len(direct),inherited_files=len(manifest['files_sha256'])-len(direct),
                    prior_bundles_verified=len(prior_hashes))
    finally:
        for z in handles.values():
            z.close()
