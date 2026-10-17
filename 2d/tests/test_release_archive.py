import hashlib
import json
import zipfile
import pytest
from scripts.release_archive import select_archive_files,verify_archive


def digest(data):
    return hashlib.sha256(data).hexdigest()


def bundle(folder,count,files,previous=None):
    manifest = dict(accepted_swingup_links=count,files_sha256={n:digest(b) for n,b in files.items()})
    if previous is not None:
        prior,pm,prior_bytes = previous
        name = 'manifests/prior.json'
        files = dict(files,**{name:prior_bytes})
        manifest['files_sha256'][name] = digest(prior_bytes)
        manifest.update(archive_schema=2,archive_files_sha256=select_archive_files(manifest['files_sha256'],pm),
            prior_evidence_bundle=dict(links=pm['accepted_swingup_links'],manifest=name,manifest_sha256=digest(prior_bytes),
                                       sha256=digest(prior.read_bytes()),bytes=prior.stat().st_size))
    data = json.dumps(manifest,indent=2).encode()
    path = folder/f'2d-cartpole-{count}-link-evidence.zip'
    direct = manifest.get('archive_files_sha256',manifest['files_sha256'])
    with zipfile.ZipFile(path,'w') as z:
        z.writestr('2d/manifest.json',data)
        for name in direct:
            z.writestr('2d/'+name,files[name])
    return path,manifest,data


def test_incremental_archive_covers_history_and_keeps_runtime_and_changed_evidence(tmp_path):
    first = bundle(tmp_path,1,{'src/a.py':b'code','evidence/old/tape':b'old','evidence/changed':b'v1'})
    files = {'src/a.py':b'code','evidence/old/tape':b'old','evidence/changed':b'v2','evidence/new/tape':b'new'}
    second = bundle(tmp_path,2,files,first)
    selected = second[1]['archive_files_sha256']
    assert 'src/a.py' in selected and 'evidence/changed' in selected and 'evidence/new/tape' in selected
    assert 'evidence/old/tape' not in selected
    coverage = verify_archive(second[0],second[1])
    assert coverage['inherited_files']==1 and coverage['prior_bundles_verified']==1


def test_incremental_archive_resolves_more_than_one_prior_bundle(tmp_path):
    first = bundle(tmp_path,1,{'src/a.py':b'v1','evidence/a':b'a'})
    second = bundle(tmp_path,2,{'src/a.py':b'v2','evidence/a':b'a','evidence/b':b'b'},first)
    third = bundle(tmp_path,3,{'src/a.py':b'v3','evidence/a':b'a','evidence/b':b'b','evidence/c':b'c'},second)
    coverage = verify_archive(third[0],third[1])
    assert coverage['prior_bundles_verified']==2 and coverage['inherited_files']==2


def test_incremental_archive_rejects_modified_prior_bundle(tmp_path):
    first = bundle(tmp_path,1,{'src/a.py':b'code','evidence/old/tape':b'old'})
    second = bundle(tmp_path,2,{'src/a.py':b'code','evidence/old/tape':b'old','evidence/new':b'new'},first)
    with first[0].open('ab') as stream:
        stream.write(b'corruption')
    with pytest.raises(AssertionError):
        verify_archive(second[0],second[1])


def test_incremental_archive_rejects_incomplete_current_evidence(tmp_path):
    first = bundle(tmp_path,1,{'src/a.py':b'code'})
    second = bundle(tmp_path,2,{'src/a.py':b'code','evidence/new':b'new'},first)
    with zipfile.ZipFile(second[0]) as z:
        entries = {n:z.read(n) for n in z.namelist() if n!='2d/evidence/new'}
    with zipfile.ZipFile(second[0],'w') as z:
        for n,b in entries.items():
            z.writestr(n,b)
    with pytest.raises(AssertionError):
        verify_archive(second[0],second[1])
