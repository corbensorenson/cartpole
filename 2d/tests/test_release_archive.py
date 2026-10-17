import hashlib
import json
import zipfile
import pytest
from scripts.release_archive import select_archive_files,verify_archive,create_evidence_archives,partition_archive_files


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


def multipart_bundle(folder,count,files,previous=None):
    root=folder/f'content_{count}';root.mkdir()
    if previous is not None:
        old,pm,raw=previous
        files=dict(files,**{'manifests/prior.json':raw})
    for name,data in files.items():
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    manifest=dict(accepted_swingup_links=count,files_sha256={n:digest(b) for n,b in files.items()})
    if previous is not None:
        manifest.update(archive_schema=2,archive_files_sha256=select_archive_files(manifest['files_sha256'],pm),
            prior_evidence_bundle=dict(links=pm['accepted_swingup_links'],manifest='manifests/prior.json',
                manifest_sha256=digest(raw),sha256=digest(old.read_bytes()),bytes=old.stat().st_size))
    path=folder/f'2d-cartpole-{count}-link-evidence.zip'
    create_evidence_archives(root,path,manifest,byte_budget=3000,
        priority_prefixes=('evidence/qualification/',),release_base_url='https://example.invalid/release')
    return path,manifest,(root/'manifest.json').read_bytes()


def multipart_files():
    return {'src/run.py':b'current runtime','evidence/qualification/controller':b'frozen controller',
            'evidence/new/a':b'a'*1900,'evidence/new/b':b'b'*1900,'evidence/new/c':b'c'*1900}


def test_multipart_covers_all_files_keeps_runtime_qualification_and_does_not_mutate_prior(tmp_path):
    first=bundle(tmp_path,1,{'src/run.py':b'old','evidence/old/tape':b'old evidence'})
    old_bytes=first[0].read_bytes()
    second=multipart_bundle(tmp_path,2,dict(multipart_files(),**{'evidence/old/tape':b'old evidence'}),first)
    m=second[1]
    assert m['archive_schema']==3 and m['supplementary_evidence_bundles']
    assert 'src/run.py' in m['archive_files_sha256']
    assert 'evidence/qualification/controller' in m['archive_files_sha256']
    coverage=verify_archive(second[0],m)
    assert coverage['inherited_files']==1 and coverage['prior_bundles_verified']==1
    assert coverage['supplementary_bundles_verified']==len(m['supplementary_evidence_bundles'])
    assert first[0].read_bytes()==old_bytes


def test_later_bundle_resolves_inherited_files_from_prior_supplements(tmp_path):
    first=multipart_bundle(tmp_path,1,multipart_files())
    files=dict(multipart_files(),**{'src/run.py':b'new code','evidence/later':b'later'})
    second=multipart_bundle(tmp_path,2,files,first)
    coverage=verify_archive(second[0],second[1])
    assert coverage['prior_bundles_verified']==1 and coverage['inherited_files']==4
    assert coverage['supplementary_bundles_verified']>=1


@pytest.mark.parametrize('damage',['missing','corrupt'])
def test_multipart_rejects_missing_or_modified_supplement(tmp_path,damage):
    current=multipart_bundle(tmp_path,1,multipart_files())
    part=tmp_path/current[1]['supplementary_evidence_bundles'][0]['name']
    if damage=='missing':
        part.unlink()
    else:
        with part.open('ab') as stream:
            stream.write(b'corruption')
    with pytest.raises((AssertionError,FileNotFoundError)):
        verify_archive(current[0],current[1])


def test_archive_creator_is_create_only_and_checks_source_before_writing(tmp_path):
    root=tmp_path/'source';root.mkdir();(root/'run.py').write_bytes(b'right')
    m=dict(accepted_swingup_links=1,files_sha256={'run.py':digest(b'wrong')})
    output=tmp_path/'2d-cartpole-1-link-evidence.zip'
    with pytest.raises(AssertionError,match='source changed'):
        create_evidence_archives(root,output,m,byte_budget=100)
    assert not output.exists()
    m['files_sha256']['run.py']=digest(b'right')
    create_evidence_archives(root,output,m,byte_budget=100)
    before=output.read_bytes()
    with pytest.raises(AssertionError,match='create-only'):
        create_evidence_archives(root,output,m,byte_budget=100)
    assert output.read_bytes()==before
    assert verify_archive(output,m)['inherited_files']==0


@pytest.mark.parametrize('mutation',['traversal','overlap','runtime'])
def test_multipart_rejects_invalid_or_overlapping_part_inventory(tmp_path,mutation):
    path,m,_=multipart_bundle(tmp_path,1,multipart_files())
    part=m['supplementary_evidence_bundles'][0]
    if mutation=='traversal':
        part['name']='../'+part['name']
    elif mutation=='overlap':
        part['files_sha256']['src/run.py']=m['files_sha256']['src/run.py']
    else:
        part['files_sha256']['scripts/hidden.py']=digest(b'bad')
        m['files_sha256']['scripts/hidden.py']=digest(b'bad')
    with zipfile.ZipFile(path) as z:
        entries={n:z.read(n) for n in z.namelist() if n!='2d/manifest.json'}
    with zipfile.ZipFile(path,'w') as z:
        z.writestr('2d/manifest.json',json.dumps(m))
        for n,b in entries.items():
            z.writestr(n,b)
    with pytest.raises(AssertionError):
        verify_archive(path,m)


def test_partition_refuses_oversize_mandatory_or_single_evidence(tmp_path):
    root=tmp_path/'source';root.mkdir();(root/'runtime').write_bytes(b'x'*100)
    with pytest.raises(AssertionError,match='mandatory'):
        partition_archive_files(root,{'runtime':digest(b'x'*100)},50)
    evidence=root/'evidence';evidence.mkdir();(evidence/'large').write_bytes(b'y'*100)
    with pytest.raises(AssertionError,match='single evidence'):
        partition_archive_files(root,{'evidence/large':digest(b'y'*100)},50)
