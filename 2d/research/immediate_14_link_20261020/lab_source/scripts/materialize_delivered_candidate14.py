"""Rebuild only candidate path metadata from an extracted immutable payload."""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):return json.loads(Path(path).read_text())


def write(path,value):Path(path).write_text(json.dumps(value,indent=2),encoding='utf-8')


def materialize(payload,out,require_hardlinks=False):
    """No original path is resolved or opened, and no numerical code is invoked."""
    payload,out=Path(payload).resolve(),Path(out).resolve();assert payload.is_dir() and not out.exists()
    template=read(payload/'relative-profile-template.json');original=read(payload/'original-candidate/frozen_manifest.json')
    assert digest(payload/'original-candidate/frozen_manifest.json')==template['original_candidate_manifest_sha256']
    assert template['controller_identity']==original['controller_identity'] and len(template['profiles'])==15
    out.mkdir(parents=True);linked=copied=0;profiles=[]
    def inside(relative):
        assert not Path(relative).is_absolute() and '..' not in Path(relative).parts
        path=(payload/relative).resolve();assert path.is_relative_to(payload);return path
    def install(src,target,expected):
        nonlocal linked,copied
        assert digest(src)==expected and not target.exists();target.parent.mkdir(parents=True,exist_ok=True)
        try:os.link(src,target);assert os.path.samefile(src,target);linked+=src.stat().st_size
        except OSError:
            if require_hardlinks:raise
            shutil.copyfile(src,target);copied+=src.stat().st_size
    for index,item in enumerate(template['profiles']):
        old=item['original_metadata'];new=copy.deepcopy(old)
        for role,files in item['packet_roles'].items():
            target=out/'packets'/f'profile{index}'/role;target.mkdir(parents=True)
            assert role in ('guide','after') and len(files)==4
            for entry in files:
                assert Path(entry['name']).name==entry['name']
                install(inside(entry['blob_path']),target/entry['name'],entry['sha256'])
            new[role]=str(target)
        for field,relative in item['file_fields'].items():
            assert field in ('actual','prefix_trace','future_trace','inventory');new[field]=str(inside(relative))
        new['input_sha256']={str(inside(relative)):value for relative,value in item['relative_input_sha256'].items()}
        diff={key for key in set(old)|set(new) if old.get(key)!=new.get(key)}
        assert diff<=set(template['allowed_profile_path_changes'])
        assert set(old['input_sha256'].values())==set(new['input_sha256'].values());profiles.append(new)
    (out/'helpers').mkdir()
    for name,value in original['helper_sha256'].items():
        assert Path(name).name==name;install(inside('scripts/'+name),out/'helpers'/name,value)
    write(out/'profiles.json',profiles);view=copy.deepcopy(original)
    view.update(runtime=str(inside('numerical-runtime')),profiles_sha256=digest(out/'profiles.json'))
    assert view['identity_payload']==original['identity_payload']
    assert {key for key in set(view)|set(original) if view.get(key)!=original.get(key)}=={'runtime','profiles_sha256'}
    write(out/'frozen_manifest.json',view);write(out/'completed.json',dict(status='completed'))
    proof=dict(passed=True,native_steps=0,controller_identity=original['controller_identity'],
        profiles=15,template_sha256=digest(payload/'relative-profile-template.json'),
        logical_hardlinked_bytes=linked,copied_payload_bytes=copied,original_path_resolution_or_io=False,
        scope='relative payload to path-only candidate view; no controller callbacks, solver, native steps or qualification')
    write(out/'materialization.json',proof);return proof
