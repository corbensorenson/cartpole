"""Hash-covered incremental evidence archives with immutable prior bundles."""
import hashlib
import json
from pathlib import Path
import re
import zipfile

MAX_ASSET_BYTES = 2**31
ARCHIVE_PAYLOAD_BYTES = 1_800_000_000


def partition_archive_files(root,files,byte_budget=ARCHIVE_PAYLOAD_BYTES,priority_prefixes=()):
    """Keep runtime/qualification in primary; move only evidence into supplements."""
    root=Path(root)
    assert byte_budget>0
    mandatory={n:h for n,h in files.items() if not n.startswith('evidence/') or n.startswith(tuple(priority_prefixes))}
    size=lambda n:(root/n).stat().st_size
    total=sum(size(n) for n in mandatory)
    assert total<=byte_budget,'mandatory primary runtime/qualification exceeds archive payload budget'
    parts=[dict(mandatory)]
    for name,digest in sorted(files.items()):
        if name in mandatory:
            continue
        count=size(name)
        assert count<=byte_budget,'single evidence file exceeds archive payload budget'
        if total+count>byte_budget:
            parts.append({});total=0
        parts[-1][name]=digest;total+=count
    return parts


def create_evidence_archives(root,archive,manifest,byte_budget=ARCHIVE_PAYLOAD_BYTES,
                             priority_prefixes=(),release_base_url=None,compression=zipfile.ZIP_DEFLATED):
    """Create hash-bound supplements and then primary; never overwrite an asset."""
    root,archive=Path(root),Path(archive)
    assert compression in (zipfile.ZIP_STORED,zipfile.ZIP_DEFLATED),'unsupported archive compression'
    assert not archive.exists(),'count-specific archives are create-only'
    direct=manifest.get('archive_files_sha256',manifest['files_sha256'])
    assert all(manifest['files_sha256'].get(n)==h for n,h in direct.items())
    parts=partition_archive_files(root,direct,byte_budget,priority_prefixes)
    count=manifest['accepted_swingup_links']
    stem=manifest.get('archive_name_stem',f'2d-cartpole-{count}-link')
    assert stem in (f'2d-cartpole-{count}-link',f'2d-cartpole-{count}-link-immediate'),'invalid archive namespace'
    paths=[archive]+[archive.parent/f'{stem}-evidence-part{i}.zip' for i in range(2,len(parts)+1)]
    assert all(not p.exists() for p in paths),'all archive parts are create-only'
    for name,digest in direct.items():
        with (root/name).open('rb') as stream:
            assert hashlib.file_digest(stream,'sha256').hexdigest()==digest,(name,'source changed before archive creation')
    def write(path,files,include_manifest=False):
        with zipfile.ZipFile(path,'x',compression) as z:
            if include_manifest:
                z.write(root/'manifest.json','2d/manifest.json')
            for name in sorted(files):
                z.write(root/name,'2d/'+name)
        assert path.stat().st_size<MAX_ASSET_BYTES,'archive part exceeds release asset limit'
    supplements=[]
    for path,files in zip(paths[1:],parts[1:]):
        assert all(name.startswith('evidence/') for name in files)
        write(path,files)
        with path.open('rb') as stream:
            digest=hashlib.file_digest(stream,'sha256').hexdigest()
        item=dict(name=path.name,sha256=digest,bytes=path.stat().st_size,files_sha256=files)
        if release_base_url is not None:
            item['url']=release_base_url.rstrip('/')+'/'+path.name
        supplements.append(item)
    if supplements:
        manifest.update(archive_schema=3,archive_files_sha256=parts[0],supplementary_evidence_bundles=supplements,
            archive_scope='complete current runtime and all new/changed evidence across listed current archives; unchanged historical evidence is covered by immutable prior bundles')
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    write(archive,parts[0],True)
    return dict(primary_files=len(parts[0]),supplementary_files=sum(map(len,parts[1:])),
        archive_parts=len(parts),direct_files=len(direct))


def select_archive_files(current,previous):
    old = previous.get('files_sha256',{})
    return {name:digest for name,digest in current.items()
            if not name.startswith('evidence/') or old.get(name)!=digest}


def verify_archive(archive,expected):
    """Verify all current files directly or through hash-bound prior bundles."""
    archive = Path(archive)
    handles,manifests,prior_hashes = {},{},{}
    supplemental_handles,supplemental_hashes,supplemental_files = {},{},{}

    def open_bundle(path):
        path = Path(path)
        if path not in handles:
            z = zipfile.ZipFile(path)
            handles[path]=z
            names = z.namelist()
            assert len(names)==len(set(names)),'duplicate archive member'
            manifest = json.loads(z.read('2d/manifest.json'))
            direct = manifest.get('archive_files_sha256',manifest['files_sha256'])
            assert set(names)=={'2d/'+n for n in direct}|{'2d/manifest.json'},'archive coverage mismatch'
            assert all(manifest['files_sha256'].get(n)==h for n,h in direct.items())
            lookup={}
            part_names=set()
            for part in manifest.get('supplementary_evidence_bundles',[]):
                assert manifest.get('archive_schema')==3 and part['files_sha256'],'invalid supplementary schema/empty part'
                count=manifest['accepted_swingup_links']
                stem=manifest.get('archive_name_stem',f'2d-cartpole-{count}-link')
                assert stem in (f'2d-cartpole-{count}-link',f'2d-cartpole-{count}-link-immediate'),'invalid archive namespace'
                match=re.fullmatch(rf"{re.escape(stem)}-evidence-part([0-9]+)\.zip",part['name'])
                assert match and int(match[1])>=2 and str(int(match[1]))==match[1],'invalid supplementary archive name'
                assert part['name'] not in part_names,'duplicate supplementary archive'
                part_names.add(part['name'])
                for n,h in part['files_sha256'].items():
                    assert n.startswith('evidence/'),'runtime/source must remain in primary archive'
                    assert n not in direct and n not in lookup,'overlapping supplementary coverage'
                    assert manifest['files_sha256'].get(n)==h,'supplementary file absent from inventory'
                    lookup[n]=part
            supplemental_files[path]=lookup
            handles[path],manifests[path] = z,manifest
        return handles[path],manifests[path]

    def verify_file(path,name,digest):
        z,m = open_bundle(path)
        assert m['files_sha256'].get(name)==digest,(name,'manifest mismatch')
        direct = m.get('archive_files_sha256',m['files_sha256'])
        if name in direct:
            assert hashlib.sha256(z.read('2d/'+name)).hexdigest()==digest,name
            return
        if name in supplemental_files[path]:
            part=supplemental_files[path][name]
            part_path=path.parent/part['name']
            if part_path not in supplemental_handles:
                assert part_path.stat().st_size==part['bytes'],'supplementary bundle size mismatch'
                with part_path.open('rb') as stream:
                    actual=hashlib.file_digest(stream,'sha256').hexdigest()
                assert actual==part['sha256'],'supplementary bundle digest mismatch'
                pz=zipfile.ZipFile(part_path)
                supplemental_handles[part_path]=pz
                entries=pz.namelist()
                assert len(entries)==len(set(entries)),'duplicate supplementary member'
                assert set(entries)=={'2d/'+n for n in part['files_sha256']},'supplementary archive coverage mismatch'
                supplemental_hashes[part_path]=actual
            assert hashlib.sha256(supplemental_handles[part_path].read('2d/'+name)).hexdigest()==digest,name
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
        supplementary=len(supplemental_files[archive])
        return dict(direct_files=len(direct)+supplementary,primary_files=len(direct),supplementary_files=supplementary,
                    inherited_files=len(manifest['files_sha256'])-len(direct)-supplementary,
                    supplementary_bundles_verified=len(supplemental_hashes),prior_bundles_verified=len(prior_hashes))
    finally:
        for z in handles.values():
            z.close()
        for z in supplemental_handles.values():
            z.close()
