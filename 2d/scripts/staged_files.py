"""Verify staged Git blobs with bounded memory, including Git LFS evidence."""
import hashlib
from pathlib import Path
import re
import subprocess
import tempfile


def verify_staged_files(checkout,files):
    checkout=Path(checkout);lfs_count=0
    with tempfile.TemporaryFile() as queries,tempfile.TemporaryFile() as errors:
        queries.write(('\n'.join(':'+name for name in files)+'\n').encode());queries.seek(0)
        with subprocess.Popen(['git','-C',str(checkout),'cat-file','--batch'],stdin=queries,
                stdout=subprocess.PIPE,stderr=errors) as process:
            for name,expected in files.items():
                header=process.stdout.readline().decode().split()
                assert len(header)==3 and header[1]=='blob',(name,header)
                remaining=int(header[2]);small=bytearray();digest=hashlib.sha256()
                while remaining:
                    chunk=process.stdout.read(min(8*1024*1024,remaining))
                    assert chunk,(name,'truncated staged blob')
                    if int(header[2])<=1024:
                        small.extend(chunk)
                    digest.update(chunk);remaining-=len(chunk)
                assert process.stdout.read(1)==b'\n',(name,'invalid staged blob terminator')
                if digest.hexdigest()==expected:
                    continue
                pointer=re.fullmatch(rb'version https://git-lfs.github.com/spec/v1\noid sha256:([0-9a-f]{64})\nsize ([0-9]+)\n',bytes(small))
                assert pointer,(name,'staged hash mismatch')
                oid,size=pointer[1].decode(),int(pointer[2]);assert oid==expected,(name,'LFS identity mismatch')
                actual=checkout/name
                assert actual.stat().st_size==size,(name,'materialized LFS size mismatch')
                with actual.open('rb') as stream:
                    assert hashlib.file_digest(stream,'sha256').hexdigest()==expected,(name,'materialized LFS hash mismatch')
                git_path=subprocess.check_output(['git','-C',str(checkout),'rev-parse','--git-path','lfs/objects'],text=True).strip()
                store=Path(git_path)
                if not store.is_absolute():
                    store=checkout/store
                object_path=store/oid[:2]/oid[2:4]/oid
                assert object_path.stat().st_size==size,(name,'local LFS object size mismatch')
                with object_path.open('rb') as stream:
                    assert hashlib.file_digest(stream,'sha256').hexdigest()==expected,(name,'local LFS object hash mismatch')
                lfs_count+=1
            assert process.stdout.read(1)==b'','unexpected extra staged output'
            assert process.wait()==0,'git cat-file failed'
    return dict(staged_files=len(files),lfs_files=lfs_count)
