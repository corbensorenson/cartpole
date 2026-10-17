import hashlib
from pathlib import Path
import subprocess
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from staged_files import verify_staged_files


def git(root,*args,input=None):
    return subprocess.check_output(['git','-C',str(root),*args],input=input,stderr=subprocess.DEVNULL)


def setup(root,content,pointer=False):
    git(root,'init','-q')
    path=root/'evidence.bin';path.write_bytes(content)
    digest=hashlib.sha256(content).hexdigest()
    if pointer:
        body=f'version https://git-lfs.github.com/spec/v1\noid sha256:{digest}\nsize {len(content)}\n'.encode()
        oid=git(root,'hash-object','-w','--stdin',input=body).decode().strip()
        git(root,'update-index','--add','--cacheinfo',f'100644,{oid},evidence.bin')
        stored=root/'.git/lfs/objects'/digest[:2]/digest[2:4]/digest
        stored.parent.mkdir(parents=True);stored.write_bytes(content)
    else:
        git(root,'-c','core.autocrlf=false','add','--','evidence.bin')
    return path,digest


def test_streaming_staged_binary_blob_with_embedded_newlines_and_nuls(tmp_path):
    path,digest=setup(tmp_path,b'\x00abc\n\r\xff'*400000)
    assert verify_staged_files(tmp_path,{path.name:digest})==dict(staged_files=1,lfs_files=0)


def test_lfs_requires_materialized_hash(tmp_path):
    path,digest=setup(tmp_path,b'\x00native cache\xff',pointer=True)
    assert verify_staged_files(tmp_path,{path.name:digest})==dict(staged_files=1,lfs_files=1)
    path.write_bytes(b'\x00native CACHE\xff')
    with pytest.raises(AssertionError,match='materialized LFS hash'):
        verify_staged_files(tmp_path,{path.name:digest})


def test_lfs_pointer_cannot_replace_missing_or_corrupt_evidence(tmp_path):
    path,digest=setup(tmp_path,b'scientific data',pointer=True)
    obj=tmp_path/'.git/lfs/objects'/digest[:2]/digest[2:4]/digest
    obj.write_bytes(b'Scientific data')
    with pytest.raises(AssertionError,match='local LFS object hash'):
        verify_staged_files(tmp_path,{path.name:digest})
    with pytest.raises(AssertionError,match='LFS identity'):
        verify_staged_files(tmp_path,{path.name:'0'*64})
