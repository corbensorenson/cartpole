"""Build and independently reproduce the native expert's explicit F64 arithmetic."""
import ctypes as ct
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import mujoco as mj
import numpy as np
from spatialpole import Config,SpatialCartpole

ROOT=Path(__file__).resolve().parents[1]
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(out):
    target=out/'build';target.mkdir();source=out/'native_source';source.mkdir()
    for name in ('spatial_core.h','trajectory_core.h','trajectory_expert.h','verify_expert.cpp'):
        shutil.copy2(ROOT/'native/puffer_spatial'/name,source/name)
    sdk=Path(mj.__file__).parent
    if os.name=='nt':
        from setuptools._distutils._msvccompiler import _get_vc_env
        env={k.lower():v for k,v in {**os.environ,**_get_vc_env('x64')}.items()}
        cl=shutil.which('cl.exe',path=env['path']);dump=shutil.which('dumpbin.exe',path=env['path']);lib=shutil.which('lib.exe',path=env['path'])
        assert cl and dump and lib
        library=sdk/'mujoco.dll'
        exports=subprocess.check_output([dump,'/exports',str(library)],env=env,text=True)
        names=re.findall(r'^\s+\d+\s+[0-9A-F]+\s+[0-9A-F]+\s+(\w+)\s*$',exports,re.MULTILINE);assert 'mj_step' in names
        definition=target/'mujoco.def';definition.write_text('LIBRARY mujoco.dll\nEXPORTS\n'+'\n'.join(names)+'\n')
        binary=target/'expert11.dll'
        commands=[[lib,'/nologo','/machine:x64',f'/def:{definition}',f'/out:{target/"mujoco.lib"}'],
            [cl,'/nologo','/LD','/O2','/fp:strict','/EHsc','/MD','/std:c++17','/DSPATIAL_LINKS=11',f'/I{sdk/"include"}',f'/I{source}',
             str(source/'verify_expert.cpp'),str(target/'mujoco.lib'),f'/Fo{target/"expert.obj"}','/link',f'/OUT:{binary}']]
    else:
        env=dict(os.environ);compiler=shutil.which('g++');assert compiler
        library=sdk/'libmujoco.so.3.3.5';binary=target/'expert11.so'
        commands=[[compiler,'-std=c++17','-O2','-shared','-fPIC','-fno-fast-math','-ffp-contract=off','-DSPATIAL_LINKS=11',
            f'-I{sdk/"include"}',f'-I{source}',str(source/'verify_expert.cpp'),str(library),f'-Wl,-rpath,{sdk}','-o',str(binary)]]
    with (target/'build.log').open('w',encoding='utf-8') as stream:
        for command in commands:
            result=subprocess.run(command,cwd=target,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            stream.write(result.stdout)
            if result.returncode:raise RuntimeError(f'Build failed: {target/"build.log"}')
    (target/'build.json').write_text(json.dumps(dict(commands=commands,binary_name=binary.name,binary_sha256=sha(binary),
        mujoco_version=mj.__version__,mujoco_library_sha256=sha(library),source_sha256={p.name:sha(p) for p in source.iterdir()}),indent=2))
    return binary


def load_dll(binary):
    directory=os.add_dll_directory(str(Path(mj.__file__).parent)) if os.name=='nt' else None
    dll=ct.CDLL(str(binary));P=ct.c_void_p
    D=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS');F=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS')
    U=np.ctypeslib.ndpointer(dtype=np.uint8,flags='C_CONTIGUOUS')
    signatures={'ex_create':([ct.c_char_p,ct.c_char_p,ct.c_char_p,U,ct.c_double],P),'ex_set':([P,D,ct.c_int],ct.c_int),
        'ex_query':([P,D,D,D,F],ct.c_int),'ex_advance':([P,F],ct.c_int),'ex_read':([P,D,F,F,D],None),
        'ex_copy':([P,D,D,D],None),'ex_destroy':([P],None)}
    for name,(args,result) in signatures.items():getattr(dll,name).argtypes=args;getattr(dll,name).restype=result
    return dll,directory


class ExpertOracle:
    """Scalar arithmetic oracle plus separately available original NumPy law."""
    def __init__(self,reference,expert):
        self.geometry=SpatialCartpole(Config(n_links=11))
        data=Path(reference).read_bytes();off=36;self.reference=[]
        for shape in ((1501,81),(1500,2),(1501,11,3),(1501,11,3)):
            count=int(np.prod(shape));self.reference.append(np.frombuffer(data,dtype='<f8',count=count,offset=off).reshape(shape));off+=count*8
        assert off==len(data);self.states,self.actions,self.directions,self.omega=self.reference
        data=Path(expert).read_bytes();assert data[24:56].hex()==sha(reference);off=56;self.expert=[]
        for shape in ((1500,2),(1500,2,70),(1501,11,3,3)):
            count=int(np.prod(shape));self.expert.append(np.frombuffer(data,dtype='<f8',count=count,offset=off).reshape(shape));off+=count*8
        assert off==len(data);self.feed,self.gains,self.rotations=self.expert

    def force(self,state,phase):
        self.geometry.set_state(state);observed=self.geometry.data.xmat[self.geometry.link_ids].reshape(-1,3,3)
        reference=self.states[phase];aligned=reference.copy();parent=np.array([1.,0.,0.,0.]);angles=[];cosines=[];sines=[]
        for i in range(11):
            R=self.rotations[phase,i];O=observed[i];a=b=c=d=0.
            for k in range(3):
                a+=float(R[k,1])*float(O[k,0]);b+=float(R[k,0])*float(O[k,1])
                c+=float(R[k,0])*float(O[k,0]);d+=float(R[k,1])*float(O[k,1])
            angle=math.atan2(a-b,c+d);angles.append(angle);cosines.append(math.cos(angle));sines.append(math.sin(angle))
            child=np.array([math.cos(angle/2),0.,0.,math.sin(angle/2)]);inverse=parent.copy();inverse[1:]*=-1
            temp=np.empty(4);mj.mju_mulQuat(temp,inverse,reference[2+4*i:6+4*i]);mj.mju_mulQuat(aligned[2+4*i:6+4*i],temp,child)
            matrix=np.empty(9);mj.mju_quat2Mat(matrix,child);matrix=matrix.reshape(3,3);j=48+3*i
            for row in range(3):
                value=0.
                for k in range(3):value+=float(matrix[k,row])*float(reference[j+k])
                aligned[j+row]=value
            parent=child
        error=np.empty(70);mj.mj_differentiatePos(self.geometry.model,error[:35],1.,aligned[:46],state[:46]);error[35:]=state[46:]-aligned[46:]
        for i in range(11):
            for half in range(2):
                j=35*half+2+3*i;x=float(error[j]);y=float(error[j+1]);cs=cosines[i];sn=sines[i]
                error[j]=cs*x-sn*y;error[j+1]=sn*x+cs*y
        raw=np.empty(2)
        for axis in range(2):
            correction=0.
            for j in range(70):correction+=float(self.gains[phase,axis,j])*float(error[j])
            raw[axis]=float(self.actions[phase,axis])+float(self.feed[phase,axis])-correction
        return raw,np.asarray(angles),error

    def numpy_force(self,state,phase):
        from spatialpole.affine_value import axial_aligned_affine_force
        return axial_aligned_affine_force(self.geometry,state,self.states[phase],self.actions[phase],self.feed[phase],self.gains[phase])

    def close(self):self.geometry.close()
