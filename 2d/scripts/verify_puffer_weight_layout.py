"""Host-compile upstream allocation/registration to check continuous XY slabs.

No CUDA kernels or physics run. The allocator and registration bodies are taken
from the pinned sources, with only cudaMalloc/cudaMemset replaced by host stubs.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

from puffer_training_support import checked_checkout, patched_sources, sha256


def function(source, name):
    start = source.index('void ' + name + '(')
    opening = source.index('{', start)
    depth = 1
    end = opening + 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[start:end]


def harness(trainer, algorithm, padded):
    allocation = trainer[trainer.index('struct AllocEntry {'):trainer.index('// Deterministic block tree-reduce.')]
    registrations = '\n'.join(function(algorithm, x) for x in
                              ('encoder_reg_params', 'decoder_reg_params', 'mingru_reg_params'))
    muon = algorithm[algorithm.index('void muon_step('):]
    loop = muon[muon.index('    long offset = 0;'):muon.index('        if (ndim(e.shape) < 2)')]
    flat = 'a.total_bytes / sizeof(precision_t)' if padded else 'a.total_elems'
    return r'''
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#if PRECISION_BYTES == 4
using precision_t = float;
#else
using precision_t = uint16_t;
#endif
struct Prec { precision_t* data; long shape[8]; };
struct Float { float* data; long shape[8]; };
struct Long { long* data; long shape[8]; };
struct Int { int* data; long shape[8]; };
long numel(long* shape) { long n=1; for(int i=0;i<8 && shape[i];++i)n*=shape[i];return n; }
int cudaMalloc(void** p, long n) { *p=malloc(n);return *p?0:1; }
void cudaMemset(void* p,int v,long n) { memset(p,v,n); }
int cudaSuccess=0;
''' + allocation + r'''
struct EncoderWeights { Prec weight; int in_dim,out_dim; };
struct DecoderWeights { Prec weight,logstd; int output_dim,hidden_dim; bool continuous; };
struct MinGRUWeights { Prec weights[8]; int num_layers,hidden; };
''' + registrations + r'''
int optimizer_mismatches(Allocator& a) {
    Allocator* m_alloc=&a;
    struct { Allocator* param_alloc; } m_value={m_alloc};
    auto* m=&m_value;
    Prec grads={(precision_t*)a.mem,{a.total_bytes/(long)sizeof(precision_t)}};
    int mismatches=0;
''' + loop + r'''
        if ((char*)gc_ptr-(char*)grads.data != (char*)*e.data_ptr-(char*)a.mem) mismatches++;
    }
    return mismatches;
}
int main() {
    bool first=true; printf("[");
    for(int n: {1,2,8}) for(int continuous: {0,1}) {
        Allocator a={};
        EncoderWeights ew={};ew.in_dim=7+7*n;ew.out_dim=64;
        DecoderWeights dw={};dw.output_dim=continuous?2:3;dw.hidden_dim=64;dw.continuous=continuous;
        MinGRUWeights mw={};mw.num_layers=2;mw.hidden=64;
        encoder_reg_params(&ew,&a);decoder_reg_params(&dw,&a);mingru_reg_params(&mw,&a);
        alloc_create(&a);
        // Unique sentinels in every byte, including the final recurrent weights.
        for(long i=0;i<a.total_bytes;++i)((unsigned char*)a.mem)[i]=(unsigned char)(1+i%251);
        long flat_elems=FLAT_COUNT;
        long copied=flat_elems*sizeof(precision_t);
        void* restored=calloc(1,a.total_bytes); memcpy(restored,a.mem,copied);
        int roundtrip_bad=0;
        for(int i=0;i<a.num_regs;++i) {
            auto& e=a.regs[i];long off=(char*)*e.data_ptr-(char*)a.mem;
            roundtrip_bad+=memcmp((char*)restored+off,*e.data_ptr,numel(e.shape)*e.elem_size)!=0;
        }
        int opt_bad=optimizer_mismatches(a);
        if(PADDED || !continuous)assert(roundtrip_bad==0 && opt_bad==0 && copied==a.total_bytes);
        else assert(roundtrip_bad>0 && opt_bad>0 && copied<a.total_bytes);
        if(!first)printf(",");first=false;
        printf("{\"links\":%d,\"continuous\":%s,\"element_bytes\":%d,\"logical_elements\":%ld,\"physical_bytes\":%ld,\"flat_bytes\":%ld,\"roundtrip_bad_tensors\":%d,\"optimizer_bad_slices\":%d}",n,continuous?"true":"false",PRECISION_BYTES,a.total_elems,a.total_bytes,copied,roundtrip_bad,opt_bad);
        free(restored);free(a.mem);free(a.regs);
    }
    printf("]\n");return 0;
}
'''.replace('FLAT_COUNT', flat).replace('PADDED', '1' if padded else '0')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkout', required=True); p.add_argument('--out', required=True)
    args = p.parse_args(); checkout = checked_checkout(args.checkout)
    out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=False)
    trainer = (checkout/'src/pufferl.cu').read_text(encoding='utf-8')
    algorithm = (checkout/'src/algo.cu').read_text(encoding='utf-8')
    patched = patched_sources(checkout)
    if os.name == 'nt':
        from setuptools._distutils._msvccompiler import _get_vc_env
        compiler_env = {k.lower():v for k,v in {**os.environ, **_get_vc_env('x64')}.items()}
        compiler = shutil.which('cl.exe', path=compiler_env['path']); assert compiler
    else:
        compiler_env = dict(os.environ)
        compiler = shutil.which('clang++') or shutil.which('g++'); assert compiler
    results=[]
    for padded in (False,True):
        for size in (4,2):
            label=('patched' if padded else 'upstream')+f'_{size}'
            source=out/(label+'.cpp')
            text=harness(patched['pufferl.cu'] if padded else trainer,
                         patched['algo.cu'] if padded else algorithm,padded)
            source.write_text('#include <initializer_list>\n'+text,encoding='utf-8')
            binary=out/(label+('.exe' if os.name=='nt' else ''))
            if os.name=='nt':
                command=[compiler,'/nologo','/std:c++20','/O2','/EHsc',f'/DPRECISION_BYTES={size}',str(source),f'/Fe:{binary}',f'/Fo:{out/(label+".obj")}']
            else:
                command=[compiler,'-std=c++20','-O2',f'-DPRECISION_BYTES={size}',str(source),'-o',str(binary)]
            built=subprocess.run(command,cwd=out,env=compiler_env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            (out/(label+'.build.log')).write_text(built.stdout)
            if built.returncode:raise RuntimeError(f'Layout harness build failed: {label}')
            cases=json.loads(subprocess.check_output([str(binary)],text=True))
            results.append(dict(patched=padded,element_bytes=size,cases=cases,command=command,source_sha256=sha256(source)))
    report=dict(passed=True,scope='host allocation/registration, flat checkpoint/actor copy and optimizer offsets; no CUDA execution',
                upstream={x:sha256(checkout/'src'/x) for x in ('pufferl.cu','algo.cu')},
                cases=results,native_steps=0,gpu_training=False)
    (out/'summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(passed=True,cases=sum(len(x['cases']) for x in results),native_steps=0)))


if __name__=='__main__':main()
