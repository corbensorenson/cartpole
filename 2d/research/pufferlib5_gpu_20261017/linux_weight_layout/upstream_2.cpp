#include <initializer_list>

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
struct AllocEntry {
    void** data_ptr;
    long* shape;
    int elem_size;
};

struct Allocator {
    AllocEntry* regs;
    int num_regs;
    void* mem;
    long total_elems;
    long total_bytes;
};

void _alloc_register(Allocator* alloc,
        void** data_ptr, long* shape, int elem_size) {
    alloc->regs = (AllocEntry*)realloc(alloc->regs,
        (alloc->num_regs + 1) * sizeof(AllocEntry));
    alloc->regs[alloc->num_regs++] = {data_ptr, shape, elem_size};
    long n = numel(shape);
    alloc->total_elems += n;
    alloc->total_bytes = (alloc->total_bytes + 15) & ~15;
    alloc->total_bytes += n * elem_size;
}
void alloc_register(Allocator* a, Prec* t) {
    _alloc_register(a, (void**)&t->data, t->shape, sizeof(precision_t));
}
void alloc_register(Allocator* a, Float* t) {
    _alloc_register(a, (void**)&t->data, t->shape, sizeof(float));
}
void alloc_register(Allocator* a, Long* t) {
    _alloc_register(a, (void**)&t->data, t->shape, sizeof(long));
}
void alloc_register(Allocator* a, Int* t) {
    _alloc_register(a, (void**)&t->data, t->shape, sizeof(int));
}

void alloc_create(Allocator* alloc) {
    assert(cudaMalloc(&alloc->mem, alloc->total_bytes) == cudaSuccess
        && "alloc_create: cudaMalloc failed");
    cudaMemset(alloc->mem, 0, alloc->total_bytes);
    long offset = 0;
    for (int i = 0; i < alloc->num_regs; i++) {
        offset = (offset + 15) & ~15;
        *alloc->regs[i].data_ptr = (char*)alloc->mem + offset;
        offset += numel(alloc->regs[i].shape) * alloc->regs[i].elem_size;
    }
}


struct EncoderWeights { Prec weight; int in_dim,out_dim; };
struct DecoderWeights { Prec weight,logstd; int output_dim,hidden_dim; bool continuous; };
struct MinGRUWeights { Prec weights[8]; int num_layers,hidden; };
void encoder_reg_params(void* w, Allocator* alloc) {
    EncoderWeights* ew = (EncoderWeights*)w;
    ew->weight = {.shape = {ew->out_dim, ew->in_dim}};
    alloc_register(alloc, &ew->weight);
}
void decoder_reg_params(void* w, Allocator* alloc) {
    DecoderWeights* dw = (DecoderWeights*)w;
    dw->weight = {.shape = {dw->output_dim + 1, dw->hidden_dim}};
    alloc_register(alloc, &dw->weight);
    if (dw->continuous) {
        dw->logstd = {.shape = {1, dw->output_dim}};
        alloc_register(alloc,&dw->logstd);
    }
}
void mingru_reg_params(void* w, Allocator* alloc) {
    MinGRUWeights* m = (MinGRUWeights*)w;
    for (int i = 0; i < m->num_layers; i++) {
        m->weights[i] = {.shape = {3 * m->hidden, m->hidden}};
        alloc_register(alloc, &m->weights[i]);
    }
}
int optimizer_mismatches(Allocator& a) {
    Allocator* m_alloc=&a;
    struct { Allocator* param_alloc; } m_value={m_alloc};
    auto* m=&m_value;
    Prec grads={(precision_t*)a.mem,{a.total_bytes/(long)sizeof(precision_t)}};
    int mismatches=0;
    long offset = 0;
    for (int _i = 0; _i < m->param_alloc->num_regs; _i++) {
        AllocEntry& e = m->param_alloc->regs[_i];
        precision_t* gc_ptr = grads.data + offset;
        long ne = numel(e.shape);
        offset += ne;

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
        long flat_elems=a.total_elems;
        long copied=flat_elems*sizeof(precision_t);
        void* restored=calloc(1,a.total_bytes); memcpy(restored,a.mem,copied);
        int roundtrip_bad=0;
        for(int i=0;i<a.num_regs;++i) {
            auto& e=a.regs[i];long off=(char*)*e.data_ptr-(char*)a.mem;
            roundtrip_bad+=memcmp((char*)restored+off,*e.data_ptr,numel(e.shape)*e.elem_size)!=0;
        }
        int opt_bad=optimizer_mismatches(a);
        if(0 || !continuous)assert(roundtrip_bad==0 && opt_bad==0 && copied==a.total_bytes);
        else assert(roundtrip_bad>0 && opt_bad>0 && copied<a.total_bytes);
        if(!first)printf(",");first=false;
        printf("{\"links\":%d,\"continuous\":%s,\"element_bytes\":%d,\"logical_elements\":%ld,\"physical_bytes\":%ld,\"flat_bytes\":%ld,\"roundtrip_bad_tensors\":%d,\"optimizer_bad_slices\":%d}",n,continuous?"true":"false",PRECISION_BYTES,a.total_elems,a.total_bytes,copied,roundtrip_bad,opt_bad);
        free(restored);free(a.mem);free(a.regs);
    }
    printf("]\n");return 0;
}
