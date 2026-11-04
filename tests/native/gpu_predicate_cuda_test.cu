#include "predicates/gpu_predicate_index.cuh"
#include <cstdio>
#include "smmh2.h"
#include <cuda_runtime.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>
using namespace go_fann::gpu_predicate;
#define CUDA(call) do { auto error=(call); if(error!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(error));std::exit(1);} }while(0)
__global__ void membership_fixture(View v, unsigned* errors) {
    for(uint32_t id=threadIdx.x;id<=v.rows;id+=blockDim.x) {
        const bool expected=id==0||id==65535||id==65536||id==65537;
        if(matches(v,0,id)!=expected) atomicAdd(errors,1);
        if(matches(v,1,id)!=(id==65537)) atomicAdd(errors,1);
    }
}
__global__ void navigation_fixture(View v, unsigned* errors) {
    if(threadIdx.x) return;
    BiasedPair q[34]; int size=1;
    // Whole-entry swaps/pop-max must retain raw scores, even on negative scores.
    for(uint32_t i=0;i<20;++i) {
        bool saturated=false; float raw=float(i)-10;
        BiasedPair e{priority(raw,5,i%2==0,saturated),i,raw};
        smmh2::insert(q,size,e);
        if(size>=10) smmh2::pop_max(q,size);
        for(int j=1;j<size;++j) if(q[j].base!=float(q[j].second)-10) ++*errors;
    }
    // Required failed bridge and nonidentity local -> sample mapping.
    const uint32_t sample[3]={2,65536,65537};
    size=1; BiasedPair seed{0,0,0}; smmh2::insert(q,size,seed);
    unsigned visited=0, passed=0;
    while(size>1) {
        int best=1; for(int j=2;j<size;++j) if(q[j].first<q[best].first) best=j;
        const auto now=q[best]; q[best]=q[--size];
        if(visited&(1u<<now.second)) continue; visited|=1u<<now.second;
        const bool pass=matches(v,1,sample[now.second]);
        if(pass) passed|=1u<<now.second;
        // A failed predicate never suppresses adjacency.
        if(now.second<2) {
            const auto next=now.second+1;
            bool saturated=false;
            const float base=-float(next);
            BiasedPair entry{priority(base,10,matches(v,1,sample[next]),saturated),next,base};
            smmh2::insert(q,size,entry);
        }
    }
    if(visited!=7||passed!=4) ++*errors;
}
struct FixturePending { __device__ bool operator()(uint32_t id) const { return (id & 0x80000000u)==0; } };
__global__ void residual_and_stop_fixture(unsigned* errors) {
    if (threadIdx.x) return;
    const uint32_t directory[]={0,1}, handles[]={0,1};
    const Descriptor descriptors[]={{0,2,0},{2,2,0}};
    const uint16_t payload[]={0,2,1,2};
    const uint32_t offsets[]={0,1,3,4,4,4};
    const Clause clauses[]={{0,0,0,0},{0,1,0,1},{1,2,0,0},{0,0,1,2}};
    const RangeAtom ranges[]={{0,1,10,0},{1,3,2,4}};
    const float values[]={10,2,11,4,11,3};
    const uint8_t covered[]={1,1,1,1,0};
    View v{directory,descriptors,payload,offsets,handles,nullptr,3,1,0};
    v.clauses=clauses; v.ranges=ranges; v.base_ranges=values;
    v.route_enabled=covered; v.range_stride=2;
    const uint32_t csr_offsets[]={0,1,2,4}, csr_labels[]={0,1,0,1};
    for (int storage=0;storage<2;++storage) {
        if (storage) {
            v.directory=nullptr; v.csr_offsets=csr_offsets; v.csr_labels=csr_labels;
        }
        for (uint32_t sample=0;sample<3;++sample) {
            if (!matches(v,0,sample)) ++*errors; // Entailed: no index probes.
            if (matches(v,1,sample)!=(sample!=0)) ++*errors; // DNF, strict boundary.
            if (matches(v,2,sample)!=(sample!=1)) ++*errors; // Half-open range.
            if (matches(v,3,sample)) ++*errors; // Empty disjunction is false.
        }
    }
    v.csr_offsets=nullptr; v.csr_labels=nullptr;
    if (enabled(v,4) || matches(v,0,3)) ++*errors;
    v.directory=nullptr;
    if (!matches(v,0,0) || !matches(v,2,2)) ++*errors; // No bitmap allocation.

    constexpr uint32_t used=0x80000000u;
    BiasedPair pool[5]{};
    pool[1]={50,used|1,1}; pool[2]={100,2,-8};
    pool[3]={0,used|3,-100}; pool[4]={2,4,2};
    BiasedPair current{70,used|5,-5}, next{80,used|6,-2};
    const FixturePending pending;
    if (minimum_pending_base(pool,5,current,next,true,true,pending)!=-8) ++*errors;
    pool[2].second|=used;
    // A low-base processed pool entry must be excluded, while a reserved
    // current/next must survive pickup's used mark and both affect stopping.
    if (minimum_pending_base(pool,5,current,next,true,true,pending)!=-5) ++*errors;
    if (minimum_pending_base(pool,5,current,next,false,true,pending)!=-2) ++*errors;
    if (minimum_pending_base(pool,5,current,next,false,false,pending)!=2) ++*errors;
    struct Compact {float first;uint32_t second;};
    const Compact compact[]={{0,0},{3,7},{-20,used|8}};
    const Compact reserved{1,used|9};
    if (minimum_pending_base(compact,3,reserved,reserved,true,true,pending)!=1) ++*errors;
    // Bias orders by priority, but the result/stop score is still base.
    if (!(pool[1].first < next.first && base_score(pool[1]) > base_score(next))) ++*errors;
}
__global__ void csr_original_id_fixture(unsigned* errors) {
    if (threadIdx.x) return;
    const uint32_t offsets[]={0,3,3,4}, labels[]={0,65536,7330000,65535};
    View v; v.rows=3; v.csr_offsets=offsets; v.csr_labels=labels;
    if (!enabled(v,0) || !contains(v,65536,0) || !contains(v,7330000,0) ||
        contains(v,65535,0) || contains(v,0,1) || !contains(v,65535,2) ||
        contains(v,0,3) || contains(v,UINT32_MAX,0)) ++*errors;
}
int main() {
    uint32_t *directory,*offsets,*handles; uint16_t* payload; Descriptor* descriptors; unsigned* errors;
    CUDA(cudaMallocManaged(&directory,4*4));CUDA(cudaMallocManaged(&offsets,3*4));CUDA(cudaMallocManaged(&handles,3*4));
    CUDA(cudaMallocManaged(&payload,192*2));CUDA(cudaMallocManaged(&descriptors,3*sizeof(Descriptor)));CUDA(cudaMallocManaged(&errors,4));
    directory[0]=0;directory[1]=1;directory[2]=UINT32_MAX;directory[3]=2;
    offsets[0]=0;offsets[1]=1;offsets[2]=3;handles[0]=0;handles[1]=0;handles[2]=1;
    descriptors[0]={0,2,0};descriptors[1]={64,2,0};descriptors[2]={128,1,0};
    payload[0]=0;payload[1]=65535;payload[64]=0;payload[65]=1;payload[128]=1;*errors=0;
    View v{directory,descriptors,payload,offsets,handles,nullptr,65538,2,0};
    membership_fixture<<<1,128>>>(v,errors);navigation_fixture<<<1,32>>>(v,errors);
    residual_and_stop_fixture<<<1,32>>>(errors);
    csr_original_id_fixture<<<1,32>>>(errors);
    CUDA(cudaGetLastError());CUDA(cudaDeviceSynchronize());assert(*errors==0);
    CUDA(cudaFree(directory));CUDA(cudaFree(offsets));CUDA(cudaFree(handles));CUDA(cudaFree(payload));CUDA(cudaFree(descriptors));CUDA(cudaFree(errors));
    std::puts("GPU predicate CUDA membership, whole-entry heap and failed-bridge fixture passed");
}
