#include "search/pq/pq_gpu_predicate.cuh"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

using go_fann::gpu_predicate::View;
#define CUDA(call) do { auto e = (call); if(e != cudaSuccess) { \
    std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(e)); std::exit(1); } } while(0)

template<class T> T* upload(const std::vector<T>& values) {
    T* p = nullptr; CUDA(cudaMallocManaged(&p, values.size()*sizeof(T)));
    std::copy(values.begin(), values.end(), p); return p;
}

// The 64-bit list is the PQ kernel's actual idx_t representation. Original
// sample IDs intentionally differ from local graph rows and include invalids.
__global__ void check_mapping(View view, const uint32_t* mapping, unsigned* errors) {
    const unsigned lane = threadIdx.x % 32, route = threadIdx.x / 32;
    __shared__ uint64_t local_rows[3][97];
    __shared__ uint32_t samples[3][97], masks[3][4];
    for (unsigned i=lane; i<97; i+=32) {
        local_rows[route][i] = (i*37)%97;
        samples[route][i] = mapping[5+local_rows[route][i]];
    }
    __syncwarp();
    for (unsigned count : {0u, 1u, 31u, 32u, 33u, 64u, 97u}) {
        for (unsigned payload=0; payload<2; ++payload) {
            if (lane<4) masks[route][lane]=0xdeadbeef;
            __syncwarp();
            const bool parallel=go_fann::pq_gpu_predicate::prepare_candidate_pass_masks(
                view,route,local_rows[route],payload?samples[route]:nullptr,
                count,5,payload?nullptr:mapping,payload,masks[route],lane);
            if (parallel != bool(GO_FANN_GPU_PREDICATE_PARALLEL && route!=2)) atomicAdd(errors,1);
            if (lane==0) {
                for (unsigned i=0;i<count;++i) {
                    const unsigned sample=samples[route][i];
                    // Independent original-label oracle: route0 label7;
                    // route1 entailed but invalid sample still cannot pass.
                    const bool expected=sample<67 && (route==1 || sample%3==0);
                    const bool scalar=go_fann::gpu_predicate::matches(view,route,sample);
                    if (route!=2 && scalar!=expected) atomicAdd(errors,1);
                    if (parallel && bool((masks[route][i/32]>>(i%32))&1)!=expected) atomicAdd(errors,1);
                }
                if (parallel && count%32 && (masks[route][count/32]>>(count%32))) atomicAdd(errors,1);
                for (unsigned i=parallel?(count+31)/32:0;i<4;++i)
                    if (masks[route][i]!=0xdeadbeef) atomicAdd(errors,1);
            }
            __syncwarp();
        }
    }
}

int main() {
    std::vector<uint32_t> offsets{0},labels,mapping(102,UINT32_MAX);
    for(unsigned sample=0;sample<67;++sample) {
        if(sample%3==0) labels.push_back(7);
        if(sample%5==0) labels.push_back(70000); // retain full uint32 labels
        offsets.push_back(labels.size());
    }
    for(unsigned i=0;i<97;++i) mapping[5+i]=i%11==0?UINT32_MAX:(i*19)%68;
    View v;v.rows=67;
    v.csr_offsets=upload(offsets);v.csr_labels=upload(labels);
    v.route_offsets=upload<uint32_t>({0,1,2,2});
    v.route_handles=upload<uint32_t>({7});
    v.clauses=upload<go_fann::gpu_predicate::Clause>({{0,1,0,0},{0,0,0,0}});
    v.route_enabled=upload<uint8_t>({1,1,0});
    auto* map=upload(mapping);auto* errors=upload<unsigned>({0});
    check_mapping<<<1,96>>>(v,map,errors);CUDA(cudaGetLastError());CUDA(cudaDeviceSynchronize());
    const unsigned count=*errors;
    for(const void* p:std::vector<const void*>{v.csr_offsets,v.csr_labels,v.route_offsets,
        v.route_handles,v.clauses,v.route_enabled,map,errors}) CUDA(cudaFree(const_cast<void*>(p)));
    if(count) {std::fprintf(stderr,"%u PQ predicate mapping errors\n",count);return 1;}
    std::printf("PQ predicate CSR mapping passed (parallel=%d): 64-bit local rows, sample payload, tails, entailed, fallback, invalid IDs\n",GO_FANN_GPU_PREDICATE_PARALLEL);
}
