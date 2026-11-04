#include "search/rabitq/rabitq_query_rotation.cuh"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
void run(uint32_t n, uint32_t d, uint32_t p, bool zero) {
    std::mt19937 rng(20260927);
    std::uniform_real_distribution<float> random(-1,1);
    std::vector<float> q(size_t(n)*d), r(size_t(p)*p), y(size_t(n)*p);
    for(auto& v:q) v=zero?0:random(rng);
    for(auto& v:r) v=random(rng)/std::sqrt(float(d));
    float *dq=nullptr,*dr=nullptr,*dy=nullptr;
    check(cudaMalloc(&dq,q.size()*4));check(cudaMalloc(&dr,r.size()*4));check(cudaMalloc(&dy,y.size()*4));
    check(cudaMemcpy(dq,q.data(),q.size()*4,cudaMemcpyHostToDevice));
    check(cudaMemcpy(dr,r.data(),r.size()*4,cudaMemcpyHostToDevice));
    cudaStream_t stream;check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    go_fann::RabitqQueryRotation rotation(true);
    rotation.apply(dq,dr,dy,n,d,p,stream);
    check(cudaStreamSynchronize(stream));
    check(cudaMemcpy(y.data(),dy,y.size()*4,cudaMemcpyDeviceToHost));
    double maximum=0;
    for(size_t i=0;i<n;++i) for(size_t j=0;j<p;++j) {
        double expected=0;
        for(size_t k=0;k<d;++k) expected+=double(q[i*d+k])*r[k*p+j];
        const double error=std::abs(y[i*p+j]-expected);maximum=std::max(maximum,error);
        if(!std::isfinite(y[i*p+j]) || error>2e-5+2e-5*std::abs(expected))
            throw std::runtime_error("FP64 reference mismatch: row/transpose/padding or accuracy");
    }
    bool rejected=false;
    try {rotation.apply(dq,dr,dy,n,p+1,p,stream);}catch(const std::invalid_argument&){rejected=true;}
    if(!rejected) throw std::runtime_error("invalid shape accepted");
    // Use query rows themselves as anchors: diagonal zero distances exercise
    // cancellation/clamping; off-diagonal pairs catch transposes and strides.
    float *dist=nullptr,*qn=nullptr,*an=nullptr;
    check(cudaMalloc(&dist,size_t(n)*n*4));check(cudaMalloc(&qn,n*4));check(cudaMalloc(&an,n*4));
    rotation.centroid_distances(dq,dq,dist,qn,an,n,n,d,stream);
    check(cudaStreamSynchronize(stream));
    std::vector<float> actual(size_t(n)*n);check(cudaMemcpy(actual.data(),dist,actual.size()*4,cudaMemcpyDeviceToHost));
    for(size_t i=0;i<n;++i)for(size_t j=0;j<n;++j) {
        double expected=0;
        for(size_t k=0;k<d;++k){const double diff=double(q[i*d+k])-q[j*d+k];expected+=diff*diff;}
        if(!std::isfinite(actual[i*n+j]) || actual[i*n+j]<0 ||
           std::abs(actual[i*n+j]-expected)>1e-3+2e-5*expected)
            throw std::runtime_error("Centroid distance FP64 reference mismatch");
    }
    check(cudaFree(dist));check(cudaFree(qn));check(cudaFree(an));
    check(cudaStreamDestroy(stream));check(cudaFree(dy));check(cudaFree(dr));check(cudaFree(dq));
    std::cout<<"shape="<<n<<'x'<<d<<'x'<<p<<" zero="<<zero<<" max_abs="<<maximum<<'\n';
}
int main() {
    try {
        run(1,1,1,false);run(7,17,32,false);run(33,65,96,false);
        run(13,512,512,false);run(3,4096,4096,false);run(7,17,32,true);
        std::cout<<"PASS: FP64 reference, rectangular/padded shapes, zero queries, non-default stream, invalid shape\n";
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
