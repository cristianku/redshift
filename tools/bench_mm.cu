// Standalone comparison: existing FP32 projections vs DP4A, including input packing.
#include "kernels.h"
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

static void check(cudaError_t status) {
    if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));
}
struct Device {
    void *ptr=nullptr;
    explicit Device(size_t bytes){check(cudaMalloc(&ptr,bytes));}
    ~Device(){cudaFree(ptr);}
    Device(const Device&)=delete;
    float *floats(){return (float*)ptr;}
};
struct Event {
    cudaEvent_t value;
    Event(){check(cudaEventCreate(&value));}
    ~Event(){cudaEventDestroy(value);}
};
static void put_half(uint8_t *p,float value) {
    __half h=__float2half(value);std::memcpy(p,&h,2);
}
static std::vector<uint8_t> weights(int kind,int K,int M) {
    const int width=kind==8?32:256,size=kind==8?34:kind==12?144:210;
    std::vector<uint8_t> data(size_t(K/width)*M*size);
    uint32_t seed=47;
    for(auto &b:data){seed=seed*1664525u+1013904223u;b=seed>>24;}
    for(size_t start=0;start<data.size();start+=size) {
        if(kind==8)put_half(data.data()+start,.0078125f);
        else if(kind==12){put_half(data.data()+start,.03125f);put_half(data.data()+start+2,.015625f);}
        else put_half(data.data()+start+208,.00390625f);
    }
    return data;
}
int main(int argc,char **argv) try {
    const int K=argc>1?std::atoi(argv[1]):5120,M=argc>2?std::atoi(argv[2]):17408;
    if(K<=0||K%256||M<=0)throw std::runtime_error("usage: bench-mm [K multiple of 256] [M]");
    std::printf("type,T,K,M,fp32_ms,dp4a_ms,max_abs,max_scaled\n");
    for(int kind:{8,12,14}) {
        const auto packed=weights(kind,K,M);
        std::vector<float> input(size_t(8)*K);
        for(size_t i=0;i<input.size();i++)input[i]=.13f*std::sin(float(i)*.07f);
        Device w(packed.size()),x(input.size()*4),old(size_t(8)*M*4),fast(size_t(8)*M*4);
        Device workspace(k_mm_workspace_bytes(8,K,M)),converted(k_mm_weight_bytes(kind,K,M));
        check(cudaMemcpy(w.ptr,packed.data(),packed.size(),cudaMemcpyHostToDevice));
        check(cudaMemcpy(x.ptr,input.data(),input.size()*4,cudaMemcpyHostToDevice));
        k_mm_pack_weights(converted.ptr,w.ptr,kind,K,M);
        check(cudaDeviceSynchronize());
        Event start,end;
        for(int T:{1,2,4,8}) {
            auto run=[&](bool optimized) {
                if(optimized)k_mm_packed(fast.floats(),converted.ptr,x.floats(),kind,T,K,M,workspace.ptr);
                else k_mm(old.floats(),w.ptr,x.floats(),kind,T,K,M);
            };
            for(int warm=0;warm<5;warm++){run(false);run(true);}
            check(cudaDeviceSynchronize());
            std::vector<float> timings[2];
            for(int repeat=0;repeat<5;repeat++)for(int order=0;order<2;order++) {
                const int optimized=order^(repeat&1);
                check(cudaEventRecord(start.value));
                for(int i=0;i<20;i++)run(optimized);
                check(cudaEventRecord(end.value));
                check(cudaEventSynchronize(end.value));check(cudaGetLastError());
                float elapsed;check(cudaEventElapsedTime(&elapsed,start.value,end.value));
                timings[optimized].push_back(elapsed/20);
            }
            for(auto &times:timings)std::sort(times.begin(),times.end());
            std::vector<float> a(size_t(T)*M),b(a.size());
            check(cudaMemcpy(a.data(),old.ptr,a.size()*4,cudaMemcpyDeviceToHost));
            check(cudaMemcpy(b.data(),fast.ptr,b.size()*4,cudaMemcpyDeviceToHost));
            double absolute=0,scaled=0;
            for(size_t i=0;i<a.size();i++) {
                if(!std::isfinite(a[i])||!std::isfinite(b[i]))throw std::runtime_error("non-finite output");
                double error=std::abs(double(a[i])-b[i]);
                absolute=std::max(absolute,error);scaled=std::max(scaled,error/(1+std::abs(a[i])));
            }
            std::printf("%d,%d,%d,%d,%.6f,%.6f,%.8g,%.8g\n",kind,T,K,M,
                        timings[0][2],timings[1][2],absolute,scaled);
            std::fflush(stdout);
        }
    }
    return 0;
} catch(const std::exception &e) {std::fprintf(stderr,"%s\n",e.what());return 1;}
