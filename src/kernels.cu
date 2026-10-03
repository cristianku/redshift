// Selected quantized and recurrent kernels adapted from DS4; see THIRD_PARTY.md.
#include "kernels.h"
#include <cuda_fp16.h>
#include <stdint.h>
#include <math.h>

namespace qv {
__device__ __forceinline__ float sum(float x) {
    for (int d=16; d; d>>=1) x += __shfl_xor_sync(0xffffffff,x,d);
    return x;
}
__device__ float sigmoid(float x) { return 1.f/(1.f+expf(-x)); }
__device__ float silu(float x) { return x*sigmoid(x); }
__device__ float softplus(float x) { return x>20 ? x : log1pf(expf(x)); }
__device__ float block_sum(float x, float *scratch) {
    x=sum(x);
    if (!(threadIdx.x&31)) scratch[threadIdx.x/32]=x;
    __syncthreads();
    // Every warp needs the block total, not just warp zero.
    const unsigned lane=threadIdx.x&31;
    float y=lane<blockDim.x/32 ? scratch[lane] : 0;
    y=sum(y);
    __syncthreads();
    return y;
}
template<unsigned ROWS>
__global__ void q4(float *out, const uint8_t *w, const float *x,
        unsigned T, unsigned K, unsigned M) {
    const unsigned row=blockIdx.x*4+threadIdx.x/32, lane=threadIdx.x&31;
    if(row>=M)return;
    const unsigned pair=lane/8, j=(lane%8)*4, g=pair*2;
    const uint8_t *wr=w+(uint64_t)row*(K/256*144);
    float acc[ROWS]={};
    for(unsigned block=0;block<K/256;block++) {
        const uint8_t *b=wr+block*144, *sc=b+4;
        const float d=__half2float(*(const __half *)b), mn=__half2float(*(const __half *)(b+2));
        const unsigned s0=g<4 ? sc[g]&63 : (sc[g+4]&15)|((sc[g-4]>>6)<<4);
        const unsigned s1=g<4 ? sc[g+1]&63 : (sc[g+5]&15)|((sc[g-3]>>6)<<4);
        const unsigned m0=g<4 ? sc[g+4]&63 : (sc[g+4]>>4)|((sc[g]>>6)<<4);
        const unsigned m1=g<4 ? sc[g+5]&63 : (sc[g+5]>>4)|((sc[g+1]>>6)<<4);
        const unsigned bits=*(const unsigned *)(b+16+pair*32+j);
        const float lo=d*s0, hi=d*s1, ml=mn*m0, mh=mn*m1;
        const float4 wl=make_float4(lo*(bits&15)-ml,lo*((bits>>8)&15)-ml,
            lo*((bits>>16)&15)-ml,lo*((bits>>24)&15)-ml);
        const float4 wh=make_float4(hi*((bits>>4)&15)-mh,hi*((bits>>12)&15)-mh,
            hi*((bits>>20)&15)-mh,hi*(bits>>28)-mh);
        #pragma unroll
        for(unsigned t=0;t<ROWS;t++)if(t<T) {
            const float *v=x+(uint64_t)t*K+block*256+g*32+j;
            const float4 vl=*(const float4 *)v, vh=*(const float4 *)(v+32);
            acc[t]+=wl.x*vl.x; acc[t]+=wl.y*vl.y; acc[t]+=wl.z*vl.z; acc[t]+=wl.w*vl.w;
            acc[t]+=wh.x*vh.x; acc[t]+=wh.y*vh.y; acc[t]+=wh.z*vh.z; acc[t]+=wh.w*vh.w;
        }
    }
    #pragma unroll
    for(unsigned t=0;t<ROWS;t++)if(t<T) {
        const float v=sum(acc[t]); if(!lane)out[(uint64_t)t*M+row]=v;
    }
}
template<unsigned ROWS>
__global__ void q6(float *out, const char *w, const float *x,
        unsigned T, unsigned K, unsigned M, uint64_t stride) {
    const unsigned row = blockIdx.x*4+threadIdx.x/32, lane = threadIdx.x&31;
    if (row >= M) return;
    const uint8_t *wr = (const uint8_t *)w+(uint64_t)row*stride;
    float acc[ROWS] = {};
    for (unsigned block = 0; block < K/256; block++) {
        const uint8_t *b = wr+(uint64_t)block*210;
        const float d = __half2float(*(const __half *)(b+208));
        #pragma unroll
        for (unsigned half = 0; half < 2; half++) {
            const unsigned l0 = b[half*64+lane], l1 = b[half*64+32+lane];
            const unsigned hi = b[128+half*32+lane];
            const int8_t *sc = (const int8_t *)(b+192+half*8+lane/16);
            const float w0 = (d*sc[0])*(float)((int)((l0&15)|((hi&3)<<4))-32);
            const float w1 = (d*sc[2])*(float)((int)((l1&15)|(((hi>>2)&3)<<4))-32);
            const float w2 = (d*sc[4])*(float)((int)((l0>>4)|(((hi>>4)&3)<<4))-32);
            const float w3 = (d*sc[6])*(float)((int)((l1>>4)|(((hi>>6)&3)<<4))-32);
            #pragma unroll
            for (unsigned t = 0; t < ROWS; t++) if (t < T) {
                const float *v = x+(uint64_t)t*K+block*256+half*128+lane;
                acc[t] += w0*v[0]; acc[t] += w1*v[32];
                acc[t] += w2*v[64]; acc[t] += w3*v[96];
            }
        }
    }
    #pragma unroll
    for (unsigned t = 0; t < ROWS; t++) if (t < T) {
        const float v = sum(acc[t]);
        if (!lane) out[(uint64_t)t*M+row] = v;
    }
}

__global__ void conv(float *x, float *history, const float *w, unsigned T,
                     unsigned C, unsigned K, bool activate,
                     float *snap, unsigned snap_t, float *snap2, unsigned snap2_t) {
    const unsigned c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float win[3], taps[4];
    for (unsigned i = 0; i < K - 1; i++) win[i] = history[(uint64_t)i * C + c];
    for (unsigned i = 0; i < K; i++) taps[i] = w[c * K + i];
    for (unsigned t = 0; t < T; t++) {
        const uint64_t pos = (uint64_t)t * C + c;
        const float raw = x[pos];
        float v = taps[K - 1] * raw;
        for (unsigned i = 0; i < K - 1; i++) v += taps[i] * win[i];
        for (unsigned i = 0; i + 2 < K; i++) win[i] = win[i + 1];
        win[K - 2] = raw;
        x[pos] = activate ? silu(v) : v;
        if (snap && t == snap_t) for (unsigned i = 0; i < K - 1; i++) snap[(uint64_t)i * C + c] = win[i];
        if (snap2 && t == snap2_t) for (unsigned i = 0; i < K - 1; i++) snap2[(uint64_t)i * C + c] = win[i];
    }
    for (unsigned i = 0; i < K - 1; i++) history[(uint64_t)i * C + c] = win[i];
}

__global__ void gdn_prep(float *qkv, float *a, float *b, const float *A, const float *bias,
                         unsigned Hk, unsigned Hv, unsigned D) {
    const unsigned h = blockIdx.x, t = blockIdx.y, lane = threadIdx.x;
    const unsigned C = (2 * Hk + Hv) * D, npt = D / 32;
    float *q = qkv + (uint64_t)t * C + h * D + lane * npt, *k = q + Hk * D;
    float qs = 0, ks = 0;
    for (unsigned i = 0; i < npt; i++) { qs += q[i] * q[i]; ks += k[i] * k[i]; }
    qs = rsqrtf(sum(qs) + 1e-6f) * rsqrtf((float)D);
    ks = rsqrtf(sum(ks) + 1e-6f);
    for (unsigned i = 0; i < npt; i++) { q[i] *= qs; k[i] *= ks; }
    if (!h) for (unsigned j = lane; j < Hv; j += 32) {
        const uint64_t p = (uint64_t)t * Hv + j;
        a[p] = expf(A[j] * softplus(a[p] + bias[j]));
        b[p] = sigmoid(b[p]);
    }
}

/* One warp owns a state row across the whole chunk. In particular, MTP
 * snapshots contain the state after the requested token, not the final row. */
template<unsigned ROWS, unsigned D>
__global__ void gdn_scan(float *out, float *state, const float *qkv,
                         const float *a, const float *b, unsigned T, unsigned Hk,
                         unsigned Hv, float *snap, unsigned st,
                         float *snap2, unsigned st2) {
    const unsigned dv = (blockIdx.x * 4 + threadIdx.x / 32)*ROWS, h = blockIdx.y;
    if (dv >= D) return;
    const unsigned npt = D / 32, k0 = (threadIdx.x & 31) * npt, kh = h % Hk;
    const unsigned C = (2 * Hk + Hv) * D;
    const uint64_t idx = ((uint64_t)h * D + dv) * D + k0;
    float s[ROWS][4];
    #pragma unroll
    for (unsigned r = 0; r < ROWS; r++)
        #pragma unroll
        for (unsigned i = 0; i < npt; i++) s[r][i] = state[idx+(uint64_t)r*D+i];
    for (unsigned t = 0; t < T; t++) {
        const float *q = qkv + (uint64_t)t * C + kh * D + k0, *k = q + Hk * D;
        const float decay = a[(uint64_t)t * Hv + h], beta = b[(uint64_t)t * Hv + h];
        #pragma unroll
        for (unsigned r = 0; r < ROWS; r++) {
            const float v = qkv[(uint64_t)t*C+2*Hk*D+h*D+dv+r];
            float u = 0;
            #pragma unroll
            for (unsigned i = 0; i < npt; i++) { s[r][i] *= decay; u += s[r][i]*k[i]; }
            const float delta = (v-sum(u))*beta;
            float o = 0;
            #pragma unroll
            for (unsigned i = 0; i < npt; i++) { s[r][i] += k[i]*delta; o += s[r][i]*q[i]; }
            o = sum(o);
            if (!(threadIdx.x&31)) out[((uint64_t)t*Hv+h)*D+dv+r] = o;
            if (snap && t == st) for (unsigned i = 0; i < npt; i++) snap[idx+(uint64_t)r*D+i] = s[r][i];
            if (snap2 && t == st2) for (unsigned i = 0; i < npt; i++) snap2[idx+(uint64_t)r*D+i] = s[r][i];
        }
    }
    #pragma unroll
    for (unsigned r = 0; r < ROWS; r++)
        #pragma unroll
        for (unsigned i = 0; i < npt; i++) state[idx+(uint64_t)r*D+i] = s[r][i];
}

template<int ROWS>
__global__ void q8(float *out,const unsigned char *w,const float *x,int T,int K,int M) {
    int row=blockIdx.x*4+threadIdx.x/32,lane=threadIdx.x%32;
    if(row>=M)return;
    const unsigned char *wr=w+(uint64_t)row*(K/32*34);
    float acc[ROWS]={};
    for(int b=0;b<K/32;b++) {
        const unsigned char *p=wr+b*34;
        float v=__half2float(*(const __half*)p)*(float)((const int8_t*)(p+2))[lane];
        #pragma unroll
        for(int t=0;t<ROWS;t++)if(t<T)acc[t]+=v*x[(uint64_t)t*K+b*32+lane];
    }
    for(int t=0;t<ROWS;t++)if(t<T){float v=sum(acc[t]);if(!lane)out[(uint64_t)t*M+row]=v;}
}
__device__ float q4_value(const uint8_t *row,int i) {
    const uint8_t *p=row+(i/256)*144,*s=p+4;
    int j=i%256,g=j/32;
    int scale=g<4 ? s[g]&63 : (s[g+4]&15)|((s[g-4]>>6)<<4);
    int zero=g<4 ? s[g+4]&63 : (s[g+4]>>4)|((s[g]>>6)<<4);
    int q=(p[16+(j/64)*32+j%32]>>(4*(g%2)))&15;
    return __half2float(*(const __half*)p)*scale*q-__half2float(*(const __half*)(p+2))*zero;
}
__global__ void embed(float *out,const uint8_t *w,const int *ids) {
    int i=blockIdx.x*256+threadIdx.x,t=blockIdx.y;
    if(i<5120)out[t*5120+i]=q4_value(w+(uint64_t)ids[t]*(5120/256*144),i);
}
__global__ void norm(float *out,const float *x,const float *w,int N,float eps) {
    int t=blockIdx.x; __shared__ float red[8]; float s=0;
    for(int i=threadIdx.x;i<N;i+=256){float v=x[(uint64_t)t*N+i];s+=v*v;}
    float inv=rsqrtf(block_sum(s,red)/N+eps);
    for(int i=threadIdx.x;i<N;i+=256)out[(uint64_t)t*N+i]=x[(uint64_t)t*N+i]*inv*w[i];
}
__global__ void add(float *x,const float *y,int N) {
    int i=blockIdx.x*256+threadIdx.x;if(i<N)x[i]+=y[i];
}
__global__ void swiglu(float *out,const float *gate,const float *up,int N) {
    int i=blockIdx.x*256+threadIdx.x;if(i<N)out[i]=silu(gate[i])*up[i];
}
__global__ void delta_out(float *out,const float *z,const float *w,int H,int D,float eps) {
    int h=blockIdx.x,t=blockIdx.y; uint64_t p=((uint64_t)t*H+h)*D;
    float s=0;for(int i=threadIdx.x;i<D;i+=32)s+=out[p+i]*out[p+i];
    float inv=rsqrtf(sum(s)/D+eps);
    for(int i=threadIdx.x;i<D;i+=32)out[p+i]*=inv*w[i]*silu(z[p+i]);
}
// Text-only rotary positions: the three MRoPE coordinates coincide.
__global__ void attn_prepare(float *q,float *gate,__half *kc,__half *vc,
        const float *qg,const float *k,const float *v,const float *qn,const float *kn,
        int base,float eps) {
    int h=blockIdx.x,t=blockIdx.y,i=threadIdx.x; bool isq=h<24;
    int kh=h-24; const float *src=isq ? qg+(t*24+h)*512 : k+(t*4+kh)*256;
    const float *gamma=isq ? qn : kn;
    __shared__ float row[256],red[8];
    float inv=rsqrtf(block_sum(src[i]*src[i],red)/256+eps);
    row[i]=src[i]*inv*gamma[i]; __syncthreads();
    if(i<32) {
        float theta=(base+t)*powf(10000000.f,-(float)i/32);
        float c=cosf(theta),s=sinf(theta),a=row[i],b=row[i+32];
        row[i]=a*c-b*s;row[i+32]=a*s+b*c;
    }
    __syncthreads();
    if(isq){q[(t*24+h)*256+i]=row[i];gate[(t*24+h)*256+i]=src[256+i];}
    else {kc[((uint64_t)(base+t)*4+kh)*256+i]=__float2half_rn(row[i]);
          vc[((uint64_t)(base+t)*4+kh)*256+i]=__float2half_rn(v[(t*4+kh)*256+i]);}
}
__global__ void attention(float *out,const float *q,const float *gate,
        const __half *kc,const __half *vc,int base) {
    int h=blockIdx.x,t=blockIdx.y,tid=threadIdx.x,lane=tid%32,warp=tid/32;
    int n=base+t+1,kh=h/6;
    __shared__ float score[2048],red[8];
    for(int p=warp;p<n;p+=8) {
        float dot=0;for(int i=lane;i<256;i+=32)
            dot+=q[(t*24+h)*256+i]*__half2float(kc[((uint64_t)p*4+kh)*256+i]);
        dot=sum(dot);if(!lane)score[p]=dot*.0625f;
    }
    __syncthreads();
    float mx=-INFINITY;for(int p=tid;p<n;p+=256)mx=fmaxf(mx,score[p]);
    for(int d=16;d;d>>=1)mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,d));
    if(!lane)red[warp]=mx;__syncthreads();
    mx=-INFINITY;for(int i=0;i<8;i++)mx=fmaxf(mx,red[i]);
    // All warps must finish reading the maxima before block_sum reuses red.
    __syncthreads();
    float total=0;for(int p=tid;p<n;p+=256){float a=expf(score[p]-mx);score[p]=a;total+=a;}
    total=block_sum(total,red);__syncthreads();
    float value=0;for(int p=0;p<n;p++)value+=score[p]*__half2float(vc[((uint64_t)p*4+kh)*256+tid]);
    out[(t*24+h)*256+tid]=value/total*sigmoid(gate[(t*24+h)*256+tid]);
}
__global__ void argmax(int *out,const float *x) {
    const int V=248320; int t=blockIdx.x,tid=threadIdx.x;
    __shared__ float val[256];__shared__ int idx[256];
    float best=-INFINITY;int id=0;
    for(int i=tid;i<V;i+=256)if(x[(uint64_t)t*V+i]>best){best=x[(uint64_t)t*V+i];id=i;}
    val[tid]=best;idx[tid]=id;__syncthreads();
    for(int d=128;d;d>>=1){if(tid<d && (val[tid+d]>val[tid] || (val[tid+d]==val[tid] && idx[tid+d]<idx[tid]))){val[tid]=val[tid+d];idx[tid]=idx[tid+d];}__syncthreads();}
    if(!tid)out[t]=idx[0];
}
} // namespace qv

void k_mm(float *o,const void *w,const float *x,int type,int T,int K,int M) {
#define DISPATCH(N) do { if(type==12)qv::q4<N><<<(M+3)/4,128>>>(o,(const uint8_t*)w,x,T,K,M); \
    else if(type==14)qv::q6<N><<<(M+3)/4,128>>>(o,(const char*)w,x,T,K,M,(uint64_t)(K/256)*210); \
    else qv::q8<N><<<(M+3)/4,128>>>(o,(const unsigned char*)w,x,T,K,M); } while(0)
    if(T==1){DISPATCH(1);}else if(T==2){DISPATCH(2);}else if(T<=4){DISPATCH(4);}else{DISPATCH(8);}
#undef DISPATCH
}
void k_norm(float *o,const float *x,const float *w,int T,int N,float eps){qv::norm<<<T,256>>>(o,x,w,N,eps);}
void k_embed(float *o,const void *w,const int *ids,int T){qv::embed<<<dim3(20,T),256>>>(o,(const uint8_t*)w,ids);}
void k_add(float *x,const float *y,int N){qv::add<<<(N+255)/256,256>>>(x,y,N);}
void k_swiglu(float *o,const float *g,const float *u,int N){qv::swiglu<<<(N+255)/256,256>>>(o,g,u,N);}
void k_delta(float *out,float *state,float *history,float *qkv,const float *z,float *a,float *b,
        const float *cv,const float *A,const float *bias,const float *norm,int T,int Hk,int Hv,int D,float eps) {
    int C=(2*Hk+Hv)*D;
    qv::conv<<<(C+255)/256,256>>>(qkv,history,cv,T,C,4,true,nullptr,0,nullptr,0);
    qv::gdn_prep<<<dim3(Hk,T),32>>>(qkv,a,b,A,bias,Hk,Hv,D);
    if(D==32)qv::gdn_scan<4,32><<<dim3(2,Hv),128>>>(out,state,qkv,a,b,T,Hk,Hv,nullptr,0,nullptr,0);
    else qv::gdn_scan<4,128><<<dim3(8,Hv),128>>>(out,state,qkv,a,b,T,Hk,Hv,nullptr,0,nullptr,0);
    qv::delta_out<<<dim3(Hv,T),32>>>(out,z,norm,Hv,D,eps);
}
void k_attention(float *o,void *kc,void *vc,float *q,float *gate,
        const float *qg,const float *k,const float *v,const float *qn,const float *kn,int T,int pos,float eps) {
    qv::attn_prepare<<<dim3(28,T),256>>>(q,gate,(__half*)kc,(__half*)vc,qg,k,v,qn,kn,pos,eps);
    qv::attention<<<dim3(24,T),256>>>(o,q,gate,(const __half*)kc,(const __half*)vc,pos);
}
void k_argmax(int *out,const float *x,int T){qv::argmax<<<T,256>>>(out,x);}
