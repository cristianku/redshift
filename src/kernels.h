#pragma once
#include <cuda_runtime.h>
#include <stdint.h>
void k_mm(float*,const void*,const float*,int,int,int,int);
void k_norm(float*,const float*,const float*,int,int,float);
void k_embed(float*,const void*,const int*,int);
void k_add(float*,const float*,int);
void k_swiglu(float*,const float*,const float*,int);
void k_delta(float*,float*,float*,float*,const float*,float*,float*,const float*,const float*,const float*,const float*,int,int,int,int,float);
void k_attention(float*,void*,void*,float*,float*,const float*,const float*,const float*,const float*,const float*,int,int,float);
void k_argmax(int*,const float*,int);
