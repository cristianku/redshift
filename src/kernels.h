#pragma once
#include <cuda_runtime.h>
#include <stdint.h>
void k_mm(float*,const void*,const float*,int,int,int,int);
// Packed projections: upload conversion and caller-owned reusable workspace.
// Split-K partials are reduced in a fixed order, for all candidate counts.
size_t k_mm_workspace_bytes(int,int,int);
size_t k_mm_weight_bytes(int,int,int);
void k_mm_pack_weights(void*,const void*,int,int,int);
void k_mm_packed(float*,const void*,const float*,int,int,int,int,void*,bool quantize_input=true);
void k_norm(float*,const float*,const float*,int,int,float);
void k_embed(float*,const void*,const int*,int);
void k_add(float*,const float*,int);
void k_swiglu(float*,const float*,const float*,int);
// Optional transitions: prepared QKV, decay and beta, each with capacity 8 rows.
void k_delta(float*,float*,float*,float*,const float*,float*,float*,const float*,const float*,const float*,const float*,int,int,int,int,float,float *transitions=nullptr);
// Replay recurrence/history only. Raw and prepared inputs remain immutable.
void k_delta_commit(float*,float*,float*,const float*,const float*,int,int,int,int);
void k_attention(float*,void*,void*,float*,float*,const float*,const float*,const float*,const float*,const float*,int,int,float);
void k_argmax(int*,const float*,int);
