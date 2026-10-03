#ifndef QVELOX_RUNTIME_H
#define QVELOX_RUNTIME_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Synchronous kernel validation ABI. All pointers refer to host memory.
 * Returns 0 on success, -1 on failure. qv_error() describes the last failure
 * on the calling thread, until its next ABI call. Caller owns buffer lengths.
 * These entry points include allocation and transfers: do not time them as
 * model inference. Batch sizes are 1..8. */
const char *qv_error(void);
int qv_test_mm(float *out, const void *weights, const float *x,
               int kind, int batch, int width, int rows);
/* DP4A path: two-byte activation blocks, FP32 scale/sum and accumulation.
 * Compare with qv_test_mm for the original FP32-activation reference. */
int qv_test_mm_fast(float *out, const void *weights, const float *x,
               int kind, int batch, int width, int rows);
int qv_test_norm(float *out, const float *x, const float *weights,
                 int batch, int width, float epsilon);

/* D = 32 or 128; 1 <= key_heads <= value_heads <= 48;
 * value_heads must be divisible by key_heads. C = (2*key_heads+value_heads)*D.
 * out/gate: [batch,value_heads,D]; state (in/out): [value_heads,D,D];
 * history (in/out): [3,C]; qkv: [batch,C]; alpha/beta: [batch,value_heads];
 * conv: [C,4]; decay/bias: [value_heads]; norm: [D].
 * decay contains the already transformed, negative SSM A coefficients. */
int qv_test_delta(float *out, float *state, float *history,
                  const float *qkv, const float *gate,
                  const float *alpha, const float *beta,
                  const float *conv, const float *decay,
                  const float *bias, const float *norm,
                  int batch, int key_heads, int value_heads, int dim,
                  float epsilon);

/* Fixed model attention: 24 query heads, 4 KV heads, head dimension 256.
 * out: [batch,24,256]; q_gate: [batch,24,512]; key/value: [batch,4,256];
 * q_norm/k_norm: [256]. Caches (in/out) hold IEEE float16 bits as
 * [position+batch,4,256]. Only the prefix [0,position) is read from host.
 * 0 <= position and position+batch <= 32768. */
int qv_test_attention(float *out, void *key_cache, void *value_cache,
                      const float *q_gate, const float *key, const float *value,
                      const float *q_norm, const float *k_norm,
                      int batch, int position, float epsilon);

/* Model ABI. One handle is single-threaded. Upload each validated manifest
 * slot once, before evaluating. Quantized weights are repacked on upload;
 * projections use two-byte activations and FP32 accumulation.
 * context: 1..32768, evaluate: 1..8 token IDs, out: [batch,248320] floats.
 * A checkpoint stores all recurrent state and the valid KV prefix. Restore
 * can be repeated; reset invalidates it. No implicit speculative acceptance. */
int qv_create(void **handle, const char *path, int context, float epsilon);
int qv_upload(void *handle, int slot, uint64_t offset, uint64_t length);
int qv_evaluate(void *handle, const int *tokens, int batch, float *out);
/* Same state transition as evaluate; downloads only each row's greedy token.
 * next: [batch] integer IDs. Ties choose the lowest vocabulary ID. */
int qv_advance(void *handle, const int *tokens, int batch, int *next);
int qv_reset(void *handle);
int qv_checkpoint(void *handle);
int qv_restore(void *handle);
int qv_position(void *handle, int *position);
void qv_destroy(void *handle);

#ifdef __cplusplus
}
#endif
#endif
