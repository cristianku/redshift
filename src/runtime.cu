#include "runtime.h"
#include "kernels.h"

#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>

namespace {
thread_local char last_error[512] = {};

void require(bool condition, const char *message) {
    if (!condition) throw std::invalid_argument(message);
}

void check(cudaError_t result) {
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}

size_t bytes(size_t count, size_t element) {
    require(count <= std::numeric_limits<size_t>::max() / element,
            "buffer size overflow");
    return count * element;
}

// Own every allocation even when a later allocation, copy or kernel fails.
class Buffer {
    void *data_ = nullptr;
    size_t size_;
public:
    explicit Buffer(size_t size, const void *source = nullptr) : size_(size) {
        check(cudaMalloc(&data_, size_));
        try {
            if (source) check(cudaMemcpy(data_, source, size_, cudaMemcpyHostToDevice));
        } catch (...) {
            cudaFree(data_);
            throw;
        }
    }
    ~Buffer() { cudaFree(data_); }
    Buffer(const Buffer &) = delete;
    Buffer &operator=(const Buffer &) = delete;
    void *data() { return data_; }
    float *floats() { return static_cast<float *>(data_); }
    void download(void *target) {
        check(cudaMemcpy(target, data_, size_, cudaMemcpyDeviceToHost));
    }
};

template<class Function>
int call(Function function) noexcept {
    last_error[0] = '\0';
    try {
        function();
        return 0;
    } catch (const std::exception &error) {
        std::snprintf(last_error, sizeof(last_error), "%s", error.what());
    } catch (...) {
        std::snprintf(last_error, sizeof(last_error), "unknown native error");
    }
    return -1;
}

void batch_size(int batch) {
    require(batch >= 1 && batch <= 8, "batch must be between 1 and 8");
}

void epsilon_value(float epsilon) {
    require(std::isfinite(epsilon) && epsilon > 0, "epsilon must be finite and positive");
}

void finish() {
    check(cudaGetLastError());
    check(cudaDeviceSynchronize());
}
} // namespace

extern "C" const char *qv_error(void) { return last_error; }

static int test_projection(float *out, const void *weights, const float *x,
                           int kind, int batch, int width, int rows, bool optimized) {
    return call([&] {
        batch_size(batch);
        require(out && weights && x, "null projection buffer");
        require(kind == 8 || kind == 12 || kind == 14, "unsupported quantization type");
        const int block = kind == 8 ? 32 : 256;
        const int block_bytes = kind == 8 ? 34 : (kind == 12 ? 144 : 210);
        require(width > 0 && width % block == 0 && rows > 0 && rows <= 248320,
                "invalid projection dimensions");
        Buffer w(bytes(bytes(size_t(width / block), block_bytes), rows), weights);
        Buffer input(bytes(bytes(size_t(batch), width), sizeof(float)), x);
        Buffer output(bytes(bytes(size_t(batch), rows), sizeof(float)));
        if(optimized) {
            Buffer workspace(k_mm_workspace_bytes(batch,width,rows));
            Buffer packed(k_mm_weight_bytes(kind,width,rows));
            k_mm_pack_weights(packed.data(),w.data(),kind,width,rows);
            check(cudaGetLastError());
            k_mm_packed(output.floats(),packed.data(),input.floats(),kind,batch,width,rows,workspace.data());
            finish();
        } else k_mm(output.floats(),w.data(),input.floats(),kind,batch,width,rows);
        finish();
        output.download(out);
    });
}

extern "C" int qv_test_mm(float *out,const void *weights,const float *x,
                          int kind,int batch,int width,int rows) {
    return test_projection(out,weights,x,kind,batch,width,rows,false);
}
extern "C" int qv_test_mm_fast(float *out,const void *weights,const float *x,
                               int kind,int batch,int width,int rows) {
    return test_projection(out,weights,x,kind,batch,width,rows,true);
}

extern "C" int qv_test_norm(float *out, const float *x, const float *weights,
                             int batch, int width, float epsilon) {
    return call([&] {
        batch_size(batch);
        epsilon_value(epsilon);
        require(out && x && weights, "null normalization buffer");
        require(width > 0 && width <= 17408, "invalid normalization width");
        const size_t size = bytes(bytes(size_t(batch), width), sizeof(float));
        Buffer input(size, x), w(bytes(size_t(width), sizeof(float)), weights), output(size);
        k_norm(output.floats(), input.floats(), w.floats(), batch, width, epsilon);
        finish();
        output.download(out);
    });
}

extern "C" int qv_test_delta(float *out, float *state, float *history,
                              const float *qkv, const float *gate,
                              const float *alpha, const float *beta,
                              const float *conv, const float *decay,
                              const float *bias, const float *norm,
                              int batch, int key_heads, int value_heads, int dim,
                              float epsilon) {
    return call([&] {
        batch_size(batch);
        epsilon_value(epsilon);
        require(out && state && history && qkv && gate && alpha && beta && conv &&
                decay && bias && norm, "null DeltaNet buffer");
        require((dim == 32 || dim == 128) && key_heads > 0 &&
                value_heads >= key_heads && value_heads <= 48 && value_heads % key_heads == 0,
                "unsupported DeltaNet dimensions");
        const size_t channels = size_t(2 * key_heads + value_heads) * dim;
        const size_t output_bytes = size_t(batch) * value_heads * dim * sizeof(float);
        const size_t head_bytes = size_t(batch) * value_heads * sizeof(float);
        Buffer s(size_t(value_heads) * dim * dim * sizeof(float), state);
        Buffer h(3 * channels * sizeof(float), history);
        Buffer q(size_t(batch) * channels * sizeof(float), qkv);
        Buffer z(output_bytes, gate), a(head_bytes, alpha), b(head_bytes, beta);
        Buffer c(4 * channels * sizeof(float), conv);
        Buffer d(size_t(value_heads) * sizeof(float), decay);
        Buffer bi(size_t(value_heads) * sizeof(float), bias);
        Buffer n(size_t(dim) * sizeof(float), norm), output(output_bytes);
        k_delta(output.floats(), s.floats(), h.floats(), q.floats(), z.floats(),
                a.floats(), b.floats(), c.floats(), d.floats(), bi.floats(), n.floats(),
                batch, key_heads, value_heads, dim, epsilon);
        finish();
        output.download(out);
        s.download(state);
        h.download(history);
    });
}

extern "C" int qv_test_attention(float *out, void *key_cache, void *value_cache,
                                  const float *q_gate, const float *key, const float *value,
                                  const float *q_norm, const float *k_norm,
                                  int batch, int position, float epsilon) {
    return call([&] {
        batch_size(batch);
        epsilon_value(epsilon);
        require(position >= 0 && position <= 2048 - batch, "attention context exceeds 2048");
        require(out && key_cache && value_cache && q_gate && key && value && q_norm && k_norm,
                "null attention buffer");
        const size_t output_bytes = size_t(batch) * 24 * 256 * sizeof(float);
        const size_t kv_bytes = size_t(batch) * 4 * 256 * sizeof(float);
        const size_t cache_bytes = size_t(position + batch) * 4 * 256 * sizeof(uint16_t);
        const size_t prefix_bytes = size_t(position) * 4 * 256 * sizeof(uint16_t);
        Buffer kc(cache_bytes), vc(cache_bytes);
        if (prefix_bytes) {
            check(cudaMemcpy(kc.data(), key_cache, prefix_bytes, cudaMemcpyHostToDevice));
            check(cudaMemcpy(vc.data(), value_cache, prefix_bytes, cudaMemcpyHostToDevice));
        }
        Buffer qg(2 * output_bytes, q_gate), k(kv_bytes, key), v(kv_bytes, value);
        Buffer qn(256 * sizeof(float), q_norm), kn(256 * sizeof(float), k_norm);
        Buffer q(output_bytes), gate(output_bytes), output(output_bytes);
        k_attention(output.floats(), kc.data(), vc.data(), q.floats(), gate.floats(),
                    qg.floats(), k.floats(), v.floats(), qn.floats(), kn.floats(),
                    batch, position, epsilon);
        finish();
        output.download(out);
        kc.download(key_cache);
        vc.download(value_cache);
    });
}

#include <array>
#include <fstream>
#include <memory>
#include <vector>

namespace {
constexpr int E=5120, F=17408, V=248320, L=64, NW=20;
constexpr int H=24, HKV=4, D=256, LK=16, LV=48, LD=128, C=(2*LK+LV)*LD;
constexpr int SLOTS=3+L*NW, TENSOR_COUNT=851;
constexpr size_t UPLOAD_BYTES=16*1024*1024;

struct Layout {
    int kind, width, rows;
    size_t length() const {
        if (kind==0) return size_t(width)*rows*sizeof(float);
        if (kind==8) return size_t(width/32)*34*rows;
        return size_t(width/256)*(kind==12 ? 144 : 210)*rows;
    }
};

Layout layout(int slot) {
    require(slot>=0 && slot<SLOTS, "invalid tensor slot");
    if(slot==0) return {12,E,V};
    if(slot==1) return {14,E,V};
    if(slot==2) return {0,E,1};
    int layer=(slot-3)/NW, index=(slot-3)%NW;
    switch(index) {
        case 0: case 1: return {0,E,1};
        case 2: case 3: return {12,E,F};
        case 4: return {12,F,E};
    }
    if(layer%4!=3) {
        switch(index) {
            case 5: return {8,E,C};
            case 6: return {8,E,LV*LD};
            case 7: case 8: return {8,E,LV};
            case 9: return {0,4,C};
            case 10: case 11: return {0,LV,1};
            case 12: return {0,LD,1};
            case 13: return {8,LV*LD,E};
        }
    } else {
        switch(index) {
            case 14: return {8,E,2*H*D};
            case 15: case 16: return {8,E,HKV*D};
            case 17: case 18: return {0,D,1};
            case 19: return {14,H*D,E};
        }
    }
    throw std::invalid_argument("tensor slot does not belong to this layer");
}

using Owned=std::unique_ptr<Buffer>;
Owned allocate(size_t size) { return std::make_unique<Buffer>(size); }
Owned float_buffer(size_t count) { return allocate(count*sizeof(float)); }

struct Model {
    int context, position=0, saved_position=0, uploaded=0;
    float epsilon;
    bool saved=false, poisoned=false;
    std::ifstream file;
    uint64_t file_bytes;
    std::vector<char> staging;
    std::array<Owned,SLOTS> weights;
    // For DeltaNet these are state/history; for attention they are K/V.
    std::array<Owned,L> state, history, saved_state, saved_history;
    Owned x, normalized, residual, qkv, gate, alpha, beta, mixed;
    Owned query, attention_gate, key, value, ffn_gate, ffn_up, logits, ids, projection_workspace;

    Model(const char *path,int capacity,float eps)
        : context(capacity),epsilon(eps),file(path,std::ios::binary|std::ios::ate),
          staging(UPLOAD_BYTES) {
        require(bool(file), "cannot open model file");
        const auto end=file.tellg();
        require(end>=0, "cannot determine model size");
        file_bytes=uint64_t(end);
        x=float_buffer(8*E); normalized=float_buffer(8*E); residual=float_buffer(8*E);
        qkv=float_buffer(8*C); gate=float_buffer(8*2*H*D);
        alpha=float_buffer(8*LV); beta=float_buffer(8*LV); mixed=float_buffer(8*LV*LD);
        query=float_buffer(8*H*D); attention_gate=float_buffer(8*H*D);
        key=float_buffer(8*HKV*D); value=float_buffer(8*HKV*D);
        ffn_gate=float_buffer(8*F); ffn_up=float_buffer(8*F);
        logits=float_buffer(8*V); ids=allocate(8*sizeof(int));
        projection_workspace=allocate(k_mm_workspace_bytes(8,F,V));
        for(int layer=0;layer<L;layer++) {
            state[layer]=allocate(state_bytes(layer));
            history[layer]=allocate(history_bytes(layer));
            saved_state[layer]=allocate(state_bytes(layer));
            saved_history[layer]=allocate(history_bytes(layer));
        }
        reset();
    }
    size_t state_bytes(int layer) const {
        return layer%4!=3 ? size_t(LV)*LD*LD*sizeof(float) : size_t(context)*HKV*D*sizeof(uint16_t);
    }
    size_t history_bytes(int layer) const {
        return layer%4!=3 ? size_t(3)*C*sizeof(float) : state_bytes(layer);
    }
    void ready() const {
        require(uploaded==TENSOR_COUNT, "model weights are incomplete");
        require(!poisoned, "model state is invalid after a CUDA failure; reset or restore");
    }
    void reset() {
        poisoned=true;
        saved=false;
        for(int layer=0;layer<L;layer++) if(layer%4!=3) {
            check(cudaMemset(state[layer]->data(),0,state_bytes(layer)));
            check(cudaMemset(history[layer]->data(),0,history_bytes(layer)));
        }
        finish();
        position=0;
        poisoned=false;
    }
    void upload(int slot,uint64_t offset,uint64_t length) {
        const Layout spec=layout(slot);
        require(!weights[slot], "tensor slot was already uploaded");
        require(length==spec.length(), "tensor length differs from supported layout");
        require(offset<=file_bytes && length<=file_bytes-offset, "tensor exceeds model file");
        Owned tensor=allocate(size_t(length));
        file.clear();
        file.seekg(std::streamoff(offset));
        require(bool(file), "cannot seek to tensor");
        for(uint64_t done=0;done<length;) {
            const size_t chunk=size_t(std::min(uint64_t(staging.size()),length-done));
            file.read(staging.data(),std::streamsize(chunk));
            require(size_t(file.gcount())==chunk, "truncated tensor payload");
            check(cudaMemcpy(static_cast<char *>(tensor->data())+done,staging.data(),chunk,cudaMemcpyHostToDevice));
            done+=chunk;
        }
        if(slot!=0 && spec.kind!=0) {
            Owned packed=allocate(k_mm_weight_bytes(spec.kind,spec.width,spec.rows));
            k_mm_pack_weights(packed->data(),tensor->data(),spec.kind,spec.width,spec.rows);
            finish();
            tensor=std::move(packed);
        }
        weights[slot]=std::move(tensor);
        uploaded++;
        if(uploaded==TENSOR_COUNT) {
            file.close();
            std::vector<char>().swap(staging);
        }
    }
    // Only adjacent projections of the same unchanged input reuse Q8 blocks.
    void projection(float *out,int slot,const float *input,int batch,bool reuse_input=false) {
        const Layout spec=layout(slot);
        k_mm_packed(out,weights[slot]->data(),input,spec.kind,batch,spec.width,spec.rows,projection_workspace->data(),!reuse_input);
        check(cudaGetLastError());
    }
    void evaluate(const int *tokens,int batch,float *out) {
        ready();
        batch_size(batch);
        require(tokens && out, "null evaluation buffer");
        require(position<=context-batch, "evaluation exceeds context capacity");
        for(int i=0;i<batch;i++) require(tokens[i]>=0 && tokens[i]<V, "token ID outside vocabulary");
        // Preflight failures above cannot mutate state. Later failures poison it.
        poisoned=true;
        check(cudaMemcpy(ids->data(),tokens,size_t(batch)*sizeof(int),cudaMemcpyHostToDevice));
        k_embed(x->floats(),weights[0]->data(),static_cast<int *>(ids->data()),batch);
        check(cudaGetLastError());
        for(int layer=0;layer<L;layer++) {
            const int base=3+layer*NW;
            k_norm(normalized->floats(),x->floats(),weights[base]->floats(),batch,E,epsilon);
            check(cudaGetLastError());
            if(layer%4!=3) {
                projection(qkv->floats(),base+5,normalized->floats(),batch);
                projection(gate->floats(),base+6,normalized->floats(),batch,true);
                projection(alpha->floats(),base+7,normalized->floats(),batch,true);
                projection(beta->floats(),base+8,normalized->floats(),batch,true);
                k_delta(mixed->floats(),state[layer]->floats(),history[layer]->floats(),
                        qkv->floats(),gate->floats(),alpha->floats(),beta->floats(),
                        weights[base+9]->floats(),weights[base+10]->floats(),
                        weights[base+11]->floats(),weights[base+12]->floats(),batch,LK,LV,LD,epsilon);
                check(cudaGetLastError());
                projection(residual->floats(),base+13,mixed->floats(),batch);
            } else {
                projection(gate->floats(),base+14,normalized->floats(),batch);
                projection(key->floats(),base+15,normalized->floats(),batch,true);
                projection(value->floats(),base+16,normalized->floats(),batch,true);
                k_attention(mixed->floats(),state[layer]->data(),history[layer]->data(),
                            query->floats(),attention_gate->floats(),gate->floats(),
                            key->floats(),value->floats(),weights[base+17]->floats(),
                            weights[base+18]->floats(),batch,position,epsilon);
                check(cudaGetLastError());
                projection(residual->floats(),base+19,mixed->floats(),batch);
            }
            k_add(x->floats(),residual->floats(),batch*E);
            check(cudaGetLastError());
            k_norm(normalized->floats(),x->floats(),weights[base+1]->floats(),batch,E,epsilon);
            check(cudaGetLastError());
            projection(ffn_gate->floats(),base+2,normalized->floats(),batch);
            projection(ffn_up->floats(),base+3,normalized->floats(),batch,true);
            k_swiglu(ffn_gate->floats(),ffn_gate->floats(),ffn_up->floats(),batch*F);
            check(cudaGetLastError());
            projection(residual->floats(),base+4,ffn_gate->floats(),batch);
            k_add(x->floats(),residual->floats(),batch*E);
            check(cudaGetLastError());
        }
        k_norm(normalized->floats(),x->floats(),weights[2]->floats(),batch,E,epsilon);
        check(cudaGetLastError());
        projection(logits->floats(),1,normalized->floats(),batch);
        finish();
        check(cudaMemcpy(out,logits->data(),size_t(batch)*V*sizeof(float),cudaMemcpyDeviceToHost));
        position+=batch;
        poisoned=false;
    }
    void copy_state(bool restoring) {
        if(restoring) require(saved, "no valid checkpoint");
        else { ready(); saved=false; }
        const int prefix=restoring ? saved_position : position;
        if(restoring) poisoned=true;
        for(int layer=0;layer<L;layer++) {
            size_t sb=state_bytes(layer),hb=history_bytes(layer);
            if(layer%4==3) sb=hb=size_t(prefix)*HKV*D*sizeof(uint16_t);
            if(sb) check(cudaMemcpy(restoring ? state[layer]->data() : saved_state[layer]->data(),
                                    restoring ? saved_state[layer]->data() : state[layer]->data(),
                                    sb,cudaMemcpyDeviceToDevice));
            if(hb) check(cudaMemcpy(restoring ? history[layer]->data() : saved_history[layer]->data(),
                                    restoring ? saved_history[layer]->data() : history[layer]->data(),
                                    hb,cudaMemcpyDeviceToDevice));
        }
        finish();
        if(restoring) { position=saved_position; poisoned=false; }
        else { saved_position=position; saved=true; }
    }
};
Model &model(void *handle) {
    require(handle!=nullptr, "null model handle");
    return *static_cast<Model *>(handle);
}
} // namespace

extern "C" int qv_create(void **handle,const char *path,int context,float epsilon) {
    return call([&] {
        require(handle!=nullptr, "null handle output");
        *handle=nullptr;
        require(path!=nullptr, "null model path");
        require(context>=1 && context<=2048, "context capacity must be 1..2048");
        epsilon_value(epsilon);
        *handle=new Model(path,context,epsilon);
    });
}
extern "C" int qv_upload(void *handle,int slot,uint64_t offset,uint64_t length) {
    return call([&] { model(handle).upload(slot,offset,length); });
}
extern "C" int qv_evaluate(void *handle,const int *tokens,int batch,float *out) {
    return call([&] { model(handle).evaluate(tokens,batch,out); });
}
extern "C" int qv_reset(void *handle) { return call([&] { model(handle).reset(); }); }
extern "C" int qv_checkpoint(void *handle) { return call([&] { model(handle).copy_state(false); }); }
extern "C" int qv_restore(void *handle) { return call([&] { model(handle).copy_state(true); }); }
extern "C" int qv_position(void *handle,int *position) {
    return call([&] { require(position!=nullptr,"null position output"); *position=model(handle).position; });
}
extern "C" void qv_destroy(void *handle) { delete static_cast<Model *>(handle); }
