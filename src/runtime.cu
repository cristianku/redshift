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

extern "C" int qv_test_mm(float *out, const void *weights, const float *x,
                           int kind, int batch, int width, int rows) {
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
        k_mm(output.floats(), w.data(), input.floats(), kind, batch, width, rows);
        finish();
        output.download(out);
    });
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
