"""Small, scalar double-precision references; never import the native kernels."""
import math
import struct


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def silu(x):
    return x * sigmoid(x)


def half(x):
    return struct.unpack('<e', struct.pack('<e', x))[0]


def delta(state, history, qkv, gate, alpha, beta, conv, decay, bias, norm,
          batch, key_heads, value_heads, dim, epsilon):
    state, history = list(state), list(history)
    channels = (2 * key_heads + value_heads) * dim
    output = []
    for t in range(batch):
        raw = qkv[t * channels:(t + 1) * channels]
        filtered = [silu(sum(conv[4 * c + j] * history[j * channels + c]
                             for j in range(3)) + conv[4 * c + 3] * raw[c])
                    for c in range(channels)]
        history = history[channels:] + list(raw)
        queries, keys = [], []
        for h in range(key_heads):
            q = filtered[h * dim:(h + 1) * dim]
            k = filtered[(key_heads + h) * dim:(key_heads + h + 1) * dim]
            queries.append([v / math.sqrt(sum(x*x for x in q) + 1e-6) / math.sqrt(dim) for v in q])
            keys.append([v / math.sqrt(sum(x*x for x in k) + 1e-6) for v in k])
        for h in range(value_heads):
            q, k = queries[h % key_heads], keys[h % key_heads]
            index = t * value_heads + h
            a = math.exp(decay[h] * math.log1p(math.exp(alpha[index] + bias[h])))
            b = sigmoid(beta[index])
            result = []
            for v in range(dim):
                start = (h * dim + v) * dim
                row = [a * x for x in state[start:start + dim]]
                target = filtered[2 * key_heads * dim + h * dim + v]
                residual = b * (target - sum(x*y for x, y in zip(row, k)))
                row = [x + residual * y for x, y in zip(row, k)]
                state[start:start + dim] = row
                result.append(sum(x*y for x, y in zip(row, q)))
            scale = 1 / math.sqrt(sum(x*x for x in result) / dim + epsilon)
            offset = (t * value_heads + h) * dim
            output.extend(x * scale * norm[i] * silu(gate[offset + i]) for i, x in enumerate(result))
    return output, state, history


def attention(key_cache, value_cache, q_gate, key, value, q_norm, k_norm,
              batch, position, epsilon):
    # Cache precision is part of the contract: round keys and values to FP16.
    kc = list(key_cache[:position * 1024])
    vc = list(value_cache[:position * 1024])
    output = []

    def prepare(row, weight, pos):
        inv = 1 / math.sqrt(sum(x*x for x in row) / 256 + epsilon)
        row = [x * inv * w for x, w in zip(row, weight)]
        for i in range(32):
            angle = pos * 10000000 ** (-i / 32)
            c, s = math.cos(angle), math.sin(angle)
            x, y = row[i], row[i + 32]
            row[i], row[i + 32] = x*c - y*s, x*s + y*c
        return row

    for t in range(batch):
        for h in range(4):
            offset = (t * 4 + h) * 256
            kc.extend(half(x) for x in prepare(key[offset:offset + 256], k_norm, position + t))
            vc.extend(half(x) for x in value[offset:offset + 256])
        for h in range(24):
            offset = (t * 24 + h) * 512
            query = prepare(q_gate[offset:offset + 256], q_norm, position + t)
            gate = q_gate[offset + 256:offset + 512]
            scores = [sum(query[i] * kc[(p * 4 + h // 6) * 256 + i] for i in range(256)) / 16
                      for p in range(position + t + 1)]
            peak = max(scores)
            scores = [math.exp(x - peak) for x in scores]
            total = sum(scores)
            for i in range(256):
                result = sum(w * vc[(p * 4 + h // 6) * 256 + i] for p, w in enumerate(scores)) / total
                output.append(result * sigmoid(gate[i]))
    return output, kc, vc
