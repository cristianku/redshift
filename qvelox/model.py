"""Explicit schema for the first supported dense checkpoint; no guessed layouts."""
import math

E, F, V = 5120, 17408, 248320
H, HKV, D = 24, 4, 256
LK, LV, LD = 16, 48, 128
C = (2 * LK + LV) * LD
LAYERS, NW = 64, 20
EXPECTED = {
    'block_count': LAYERS, 'embedding_length': E, 'feed_forward_length': F,
    'attention.head_count': H, 'attention.head_count_kv': HKV,
    'attention.key_length': D, 'attention.value_length': D,
    'rope.dimension_count': 64, 'rope.freq_base': 10000000.0,
    'full_attention_interval': 4, 'ssm.conv_kernel': 4,
    'ssm.state_size': LD, 'ssm.group_count': LK,
    'ssm.time_step_rank': LV, 'ssm.inner_size': LV * LD,
}


def schema():
    yield 0, 'token_embd.weight', (E, V), 12
    yield 1, 'output.weight', (E, V), 14
    yield 2, 'output_norm.weight', (E,), 0
    for i in range(LAYERS):
        common = [(0, 'attn_norm.weight', (E,), 0),
                  (1, 'post_attention_norm.weight', (E,), 0),
                  (2, 'ffn_gate.weight', (E, F), 12),
                  (3, 'ffn_up.weight', (E, F), 12),
                  (4, 'ffn_down.weight', (F, E), 12)]
        if i % 4 != 3:
            special = [(5, 'attn_qkv.weight', (E, C), 8),
                       (6, 'attn_gate.weight', (E, LV * LD), 8),
                       (7, 'ssm_alpha.weight', (E, LV), 8),
                       (8, 'ssm_beta.weight', (E, LV), 8),
                       (9, 'ssm_conv1d.weight', (4, C), 0),
                       (10, 'ssm_a', (LV,), 0),
                       (11, 'ssm_dt.bias', (LV,), 0),
                       (12, 'ssm_norm.weight', (LD,), 0),
                       (13, 'ssm_out.weight', (LV * LD, E), 8)]
        else:
            special = [(14, 'attn_q.weight', (E, 2 * H * D), 8),
                       (15, 'attn_k.weight', (E, HKV * D), 8),
                       (16, 'attn_v.weight', (E, HKV * D), 8),
                       (17, 'attn_q_norm.weight', (D,), 0),
                       (18, 'attn_k_norm.weight', (D,), 0),
                       (19, 'attn_output.weight', (H * D, E), 14)]
        for slot, name, shape, kind in common + special:
            yield 3 + i * NW + slot, f'blk.{i}.{name}', shape, kind


def validate_qwen27b(model):
    if model.metadata.get('general.architecture') != 'qwen35':
        raise ValueError('this prototype requires the qwen35 dense GGUF layout')
    for key, expected in EXPECTED.items():
        actual = model.metadata.get('qwen35.' + key)
        if actual != expected:
            raise ValueError(f'unsupported qwen35.{key}: {actual!r}, expected {expected}')
    eps = model.metadata.get('qwen35.attention.layer_norm_rms_epsilon')
    if not isinstance(eps, (int, float)) or not math.isfinite(eps) or eps <= 0:
        raise ValueError('invalid normalization epsilon')
    for _, name, shape, kind in schema():
        t = model.tensors.get(name)
        if t is None or (t.shape, t.kind) != (shape, kind):
            raise ValueError(f'unsupported or missing tensor {name}; expected {shape}, type {kind}')
    expected_names = {name for _, name, _, _ in schema()}
    if set(model.tensors) != expected_names:
        raise ValueError('unexpected model tensors; layout requires review')
    return float(eps)
