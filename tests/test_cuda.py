"""GPU tests use independently decoded weights and double-precision CPU sums."""
import ctypes as ct
import math
import os
import random
import struct
import unittest

from cpu_reference import attention, delta, half
from qvelox.runtime import load_library


def floats(values):
    return (ct.c_float * len(values))(*values)


def quantized(kind, k, m):
    rng = random.Random(471 + kind)
    data, reference = bytearray(), []
    for _ in range(m):
        row = []
        for _ in range(k // (32 if kind == 8 else 256)):
            if kind == 8:
                q = [rng.randrange(-127, 128) for _ in range(32)]
                d = 0.0078125
                data += struct.pack('<e32b', d, *q)
                row.extend(d * a for a in q)
            elif kind == 12:
                scales = bytes(rng.randrange(256) for _ in range(12))
                packed = bytes(rng.randrange(256) for _ in range(128))
                data += struct.pack('<ee', .03125, .015625) + scales + packed
                for i in range(256):
                    group = i // 32
                    if group < 4:
                        s, z = scales[group] & 63, scales[group + 4] & 63
                    else:
                        s = (scales[group + 4] & 15) | ((scales[group - 4] >> 6) << 4)
                        z = (scales[group + 4] >> 4) | ((scales[group] >> 6) << 4)
                    packed_index = i // 64 * 32 + i % 32
                    q = (packed[packed_index] >> (4 * (group % 2))) & 15
                    row.append(.03125 * s * q - .015625 * z)
            else:
                lo = bytes(rng.randrange(256) for _ in range(128))
                hi = bytes(rng.randrange(256) for _ in range(64))
                scales = [rng.randrange(-128, 128) for _ in range(16)]
                data += lo + hi + struct.pack('<16be', *scales, .00390625)
                for i in range(256):
                    half, within = divmod(i, 128)
                    quarter, lane = divmod(within, 32)
                    low = (lo[half * 64 + (quarter % 2) * 32 + lane] >> (4 * (quarter // 2))) & 15
                    high = (hi[half * 32 + lane] >> (2 * quarter)) & 3
                    row.append(((high * 16 + low) - 32) * scales[i // 16] * .00390625)
        reference.append(row)
    return bytes(data), reference


@unittest.skipUnless(os.environ.get('QVELOX_CUDA') == '1', 'set QVELOX_CUDA=1 on the V100')
class CUDATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lib = load_library()

    def check(self, result):
        self.assertEqual(result, 0, self.lib.qv_error())

    def close(self, actual, expected, tolerance=3e-5):
        self.assertEqual(len(actual), len(expected))
        for i, (a, e) in enumerate(zip(actual, expected)):
            self.assertTrue(math.isfinite(a), f'non-finite value at {i}')
            self.assertAlmostEqual(a, e, delta=tolerance * (1 + abs(e)), msg=f'index {i}')

    def test_rejects_invalid_arguments_before_accessing_buffers(self):
        value = floats([1.])
        for projection in (self.lib.qv_test_mm, self.lib.qv_test_mm_fast):
            for kind, batch, width, rows in ((0, 1, 256, 1), (8, 0, 32, 1),
                                             (12, 9, 256, 1), (14, 1, 255, 1),
                                             (8, 1, 32, 0)):
                self.assertEqual(projection(value, value, value, kind, batch, width, rows), -1)
                self.assertTrue(self.lib.qv_error())
            self.assertEqual(projection(None, value, value, 8, 1, 32, 1), -1)
        for epsilon in (0., -1., math.nan, math.inf):
            self.assertEqual(self.lib.qv_test_norm(value, value, value, 1, 1, epsilon), -1)
        # These dimensions must be rejected before any caller memory is read.
        self.assertEqual(self.lib.qv_test_delta(*([value] * 11), 1, 1, 1, 64, 1e-6), -1)
        self.assertEqual(self.lib.qv_test_attention(*([value] * 8), 8, 32761, 1e-6), -1)
        self.check(self.lib.qv_test_norm(value, value, value, 1, 1, 1e-6))
        self.assertEqual(self.lib.qv_error(), b'')

    def test_all_quantized_formats_all_candidate_rows(self):
        for kind in (8, 12, 14):
            data, weights = quantized(kind, 512, 7)
            packed = ct.create_string_buffer(data)
            for batch in (1, 2, 3, 4, 5, 8):
                with self.subTest(kind=kind, batch=batch):
                    x = floats([math.sin(i * .07) * .13 for i in range(batch * 512)])
                    out = floats([0.] * (batch * 7))
                    rc = self.lib.qv_test_mm(out, packed, x, kind, batch, 512, 7)
                    self.assertEqual(rc, 0, self.lib.qv_error())
                    for t in range(batch):
                        for row, w in enumerate(weights):
                            ref = sum(a * x[t * 512 + i] for i, a in enumerate(w))
                            self.assertTrue(math.isfinite(out[t * 7 + row]))
                            self.assertAlmostEqual(out[t * 7 + row], ref, delta=2e-5 * (1 + abs(ref)))

    def test_dp4a16_matches_independent_quantized_reference(self):
        # Check arithmetic separately from the lossy input representation. Zero
        # blocks and an outlier exercise scaling; short Q8 widths exercise tails.
        for kind in (8, 12, 14):
            for width in ((32, 96, 512, 768) if kind == 8 else (256, 512, 768)):
                rows = 131 if width == 512 else (35 if width == 768 else 7)
                data, weights = quantized(kind, width, rows)
                packed = ct.create_string_buffer(data)
                for batch in range(1, 9):
                    with self.subTest(kind=kind, width=width, batch=batch):
                        x = floats([.13 * math.sin(i * .07) for i in range(batch * width)])
                        for i in range(32):
                            x[i] = 0.
                        if width > 32:
                            x[37] = -3.25
                        qx = []
                        for start in range(0, len(x), 32):
                            values = x[start:start+32]
                            scale = ct.c_float(max(map(abs, values)) / 32639).value
                            for value in values:
                                ratio = ct.c_float(value / scale).value if scale else 0.
                                rounded = math.copysign(math.floor(abs(ratio) + .5), ratio)
                                qx.append(scale * rounded)
                        out = floats([0.] * (batch * rows))
                        self.check(self.lib.qv_test_mm_fast(out, packed, x, kind, batch, width, rows))
                        for t in range(batch):
                            for row, w in enumerate(weights):
                                offsets = [0.] * width
                                if kind == 12:
                                    for block in range(width // 256):
                                        pos = (row * (width // 256) + block) * 144
                                        minimum = struct.unpack_from('<e', data, pos+2)[0]
                                        scales = data[pos+4:pos+16]
                                        for group in range(8):
                                            z = scales[group+4] & 63 if group < 4 else ((scales[group+4] >> 4) | ((scales[group] >> 6) << 4))
                                            offsets[block*256+group*32:block*256+(group+1)*32] = [minimum*z] * 32
                                reference = sum((a+offsets[i])*qx[t*width+i] - offsets[i]*x[t*width+i]
                                                for i,a in enumerate(w))
                                self.assertAlmostEqual(out[t*rows+row], reference,
                                                       delta=3e-5*(1+abs(reference)))
                                original = sum(a*x[t*width+i] for i,a in enumerate(w))
                                # The exact per-element quantization-error bound,
                                # plus the same FP32 accumulation allowance.
                                bound = sum(abs((a+offsets[i])*(qx[t*width+i]-x[t*width+i]))
                                            for i,a in enumerate(w))
                                self.assertLessEqual(abs(out[t*rows+row]-original),
                                                     bound+3e-5*(1+abs(reference)))

    def test_weighted_rms_on_each_row(self):
        batch, width = 3, 5120
        x = floats([math.sin(i * .031) for i in range(batch * width)])
        w = floats([.8 + .2 * math.cos(i * .07) for i in range(width)])
        out = floats([0.] * (batch * width))
        self.assertEqual(self.lib.qv_test_norm(out, x, w, batch, width, 1e-6), 0)
        for t in range(batch):
            inv = 1 / math.sqrt(sum(x[t * width + i] ** 2 for i in range(width)) / width + 1e-6)
            for i in range(width):
                self.assertAlmostEqual(out[t * width + i], x[t * width + i] * inv * w[i], delta=2e-6)

    def test_delta_matches_cpu_and_sequential_nonzero_state(self):
        for dim in (32, 128):
            with self.subTest(dim=dim):
                batch, hk, hv = 3, 2, 4
                channels = (2 * hk + hv) * dim
                def wave(n, phase, scale=.1):
                    return floats([scale * math.sin(i * .037 + phase) for i in range(n)])
                state = wave(hv * dim * dim, .3, .01)
                history = wave(3 * channels, .7)
                qkv = wave(batch * channels, 1.1)
                gate = wave(batch * hv * dim, 1.7, .8)
                alpha, beta = wave(batch * hv, .2), wave(batch * hv, .9)
                conv = wave(channels * 4, .6, .4)
                decay, bias = floats([-.5 - h * .1 for h in range(hv)]), wave(hv, .4)
                norm = floats([.8 + i / dim * .2 for i in range(dim)])
                expected = delta(state, history, qkv, gate, alpha, beta, conv, decay,
                                 bias, norm, batch, hk, hv, dim, 1e-6)
                s, hist = floats(state), floats(history)
                out = floats([0.] * (batch * hv * dim))
                self.check(self.lib.qv_test_delta(out, s, hist, qkv, gate, alpha, beta,
                                                  conv, decay, bias, norm, batch, hk, hv, dim, 1e-6))
                for actual, reference in zip((out, s, hist), expected):
                    self.close(actual, reference)
                # Splitting a chunk must carry convolution AND recurrent state.
                seq_s, seq_h = floats(state), floats(history)
                sequential = []
                for t in range(batch):
                    row = floats([0.] * (hv * dim))
                    self.check(self.lib.qv_test_delta(
                        row, seq_s, seq_h, floats(qkv[t*channels:(t+1)*channels]),
                        floats(gate[t*hv*dim:(t+1)*hv*dim]), floats(alpha[t*hv:(t+1)*hv]),
                        floats(beta[t*hv:(t+1)*hv]), conv, decay, bias, norm, 1, hk, hv, dim, 1e-6))
                    sequential.extend(row)
                self.close(out, sequential)
                self.close(s, seq_s)
                self.close(hist, seq_h, 0)

    def test_attention_cpu_prefix_causality_and_chunking(self):
        batch, position = 3, 5
        def wave(n, phase, scale=.3):
            return floats([scale * math.sin(i * .051 + phase) for i in range(n)])
        def cache(values):
            return ct.create_string_buffer(struct.pack('<' + 'e' * len(values), *values))
        def unpack(buffer):
            return struct.unpack('<' + 'e' * ((position + batch) * 1024), buffer.raw[:-1])
        prefix_k = [half(v) for v in wave(position * 1024, .5)]
        prefix_v = [half(v) for v in wave(position * 1024, .9)]
        kc = cache(prefix_k + [0.] * (batch * 1024))
        vc = cache(prefix_v + [0.] * (batch * 1024))
        qg, key, value = wave(batch * 12288, 1.5), wave(batch * 1024, .1), wave(batch * 1024, .7)
        qn, kn = floats([1 + i / 512 for i in range(256)]), floats([1 - i / 512 for i in range(256)])
        expected = attention(prefix_k, prefix_v, qg, key, value, qn, kn, batch, position, 1e-6)
        out = floats([0.] * (batch * 6144))
        self.check(self.lib.qv_test_attention(out, kc, vc, qg, key, value, qn, kn, batch, position, 1e-6))
        self.close(out, expected[0], 3e-4)
        self.close(unpack(kc), expected[1], 2e-3)
        self.close(unpack(vc), expected[2], 0)
        self.assertEqual(list(unpack(kc)[:position * 1024]), prefix_k)
        # Sequential evaluation cannot see the later tokens in the chunk.
        seq_k = cache(prefix_k + [0.] * (batch * 1024))
        seq_v = cache(prefix_v + [0.] * (batch * 1024))
        sequential = []
        for t in range(batch):
            row = floats([0.] * 6144)
            self.check(self.lib.qv_test_attention(
                row, seq_k, seq_v, floats(qg[t*12288:(t+1)*12288]),
                floats(key[t*1024:(t+1)*1024]), floats(value[t*1024:(t+1)*1024]),
                qn, kn, 1, position + t, 1e-6))
            sequential.extend(row)
        self.close(out, sequential)
        self.assertEqual(kc.raw, seq_k.raw)
        self.assertEqual(vc.raw, seq_v.raw)

    def test_attention_long_context_against_closed_form(self):
        # Periodic exactly representable K/V give a scalar oracle without an
        # O(context*heads*dimension) Python reference. Nonuniform scores ensure
        # this tests softmax normalization and tile rescaling, not just averaging.
        key_rows = [struct.pack('<1024e', *([0.]*64+[level]*192)*4)
                    for level in ((i-3)*.25 for i in range(7))]
        value_rows = [struct.pack('<1024e', *[
            (p-5)*.125+h*.03125+(d%4)*.0625 for h in range(4) for d in range(256)])
            for p in range(11)]
        query = [0.]*64+[(i%4+1)*.03125 for i in range(192)]
        scale = 1/math.sqrt(sum(x*x for x in query)/256+1e-6)
        qsum = sum(query)*scale
        for position, batch in ((2047,3), (32767,1)):
            with self.subTest(position=position, batch=batch):
                prefix_k = b''.join(key_rows[p%7] for p in range(position))
                prefix_v = b''.join(value_rows[p%11] for p in range(position))
                kc = ct.create_string_buffer(prefix_k+bytes(batch*2048))
                vc = ct.create_string_buffer(prefix_v+bytes(batch*2048))
                qg = floats((query+[0.]*256)*(batch*24))
                k = floats([0.]*(batch*1024))
                v = floats([(t+1)*.125+h*.03125+(d%4)*.0625
                            for t in range(batch) for h in range(4) for d in range(256)])
                norm = floats([1.]*256)
                out = floats([0.]*(batch*6144))
                self.check(self.lib.qv_test_attention(out,kc,vc,qg,k,v,norm,norm,batch,position,1e-6))
                weights = [math.exp(qsum*((p%7-3)*.25)/16) for p in range(position)]
                total = sum(weights)
                numerator = sum(w*(p%11-5)*.125 for p,w in enumerate(weights))
                expected = []
                for t in range(batch):
                    total += 1
                    numerator += (t+1)*.125
                    expected.extend(.5*(numerator/total+(h//6)*.03125+(d%4)*.0625)
                                    for h in range(24) for d in range(256))
                self.close(out,expected,3e-5)
                self.assertEqual(kc.raw[:len(prefix_k)],prefix_k)
                self.assertEqual(vc.raw[:len(prefix_v)],prefix_v)
