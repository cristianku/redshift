"""GPU tests use independently decoded weights and double-precision CPU sums."""
import ctypes as ct
import math
import os
import random
import struct
import unittest


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
        cls.lib = ct.CDLL('build/libqvelox.so')
        cls.lib.qv_error.restype = ct.c_char_p
        cls.lib.qv_test_mm.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_void_p] + [ct.c_int] * 4
        cls.lib.qv_test_norm.argtypes = [ct.c_void_p] * 3 + [ct.c_int, ct.c_int, ct.c_float]

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
