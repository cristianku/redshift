import struct
import tempfile
import unittest
from pathlib import Path

from qvelox.gguf import read_gguf
from qvelox.gguf import GGUF, Tensor
from qvelox.model import EXPECTED, schema, validate_qwen27b


def string(s):
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def fixture(offset=0, dims=(256, 2), payload=288, extra=b""):
    metadata = string("general.architecture") + struct.pack("<I", 8) + string("qwen35")
    header = b"GGUF" + struct.pack("<IQQ", 3, 1, 1) + metadata
    header += string("weight") + struct.pack("<I", len(dims))
    header += struct.pack("<" + "Q" * len(dims), *dims) + struct.pack("<IQ", 12, offset)
    return header + bytes((-len(header)) % 32) + bytes(payload) + extra


class GGUFTests(unittest.TestCase):
    def read(self, data):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "fixture.gguf"
            p.write_bytes(data)
            return read_gguf(p)

    def test_quantized_tensor_size_and_absolute_offset(self):
        m = self.read(fixture())
        self.assertEqual(m.metadata["general.architecture"], "qwen35")
        t = m.tensors["weight"]
        self.assertEqual((t.shape, t.kind, t.nbytes), ((256, 2), 12, 288))
        self.assertEqual(t.offset % 32, 0)

    def test_rejects_truncated_header_and_weights(self):
        for data in (b"GGUF", fixture()[:-1]):
            with self.subTest(size=len(data)), self.assertRaises(ValueError):
                self.read(data)

    def test_rejects_invalid_quantization_shape(self):
        with self.assertRaises(ValueError):
            self.read(fixture(dims=(255, 2)))

    def test_rejects_out_of_file_tensor(self):
        with self.assertRaises(ValueError):
            self.read(fixture(offset=4096))

    def test_rejects_huge_counts_before_allocation(self):
        with self.assertRaises(ValueError):
            self.read(b"GGUF" + struct.pack("<IQQ", 3, 2**63, 0))

    def test_rejects_wrong_magic(self):
        with self.assertRaises(ValueError):
            self.read(b"NOPE" + fixture()[4:])

    def test_rejects_overlapping_tensor_payloads(self):
        header = b'GGUF' + struct.pack('<IQQ', 3, 2, 0)
        for name, offset in (('a', 0), ('b', 256)):
            header += string(name) + struct.pack('<IQQIQ', 2, 256, 2, 12, offset)
        data = header + bytes((-len(header)) % 32) + bytes(544)
        with self.assertRaisesRegex(ValueError, 'overlapping'):
            self.read(data)

    def test_rejects_duplicate_tensor_names(self):
        header = b'GGUF' + struct.pack('<IQQ', 3, 2, 0)
        for offset in (0, 320):
            header += string('a') + struct.pack('<IQQIQ', 2, 256, 2, 12, offset)
        data = header + bytes((-len(header)) % 32) + bytes(608)
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            self.read(data)


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.metadata = {'general.architecture': 'qwen35',
                         'qwen35.attention.layer_norm_rms_epsilon': 1e-6}
        self.metadata.update({'qwen35.' + key: value for key, value in EXPECTED.items()})
        self.tensors = {name: Tensor(name, shape, kind, 0, 0) for _, name, shape, kind in schema()}
        self.model = GGUF(Path('synthetic.gguf'), self.metadata, self.tensors)

    def test_accepts_supported_manifest(self):
        self.assertEqual(validate_qwen27b(self.model), 1e-6)

    def test_rejects_unsupported_architecture_dimensions_and_epsilon(self):
        for key, value in (('general.architecture', 'llama'),
                           ('qwen35.attention.head_count', 32),
                           ('qwen35.attention.layer_norm_rms_epsilon', float('nan'))):
            with self.subTest(key=key):
                original = self.metadata[key]
                self.metadata[key] = value
                with self.assertRaises(ValueError):
                    validate_qwen27b(self.model)
                self.metadata[key] = original

    def test_rejects_missing_extra_and_wrongly_quantized_tensor(self):
        weight = self.tensors.pop('output.weight')
        with self.assertRaises(ValueError):
            validate_qwen27b(self.model)
        self.tensors['output.weight'] = Tensor(weight.name, weight.shape, 12, 0, 0)
        with self.assertRaises(ValueError):
            validate_qwen27b(self.model)
        self.tensors['output.weight'] = weight
        self.tensors['unknown.weight'] = weight
        with self.assertRaises(ValueError):
            validate_qwen27b(self.model)


if __name__ == "__main__":
    unittest.main()
