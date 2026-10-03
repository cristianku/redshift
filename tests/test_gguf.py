import struct
import tempfile
import unittest
from pathlib import Path

from qvelox.gguf import read_gguf


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


if __name__ == "__main__":
    unittest.main()
