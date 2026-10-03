"""Bounded GGUF v3 header reader. Weight payloads are never mapped or loaded."""
from dataclasses import dataclass
import math
from pathlib import Path
import struct

# GGML storage sizes: elements per block, bytes per block.
BLOCKS = {0: (1, 4), 1: (1, 2), 2: (32, 18), 8: (32, 34),
          12: (256, 144), 14: (256, 210)}
FORMATS = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i',
           6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}


@dataclass(frozen=True)
class Tensor:
    name: str
    shape: tuple
    kind: int
    offset: int
    nbytes: int


@dataclass(frozen=True)
class GGUF:
    path: Path
    metadata: dict
    tensors: dict


class Reader:
    def __init__(self, file, size):
        self.file, self.size = file, size

    def read(self, size):
        if size < 0 or size > self.size - self.file.tell():
            raise ValueError('truncated GGUF header')
        data = self.file.read(size)
        if len(data) != size:
            raise ValueError('truncated GGUF header')
        return data

    def number(self, fmt):
        return struct.unpack('<' + fmt, self.read(struct.calcsize('<' + fmt)))[0]

    def string(self):
        n = self.number('Q')
        if n > 16 * 1024 * 1024:
            raise ValueError('GGUF string exceeds header limit')
        try:
            return self.read(n).decode('utf-8')
        except UnicodeDecodeError as error:
            raise ValueError('invalid UTF-8 in GGUF header') from error

    def value(self, kind):
        if kind in FORMATS:
            return self.number(FORMATS[kind])
        if kind == 8:
            return self.string()
        if kind == 9:
            element, count = self.number('I'), self.number('Q')
            if element not in (*FORMATS, 8):
                raise ValueError('unsupported GGUF array element type')
            minimum = 8 if element == 8 else struct.calcsize('<' + FORMATS[element])
            if count > 2_000_000 or count > (self.size - self.file.tell()) // minimum:
                raise ValueError('invalid GGUF array length')
            return [self.value(element) for _ in range(count)]
        raise ValueError(f'unsupported GGUF metadata type {kind}')


def read_gguf(path):
    path = Path(path)
    with path.open('rb') as file:
        size = path.stat().st_size
        r = Reader(file, size)
        if r.read(4) != b'GGUF' or r.number('I') != 3:
            raise ValueError('expected little-endian GGUF version 3')
        count, fields = r.number('Q'), r.number('Q')
        if count > min(100_000, size // 24) or fields > min(100_000, size // 12):
            raise ValueError('invalid GGUF header counts')
        metadata = {}
        for _ in range(fields):
            name = r.string()
            if name in metadata:
                raise ValueError(f'duplicate metadata key {name}')
            metadata[name] = r.value(r.number('I'))
        alignment = metadata.get('general.alignment', 32)
        if type(alignment) is not int or not 1 <= alignment <= 4096 or alignment & (alignment - 1):
            raise ValueError('invalid tensor alignment')
        records, names = [], set()
        for _ in range(count):
            name, ndim = r.string(), r.number('I')
            if name in names or not 1 <= ndim <= 4:
                raise ValueError('duplicate tensor name or invalid dimensions')
            names.add(name)
            shape = tuple(r.number('Q') for _ in range(ndim))
            kind, offset = r.number('I'), r.number('Q')
            if kind not in BLOCKS:
                raise ValueError(f'unsupported tensor storage type {kind}')
            block, nbytes = BLOCKS[kind]
            if not all(shape) or shape[0] % block:
                raise ValueError(f'invalid quantization shape for {name}')
            nbytes *= math.prod(shape) // block
            if nbytes > size or offset % alignment:
                raise ValueError(f'invalid tensor size or offset for {name}')
            records.append((name, shape, kind, offset, nbytes))
        data_start = (file.tell() + alignment - 1) // alignment * alignment
        end, tensors = data_start, {}
        for name, shape, kind, offset, nbytes in sorted(records, key=lambda x: x[3]):
            absolute = data_start + offset
            if absolute < end or absolute + nbytes > size:
                raise ValueError(f'overlapping or truncated tensor {name}')
            tensors[name] = Tensor(name, shape, kind, absolute, nbytes)
            end = absolute + nbytes
        return GGUF(path, metadata, tensors)
