"""ctypes declarations for the synchronous native kernel validation ABI."""
import ctypes as ct
from pathlib import Path


def load_library(path=None):
    path = Path(path) if path is not None else Path(__file__).resolve().parents[1] / 'build/libqvelox.so'
    lib = ct.CDLL(str(path.resolve()))
    lib.qv_error.argtypes = []
    lib.qv_error.restype = ct.c_char_p
    signatures = {
        'qv_test_mm': [ct.c_void_p] * 3 + [ct.c_int] * 4,
        'qv_test_norm': [ct.c_void_p] * 3 + [ct.c_int] * 2 + [ct.c_float],
        'qv_test_delta': [ct.c_void_p] * 11 + [ct.c_int] * 4 + [ct.c_float],
        'qv_test_attention': [ct.c_void_p] * 8 + [ct.c_int] * 2 + [ct.c_float],
    }
    for name, signature in signatures.items():
        function = getattr(lib, name)
        function.argtypes = signature
        function.restype = ct.c_int
    return lib


class Runtime:
    """Single-threaded numeric model session. Input is token IDs, output is all logits."""
    def __init__(self, path, context=2048, library=None):
        from .gguf import read_gguf
        from .model import schema, validate_qwen27b
        self._handle = ct.c_void_p()
        self.model = read_gguf(path)
        epsilon = validate_qwen27b(self.model)
        if type(context) is not int or not 1 <= context <= 2048:
            raise ValueError('context must be an integer in 1..2048')
        self.lib = load_library(library)
        signatures = {
            'qv_create': [ct.POINTER(ct.c_void_p), ct.c_char_p, ct.c_int, ct.c_float],
            'qv_upload': [ct.c_void_p, ct.c_int, ct.c_uint64, ct.c_uint64],
            'qv_evaluate': [ct.c_void_p, ct.POINTER(ct.c_int), ct.c_int, ct.POINTER(ct.c_float)],
            'qv_position': [ct.c_void_p, ct.POINTER(ct.c_int)],
            'qv_reset': [ct.c_void_p], 'qv_checkpoint': [ct.c_void_p], 'qv_restore': [ct.c_void_p],
        }
        for name, signature in signatures.items():
            getattr(self.lib, name).argtypes = signature
            getattr(self.lib, name).restype = ct.c_int
        self.lib.qv_destroy.argtypes = [ct.c_void_p]
        self.lib.qv_destroy.restype = None
        import os
        self._check(self.lib.qv_create(ct.byref(self._handle), os.fsencode(self.model.path), context, epsilon))
        try:
            for slot, name, _, _ in schema():
                tensor = self.model.tensors[name]
                self._check(self.lib.qv_upload(self._handle, slot, tensor.offset, tensor.nbytes))
        except BaseException:
            self.close()
            raise

    def _check(self, code):
        if code:
            raise RuntimeError(self.lib.qv_error().decode('utf-8', errors='replace'))

    def _open(self):
        if not self._handle:
            raise RuntimeError('model session is closed')

    @property
    def position(self):
        self._open()
        result = ct.c_int()
        self._check(self.lib.qv_position(self._handle, ct.byref(result)))
        return result.value

    def evaluate(self, tokens):
        from array import array
        self._open()
        tokens = list(tokens)
        if not 1 <= len(tokens) <= 8 or any(type(t) is not int or not 0 <= t < 248320 for t in tokens):
            raise ValueError('expected 1..8 integer token IDs in 0..248319')
        ids = (ct.c_int * len(tokens))(*tokens)
        output = array('f', [0.]) * (len(tokens) * 248320)
        pointer = (ct.c_float * len(output)).from_buffer(output)
        self._check(self.lib.qv_evaluate(self._handle, ids, len(tokens), pointer))
        return [output[i*248320:(i+1)*248320] for i in range(len(tokens))]

    def reset(self):
        self._open()
        self._check(self.lib.qv_reset(self._handle))

    def checkpoint(self):
        self._open()
        self._check(self.lib.qv_checkpoint(self._handle))

    def restore(self):
        self._open()
        self._check(self.lib.qv_restore(self._handle))

    def close(self):
        if self._handle:
            self.lib.qv_destroy(self._handle)
            self._handle = ct.c_void_p()

    def __enter__(self):
        self._open()
        return self

    def __exit__(self, *_):
        self.close()
