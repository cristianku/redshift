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
