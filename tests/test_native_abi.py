"""The CPU compiler checks the ABI header without needing a CUDA toolkit."""
import pathlib
import shutil
import subprocess
import tempfile
import unittest


class NativeABITests(unittest.TestCase):
    @unittest.skipUnless(shutil.which('cc'), 'a C compiler is required')
    def test_public_header_is_usable_from_c(self):
        source = '''
#include "runtime.h"
int main(void) {
    const char *(*error)(void) = qv_error;
    int (*mm)(float *, const void *, const float *, int, int, int, int) = qv_test_mm;
    int (*norm)(float *, const float *, const float *, int, int, float) = qv_test_norm;
    return error == 0 || mm == 0 || norm == 0;
}
'''
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / 'abi.c'
            path.write_text(source)
            result = subprocess.run(
                ['cc', '-std=c11', '-Wall', '-Wextra', '-Werror', '-fsyntax-only',
                 '-Isrc', str(path)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
