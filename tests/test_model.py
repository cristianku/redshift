"""Full-checkpoint tests are opt-in and require the existing 27B GGUF."""
import math
import os
import unittest


@unittest.skipUnless(os.environ.get('QVELOX_MODEL'), 'set QVELOX_MODEL to the existing GGUF')
class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from qvelox.runtime import Runtime
        cls.runtime = Runtime(os.environ['QVELOX_MODEL'], context=32)
        cls.addClassCleanup(cls.runtime.close)

    def close_rows(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for row, reference in zip(actual, expected):
            self.assertEqual(len(row), 248320)
            self.assertEqual(len(reference), 248320)
            worst = max(abs(a-b) / (1+abs(b)) for a, b in zip(row, reference))
            self.assertTrue(all(math.isfinite(x) for x in row))
            self.assertLessEqual(worst, 3e-4)
            self.assertEqual(max(range(len(row)), key=row.__getitem__),
                             max(range(len(reference)), key=reference.__getitem__))

    def test_all_rows_match_sequential_at_nonzero_prefix(self):
        runtime = self.runtime
        runtime.reset()
        runtime.evaluate([10, 20, 30])
        runtime.checkpoint()
        for batch in range(1, 9):
            with self.subTest(batch=batch):
                ids = [100 + i * 13 for i in range(batch)]
                runtime.restore()
                grouped = runtime.evaluate(ids)
                runtime.restore()
                sequential = [runtime.evaluate([token])[0] for token in ids]
                self.close_rows(grouped, sequential)

    def test_rejected_suffix_restore_and_continuation(self):
        runtime = self.runtime
        runtime.reset()
        runtime.evaluate([10, 20, 30])
        runtime.checkpoint()
        runtime.evaluate([100, 101, 102, 103, 104, 105, 106, 107])
        runtime.restore()
        runtime.evaluate([100, 101, 102])
        actual = runtime.evaluate([900, 901])
        runtime.reset()
        runtime.evaluate([10, 20, 30, 100, 101, 102])
        self.close_rows(actual, runtime.evaluate([900, 901]))

    def test_rejected_call_leaves_position_unchanged(self):
        runtime = self.runtime
        runtime.reset()
        for _ in range(4):
            runtime.evaluate([10] * 8)
        with self.assertRaisesRegex(RuntimeError, 'context'):
            runtime.evaluate([10])
        self.assertEqual(runtime.position, 32)
        runtime.reset()
        with self.assertRaises(ValueError):
            runtime.evaluate([-1])
        self.assertEqual(runtime.position, 0)
        with self.assertRaisesRegex(RuntimeError, 'checkpoint'):
            runtime.restore()
