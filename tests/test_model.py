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

    def test_checkpoint_grows_with_prefix_and_restores_continuation(self):
        runtime = self.runtime
        for prefix in ([], [10], [10, 20, 30, 40, 50]):
            with self.subTest(prefix=prefix):
                runtime.reset()
                if prefix:
                    runtime.evaluate(prefix)
                runtime.checkpoint()
                expected = runtime.evaluate([60, 70])
                runtime.restore()
                self.close_rows(runtime.evaluate([60, 70]), expected)

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

    def test_transition_commit_matches_serial_continuation(self):
        runtime = self.runtime
        runtime.reset()
        runtime.evaluate([10, 20, 30])
        runtime.checkpoint()
        for batch in (2, 4, 8):
            tokens = [100 + 13*i for i in range(batch)]
            for accepted in range(batch+1):
                with self.subTest(batch=batch, accepted=accepted):
                    runtime.restore()
                    reference_rows = runtime.evaluate(tokens)
                    runtime.restore()
                    rows = runtime.verify(tokens)
                    self.close_rows(rows, reference_rows)
                    with self.assertRaises(RuntimeError):
                        runtime.evaluate([999])
                    with self.assertRaises((RuntimeError, ValueError)):
                        runtime.commit(batch+1)
                    runtime.commit(accepted)
                    self.assertEqual(runtime.position, 3+accepted)
                    with self.assertRaises(RuntimeError):
                        runtime.commit(accepted)
                    actual = runtime.evaluate([900, 901])
                    runtime.restore()
                    for token in tokens[:accepted]:
                        runtime.evaluate([token])
                    self.close_rows(actual, runtime.evaluate([900, 901]))
                    # Rejected KV rows must be overwritten on the next cycle.
                    runtime.restore()
                    runtime.verify(tokens)
                    runtime.commit(accepted)
                    runtime.verify([900, 901])
                    runtime.commit(1)
                    actual = runtime.evaluate([902])
                    runtime.restore()
                    for token in [*tokens[:accepted], 900]:
                        runtime.evaluate([token])
                    self.close_rows(actual, runtime.evaluate([902]))

    def test_restore_and_reset_abort_pending_verification(self):
        runtime = self.runtime
        runtime.reset()
        runtime.evaluate([10])
        runtime.checkpoint()
        runtime.verify([100, 101])
        with self.assertRaises(RuntimeError):
            runtime.checkpoint()
        with self.assertRaises(RuntimeError):
            runtime.verify([102])
        runtime.restore()
        self.assertEqual(runtime.position, 1)
        with self.assertRaises(RuntimeError):
            runtime.commit(1)
        runtime.verify([100])
        runtime.reset()
        self.assertEqual(runtime.position, 0)
        with self.assertRaises(RuntimeError):
            runtime.commit(0)

    def test_advance_matches_all_logit_argmax_and_state(self):
        runtime = self.runtime
        runtime.reset()
        runtime.evaluate([10,20,30])
        runtime.checkpoint()
        for batch in range(1,9):
            ids = [100+13*i for i in range(batch)]
            runtime.restore()
            reference = runtime.evaluate(ids)
            continuation = runtime.evaluate([900])
            runtime.restore()
            actual = runtime.advance(ids)
            self.assertEqual(actual,[max(range(len(row)),key=row.__getitem__) for row in reference])
            self.close_rows(runtime.evaluate([900]),continuation)

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
