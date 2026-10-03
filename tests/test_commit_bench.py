"""Benchmark accounting tests; GPU correctness is covered by test_model."""
import unittest


class NumericSession:
    """CPU fixture with observable prefix-dependent outputs and commit state."""
    def __init__(self):
        self.tokens = [10, 20]
        self.saved = None
        self.pending = None
        self.forward_inputs = 0

    @property
    def position(self):
        return len(self.tokens)

    def checkpoint(self):
        self.saved = list(self.tokens)

    def restore(self):
        self.tokens = list(self.saved)
        self.pending = None

    def evaluate(self, tokens):
        if self.pending is not None:
            raise RuntimeError('pending')
        self.forward_inputs += len(tokens)
        rows = []
        for token in tokens:
            self.tokens.append(token)
            rows.append([float(sum(self.tokens)), float(len(self.tokens))])
        return rows

    def verify(self, tokens):
        start = list(self.tokens)
        rows = self.evaluate(tokens)
        self.pending = (start, list(tokens))
        return rows

    def advance(self, tokens):
        return [max(range(len(row)), key=row.__getitem__) for row in self.evaluate(tokens)]

    def commit(self, accepted):
        start, tokens = self.pending
        self.tokens = start + tokens[:accepted]
        self.pending = None


class CommitBenchTests(unittest.TestCase):
    def test_controlled_acceptance_keeps_only_the_selected_prefix(self):
        from tools.bench_commit import run_cycle
        for strategy in ('replay', 'transitions'):
            for accepted in range(9):
                with self.subTest(strategy=strategy, accepted=accepted):
                    session = NumericSession()
                    result = run_cycle(session, list(range(100, 108)), accepted, strategy)
                    self.assertEqual(session.tokens, [10, 20, *range(100, 100+accepted)])
                    self.assertEqual(session.forward_inputs, 8+accepted if strategy=='replay' and accepted<8 else 8)
                    self.assertEqual(result['accepted'], accepted)
                    self.assertEqual(result['position_after'], 2+accepted)
                    self.assertAlmostEqual(result['total_seconds'],
                        sum(result[key] for key in ('checkpoint_seconds', 'verify_seconds',
                            'restore_seconds', 'replay_seconds', 'commit_seconds')))
                    self.assertGreaterEqual(result['total_seconds'], 0)

    def test_bad_cell_does_not_mutate_runtime(self):
        from tools.bench_commit import run_cycle
        for tokens, accepted, strategy in (([100], 2, 'replay'),
                ([100], -1, 'transitions'), ([], 0, 'replay'),
                ([100]*9, 0, 'transitions'), ([100], 0, 'unknown')):
            session = NumericSession()
            with self.assertRaises(ValueError):
                run_cycle(session, tokens, accepted, strategy)
            self.assertEqual(session.tokens, [10, 20])

    def test_logit_validation_rejects_drift_and_nan(self):
        from tools.bench_commit import compare_rows
        self.assertEqual(compare_rows([[1., 2.]], [[1., 2.]])['max_scaled_error'], 0)
        for actual in ([[1., 3.]], [[float('nan'), 2.]], [[1.]], []):
            with self.assertRaises(ValueError):
                compare_rows(actual, [[1., 2.]])

    def test_summary_uses_complete_cycle_and_zero_acceptance_is_zero(self):
        from tools.bench_commit import summarize
        samples = []
        for accepted, total in ((0, 2.), (0, 4.), (3, 2.), (3, 4.)):
            samples.append(dict(strategy='replay', prefix=128, batch=8, accepted=accepted,
                total_seconds=total, checkpoint_seconds=total*.1, verify_seconds=total*.5,
                restore_seconds=total*.1, replay_seconds=total*.3, commit_seconds=0.))
        cells = summarize(samples)
        self.assertEqual(len(cells), 2)
        self.assertEqual(cells[0]['accepted_inputs_per_second'], 0.)
        self.assertEqual(cells[1]['accepted_inputs_per_second'], 1.)
        self.assertEqual(cells[1]['timings']['total_seconds']['median'], 3.)
