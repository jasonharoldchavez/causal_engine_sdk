import json
import math
import unittest

from causal_engine.intrical.coherence import CoherenceEngine, Evidence, normalize_term


class CoherenceEngineTests(unittest.TestCase):
    def test_bayes_update_with_likelihood_ratio(self):
        engine = CoherenceEngine()
        engine.register('Temporal fluidity', prior=0.5)
        updated = engine.update('temporal FLUIDITY', Evidence('source-1', 3.0, dimensions=('time',), terms=('Temporal Flow',)))
        self.assertAlmostEqual(updated.posterior, 0.75)
        self.assertIn('time', updated.dimensions)
        self.assertEqual(engine.find(' TEMPORAL   FLOW ')[0].concept, 'Temporal fluidity')

    def test_inertia_smooths_update(self):
        engine = CoherenceEngine()
        engine.register('coherence', prior=0.5, inertia=0.5)
        self.assertAlmostEqual(engine.update('coherence', Evidence('e1', 3.0)).posterior, 0.625)

    def test_rejects_invalid_likelihood_ratios(self):
        for bad in (0, -1, float('nan'), float('inf'), True, '2', None):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                Evidence('bad', bad)

    def test_rejects_invalid_prior_and_inertia(self):
        for bad in (0, 1, float('nan'), True, '0.5', None):
            with self.subTest(prior=bad), self.assertRaises(ValueError):
                CoherenceEngine().register('bad', prior=bad)
        for bad in (-0.1, 1, float('nan'), True, '0.5', None):
            with self.subTest(inertia=bad), self.assertRaises(ValueError):
                CoherenceEngine().register('bad', inertia=bad)

    def test_rejects_malformed_dimensions_and_lexicon(self):
        with self.assertRaises(ValueError):
            Evidence('e1', 1.2, dimensions='time')
        with self.assertRaises(ValueError):
            Evidence('e2', 1.2, terms=('valid', ' '))
        with self.assertRaises(ValueError):
            CoherenceEngine().register('bad', dimensions='time')
        with self.assertRaises(ValueError):
            CoherenceEngine().register('bad', lexicon=('valid', ''))

    def test_extreme_likelihood_ratios_remain_finite(self):
        engine = CoherenceEngine()
        engine.register('extreme')
        for source_id, ratio in (('high', 1e300), ('low', 1e-300)):
            posterior = engine.update('extreme', Evidence(source_id, ratio)).posterior
            self.assertTrue(math.isfinite(posterior))
            self.assertGreater(posterior, 0.0)
            self.assertLess(posterior, 1.0)

    def test_duplicate_source_and_unknown_concept_rejected(self):
        engine = CoherenceEngine()
        engine.register('known')
        item = Evidence('source', 2.0)
        engine.update('known', item)
        with self.assertRaises(ValueError):
            engine.update('known', item)
        with self.assertRaises(KeyError):
            engine.update('missing', Evidence('other', 2.0))

    def test_export_is_json_compatible_and_disclaims_quantum_claim(self):
        engine = CoherenceEngine()
        engine.register('sentience', dimensions=('philosophy',), lexicon=('awareness',))
        engine.update('sentience', Evidence('paper:example', 1.2, note='illustrative only'))
        exported = engine.export()
        self.assertFalse(exported['quantum_claim'])
        self.assertEqual(exported['model'], 'classical_bayesian_log_odds_with_inertia')
        self.assertEqual(len(exported['states']['sentience']['evidence']), 1)
        self.assertTrue(exported['limitations'])
        json.dumps(exported)

    def test_term_normalization(self):
        self.assertEqual(normalize_term('  Multi   Dimensional '), 'multi dimensional')
        with self.assertRaises(ValueError):
            normalize_term('  ')


if __name__ == '__main__':
    unittest.main()
