import unittest

from causal_engine.intrical.coherence import CoherenceEngine, Evidence, normalize_term


class CoherenceEngineTests(unittest.TestCase):
    def test_bayes_update_with_likelihood_ratio(self):
        engine = CoherenceEngine()
        engine.register("Temporal fluidity", prior=0.5)
        updated = engine.update("temporal FLUIDITY", Evidence("source-1", 3.0, note="illustrative", dimensions=("time",), terms=("Temporal Flow",)))
        self.assertAlmostEqual(updated.posterior, 0.75)
        self.assertIn("time", updated.dimensions)
        self.assertEqual(engine.find(" TEMPORAL   FLOW ")[0].concept, "Temporal fluidity")

    def test_inertia_smooths_but_does_not_reverse_update(self):
        engine = CoherenceEngine()
        engine.register("coherence", prior=0.5, inertia=0.5)
        state = engine.update("coherence", Evidence("e1", 3.0))
        self.assertAlmostEqual(state.posterior, 0.625)

    def test_rejects_bad_probability_likelihood_and_duplicate_evidence(self):
        for bad in (0, -1, float("nan"), float("inf"), True):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                Evidence("bad", bad)
        engine = CoherenceEngine()
        with self.assertRaises(ValueError):
            engine.register("bad", prior=1.0)
        engine.register("ok")
        item = Evidence("same-id", 2.0)
        engine.update("ok", item)
        with self.assertRaises(ValueError):
            engine.update("ok", item)

    def test_export_is_provenance_aware_and_does_not_claim_quantum(self):
        engine = CoherenceEngine()
        engine.register("sentience", dimensions=("philosophy",), lexicon=("awareness",))
        engine.update("sentience", Evidence("paper:example", 1.2, note="placeholder; not empirical evidence"))
        exported = engine.export()
        self.assertFalse(exported["quantum_claim"])
        self.assertEqual(exported["model"], "classical_bayesian_log_odds_with_inertia")
        self.assertEqual(len(exported["states"]["sentience"]["evidence"]), 1)
        self.assertTrue(exported["limitations"])

    def test_term_normalization(self):
        self.assertEqual(normalize_term("  Multi   Dimensional "), "multi dimensional")
        with self.assertRaises(ValueError):
            normalize_term("  ")


if __name__ == "__main__":
    unittest.main()
