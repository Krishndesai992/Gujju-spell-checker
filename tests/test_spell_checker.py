"""
test_spell_checker.py - Unit tests for Gujarati Spell Checker pipeline.
"""

import unittest
from pathlib import Path
from spell_checker import GujaratiSpellChecker


class TestSpellChecker(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model_path = Path(__file__).resolve().parent.parent / "model" / "ngram_model.json"
        if not model_path.exists():
            raise unittest.SkipTest("Model file not trained. Run train_ngram.py first.")
        cls.checker = GujaratiSpellChecker(str(model_path))

    def test_checker_loaded(self):
        self.assertTrue(self.checker.is_loaded)
        self.assertGreater(self.checker.model.vocab_size, 0)

    def test_candidate_generation_at_least_six(self):
        # Misspelled word: 'શાલામાં' (expected: 'શાળામાં')
        candidates = self.checker.generate_candidates("શાલામાં", min_needed=6)
        self.assertGreaterEqual(len(candidates), 6)
        candidate_words = [c[0] for c in candidates]
        self.assertIn("શાળામાં", candidate_words)

    def test_single_word_correction(self):
        res = self.checker.check_text("શાલામાં")
        self.assertTrue(res["has_error"])
        self.assertEqual(res["corrected_text"], "શાળામાં")

    def test_valid_word_preserved(self):
        res = self.checker.check_text("શાળામાં")
        self.assertFalse(res["has_error"])
        self.assertEqual(res["corrected_text"], "શાળામાં")

    def test_sentence_correction(self):
        res = self.checker.check_text("હું શાલામાં જાઉં છું.")
        self.assertTrue(res["has_error"])
        self.assertIn("શાળામાં", res["corrected_text"])


if __name__ == "__main__":
    unittest.main()
