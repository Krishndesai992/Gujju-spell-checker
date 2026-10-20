"""
test_ngram_model.py - Unit tests for statistical N-gram language model.
"""

import unittest
import tempfile
from pathlib import Path
from ngram_model import NGramLanguageModel


class TestNGramModel(unittest.TestCase):
    def setUp(self):
        self.model = NGramLanguageModel()
        self.corpus = [
            ["હું", "શાળામાં", "જાઉં", "છું"],
            ["તે", "શાળામાં", "અભ્યાસ", "કરે", "છે"],
            ["હું", "પુસ્તક", "વાંચું", "છું"]
        ]
        self.model.train(self.corpus)

    def test_vocab_size(self):
        self.assertGreater(self.model.vocab_size, 0)
        self.assertIn("શાળામાં", self.model.vocab_set)
        self.assertIn("હું", self.model.vocab_set)

    def test_unigram_prob(self):
        p_hu = self.model.get_unigram_prob("હું")
        self.assertGreater(p_hu, 0.0)
        self.assertLessEqual(p_hu, 1.0)
        self.assertEqual(self.model.get_unigram_prob("અજ્ઞાતશબ્દ"), 0.0)

    def test_bigram_prob_with_smoothing(self):
        p_bg = self.model.get_bigram_prob("હું", "શાળામાં", smooth=True)
        self.assertGreater(p_bg, 0.0)
        # Unseen transition with Laplace smoothing should still be > 0
        p_unseen = self.model.get_bigram_prob("હું", "કમ્પ્યુટર", smooth=True)
        self.assertGreater(p_unseen, 0.0)

    def test_save_and_load(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            self.model.save(str(tmp_path))
            new_model = NGramLanguageModel()
            new_model.load(str(tmp_path))
            self.assertEqual(new_model.vocab_size, self.model.vocab_size)
            self.assertEqual(new_model.total_words, self.model.total_words)
            self.assertEqual(new_model.unigram_counts, self.model.unigram_counts)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()


if __name__ == "__main__":
    unittest.main()
