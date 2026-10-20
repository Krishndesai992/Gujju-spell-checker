"""
test_preprocessing.py - Unit tests for Gujarati text preprocessing module.
"""

import unittest
from preprocessing import (
    unicode_normalize,
    whitespace_normalize,
    handle_punctuation,
    tokenize,
    handle_stopwords,
    preprocess,
    preprocess_with_details,
)


class TestPreprocessing(unittest.TestCase):
    def test_unicode_normalize(self):
        text = "ગુજરાતી"
        normalized = unicode_normalize(text)
        self.assertEqual(normalized, text)
        self.assertEqual(unicode_normalize(""), "")

    def test_whitespace_normalize(self):
        text = "  હું    શાલામાં   જાઉં    છું.  "
        expected = "હું શાલામાં જાઉં છું."
        self.assertEqual(whitespace_normalize(text), expected)
        self.assertEqual(whitespace_normalize(""), "")

    def test_handle_punctuation(self):
        text = "નમસ્તે! આ પુસ્તક, પેન અને નોટબુક છે। શું તમે આવશો?"
        cleaned = handle_punctuation(text)
        # Verify punctuation removed
        for punct in ["!", ",", "।", "?"]:
            self.assertNotIn(punct, cleaned)
        # Verify core words preserved
        self.assertIn("નમસ્તે", cleaned)
        self.assertIn("પુસ્તક", cleaned)

    def test_tokenize(self):
        text = "હું શાળામાં જાઉં છું"
        tokens = tokenize(text)
        self.assertEqual(tokens, ["હું", "શાળામાં", "જાઉં", "છું"])
        self.assertEqual(tokenize(""), [])

    def test_handle_stopwords_policy(self):
        tokens = ["હું", "શાળામાં", "જાઉં", "છું"]
        res = handle_stopwords(tokens)
        self.assertEqual(res["status"], "DISABLED")
        self.assertEqual(res["tokens"], tokens)

    def test_preprocess_end_to_end(self):
        raw = "  હું   શાલામાં,  જાઉં   છું! "
        tokens = preprocess(raw)
        self.assertEqual(tokens, ["હું", "શાલામાં", "જાઉં", "છું"])

    def test_preprocess_with_details(self):
        details = preprocess_with_details("હું શાળામાં જાઉં છું.")
        self.assertIn("original_text", details)
        self.assertIn("unicode_normalized", details)
        self.assertIn("whitespace_normalized", details)
        self.assertIn("punctuation_handled", details)
        self.assertIn("tokens", details)
        self.assertIn("stopword_handling", details)
        self.assertIn("final_tokens", details)
        self.assertEqual(details["final_tokens"], ["હું", "શાળામાં", "જાઉં", "છું"])


if __name__ == "__main__":
    unittest.main()
