"""
test_levenshtein.py - Unit tests for dynamic programming Levenshtein distance.
"""

import unittest
from levenshtein import levenshtein_distance


class TestLevenshteinDistance(unittest.TestCase):
    def test_identical_strings(self):
        self.assertEqual(levenshtein_distance("શાળા", "શાળા"), 0)
        self.assertEqual(levenshtein_distance("", ""), 0)

    def test_single_substitution(self):
        # 'શાલામાં' vs 'શાળામાં' -> 1 substitution (લ -> ળ)
        self.assertEqual(levenshtein_distance("શાલામાં", "શાળામાં"), 1)

    def test_insertion_and_deletion(self):
        # 'શાલા' (4 chars) -> 'શાળામાં' (7 chars): 1 substitution (લ->ળ) + 3 insertions (મ, ા, ં) = 4
        self.assertEqual(levenshtein_distance("શાલા", "શાળામાં"), 4)
        self.assertEqual(levenshtein_distance("", "ગુજરાત"), len("ગુજરાત"))
        self.assertEqual(levenshtein_distance("ગુજરાત", ""), len("ગુજરાત"))

    def test_symmetric_property(self):
        w1 = "અભ્યાસ"
        w2 = "અભાસ"
        self.assertEqual(levenshtein_distance(w1, w2), levenshtein_distance(w2, w1))


if __name__ == "__main__":
    unittest.main()
