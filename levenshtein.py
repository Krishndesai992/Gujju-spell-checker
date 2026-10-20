"""
levenshtein.py - Levenshtein Distance Implementation

This module provides a manual dynamic-programming implementation of Levenshtein
(Edit) Distance to calculate the minimum number of single-character edits
(insertions, deletions, substitutions) required to transform one word into another.

Designed specifically for educational transparency and viva explanation.
"""

from typing import List


def levenshtein_distance(str1: str, str2: str) -> int:
    """
    Computes the Levenshtein Distance between two strings using Dynamic Programming.

    Parameters:
        str1 (str): Source word (e.g. misspelled input)
        str2 (str): Target word (e.g. dictionary vocabulary candidate)

    Returns:
        int: Minimum number of edit operations (insertions, deletions, substitutions)

    Algorithm Explanation:
        Let DP[i][j] be the edit distance between str1[0..i-1] and str2[0..j-1].
        - Base Cases:
            DP[i][0] = i (deleting all characters of str1)
            DP[0][j] = j (inserting all characters of str2)
        - Transition:
            If characters match (str1[i-1] == str2[j-1]):
                DP[i][j] = DP[i-1][j-1] (no cost)
            Else:
                DP[i][j] = 1 + min(
                    DP[i-1][j],      # Deletion
                    DP[i][j-1],      # Insertion
                    DP[i-1][j-1]     # Substitution
                )
    """
    m = len(str1)
    n = len(str2)

    # Initialize DP matrix of dimensions (m + 1) x (n + 1)
    dp: List[List[int]] = [[0] * (n + 1) for _ in range(m + 1)]

    # Base cases: converting prefix of str1 to empty string requires 'i' deletions
    for i in range(m + 1):
        dp[i][0] = i

    # Base cases: converting empty string to prefix of str2 requires 'j' insertions
    for j in range(n + 1):
        dp[0][j] = j

    # Fill DP matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                cost = 0
            else:
                cost = 1

            dp[i][j] = min(
                dp[i - 1][j] + 1,       # Deletion from str1
                dp[i][j - 1] + 1,       # Insertion into str1
                dp[i - 1][j - 1] + cost  # Substitution
            )

    return dp[m][n]


if __name__ == "__main__":
    import sys
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            try:
                stream.reconfigure(encoding='utf-8', errors='replace')
            except Exception:
                pass
    # Test example from requirement specification:
    word1 = "શાલામાં"
    word2 = "શાળામાં"
    dist = levenshtein_distance(word1, word2)
    print(f"Levenshtein Distance between '{word1}' and '{word2}': {dist}")
    assert dist == 1, f"Expected 1, got {dist}"
    print("Levenshtein test passed successfully!")
