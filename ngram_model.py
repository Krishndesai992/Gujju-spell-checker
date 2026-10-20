"""
ngram_model.py - Statistical N-Gram Language Model

This module implements a manual statistical N-gram language model for Gujarati:
- Unigram, Bigram, and Trigram count tracking
- Raw and Laplace (add-one) smoothed probability calculations
- Model serialization (save/load to JSON)
- Contextual likelihood scoring for spell-checker candidate ranking
"""

import json
from collections import Counter
from typing import Dict, List, Tuple, Any, Optional


class NGramLanguageModel:
    """
    Statistical N-gram Language Model (Unigram, Bigram, Trigram)
    designed with educational clarity and explainability.
    """

    def __init__(self):
        self.vocab: List[str] = []
        self.vocab_set: set = set()
        self.total_words: int = 0
        self.unigram_counts: Counter = Counter()
        self.bigram_counts: Counter = Counter()
        self.trigram_counts: Counter = Counter()

    def train(self, tokenized_sentences: List[List[str]]) -> None:
        """
        Trains the N-gram language model from a list of tokenized sentences.
        Counts unigrams, bigrams, and trigrams.
        """
        self.unigram_counts.clear()
        self.bigram_counts.clear()
        self.trigram_counts.clear()
        self.vocab_set.clear()
        self.total_words = 0

        for tokens in tokenized_sentences:
            if not tokens:
                continue

            # Update unigrams
            for token in tokens:
                self.unigram_counts[token] += 1
                self.vocab_set.add(token)
                self.total_words += 1

            # Update bigrams: (w_{i-1}, w_i)
            for i in range(len(tokens) - 1):
                bg_key = f"{tokens[i]} {tokens[i+1]}"
                self.bigram_counts[bg_key] += 1

            # Update trigrams: (w_{i-2}, w_{i-1}, w_i)
            for i in range(len(tokens) - 2):
                tg_key = f"{tokens[i]} {tokens[i+1]} {tokens[i+2]}"
                self.trigram_counts[tg_key] += 1

        self.vocab = sorted(list(self.vocab_set))

    @property
    def vocab_size(self) -> int:
        return len(self.vocab_set)

    def get_unigram_prob(self, word: str) -> float:
        """
        Unigram Probability:
        P(w) = Count(w) / Total_Words
        """
        if self.total_words == 0:
            return 0.0
        return self.unigram_counts.get(word, 0) / self.total_words

    def get_bigram_prob(self, w1: str, w2: str, smooth: bool = True) -> float:
        """
        Bigram Probability:
        P(w2 | w1) = Count(w1, w2) / Count(w1)
        With Laplace smoothing:
        P_laplace(w2 | w1) = (Count(w1, w2) + 1) / (Count(w1) + V)
        """
        bg_key = f"{w1} {w2}"
        bg_count = self.bigram_counts.get(bg_key, 0)
        w1_count = self.unigram_counts.get(w1, 0)

        v = max(self.vocab_size, 1)

        if smooth:
            return (bg_count + 1) / (w1_count + v)
        else:
            return (bg_count / w1_count) if w1_count > 0 else 0.0

    def get_trigram_prob(self, w1: str, w2: str, w3: str, smooth: bool = True) -> float:
        """
        Trigram Probability:
        P(w3 | w1, w2) = Count(w1, w2, w3) / Count(w1, w2)
        With Laplace smoothing:
        P_laplace(w3 | w1, w2) = (Count(w1, w2, w3) + 1) / (Count(w1, w2) + V)
        """
        tg_key = f"{w1} {w2} {w3}"
        tg_count = self.trigram_counts.get(tg_key, 0)
        bg_key = f"{w1} {w2}"
        bg_count = self.bigram_counts.get(bg_key, 0)

        v = max(self.vocab_size, 1)

        if smooth:
            return (tg_count + 1) / (bg_count + v)
        else:
            return (tg_count / bg_count) if bg_count > 0 else 0.0

    def calculate_context_probability(self, candidate: str, index: int, tokens: List[str]) -> Dict[str, Any]:
        """
        Calculates the contextual probability score for substituting 'candidate'
        at 'index' in the given token sequence.

        Combines:
        1. Trigram context (P(candidate | w_{i-2}, w_{i-1}) and following trigrams)
        2. Bigram context (P(candidate | w_{i-1}) and P(w_{i+1} | candidate))
        3. Unigram prior (P(candidate))

        Returns a detailed dictionary of calculations for UI transparency and viva.
        """
        n = len(tokens)
        uni_prob = self.get_unigram_prob(candidate)

        details = {
            "candidate": candidate,
            "unigram_prob": round(uni_prob, 6),
            "bigrams_evaluated": [],
            "trigrams_evaluated": [],
            "raw_bigram_matches": 0,
            "raw_trigram_matches": 0,
            "context_score": 0.0,
            "explanation": ""
        }

        # Check for isolated single-word scenario
        if n <= 1:
            details["context_score"] = round(uni_prob, 6)
            details["explanation"] = f"Single word input: Unigram frequency prior P({candidate}) = {uni_prob:.6f}"
            return details

        # Collect bigram probabilities
        bigram_probs = []
        # Preceding bigram: (tokens[i-1], candidate)
        if index > 0:
            w_prev = tokens[index - 1]
            raw_count = self.bigram_counts.get(f"{w_prev} {candidate}", 0)
            prob = self.get_bigram_prob(w_prev, candidate, smooth=True)
            bigram_probs.append(prob)
            if raw_count > 0:
                details["raw_bigram_matches"] += 1
            details["bigrams_evaluated"].append({
                "context": f"({w_prev}, {candidate})",
                "raw_count": raw_count,
                "prob": round(prob, 6)
            })

        # Following bigram: (candidate, tokens[i+1])
        if index < n - 1:
            w_next = tokens[index + 1]
            raw_count = self.bigram_counts.get(f"{candidate} {w_next}", 0)
            prob = self.get_bigram_prob(candidate, w_next, smooth=True)
            bigram_probs.append(prob)
            if raw_count > 0:
                details["raw_bigram_matches"] += 1
            details["bigrams_evaluated"].append({
                "context": f"({candidate}, {w_next})",
                "raw_count": raw_count,
                "prob": round(prob, 6)
            })

        # Collect trigram probabilities
        trigram_probs = []
        # Preceding trigram: (tokens[i-2], tokens[i-1], candidate)
        if index >= 2:
            w_p2 = tokens[index - 2]
            w_p1 = tokens[index - 1]
            raw_count = self.trigram_counts.get(f"{w_p2} {w_p1} {candidate}", 0)
            prob = self.get_trigram_prob(w_p2, w_p1, candidate, smooth=True)
            trigram_probs.append(prob)
            if raw_count > 0:
                details["raw_trigram_matches"] += 1
            details["trigrams_evaluated"].append({
                "context": f"({w_p2}, {w_p1}, {candidate})",
                "raw_count": raw_count,
                "prob": round(prob, 6)
            })

        # Surrounding trigram: (tokens[i-1], candidate, tokens[i+1])
        if 0 < index < n - 1:
            w_prev = tokens[index - 1]
            w_next = tokens[index + 1]
            raw_count = self.trigram_counts.get(f"{w_prev} {candidate} {w_next}", 0)
            prob = self.get_trigram_prob(w_prev, candidate, w_next, smooth=True)
            trigram_probs.append(prob)
            if raw_count > 0:
                details["raw_trigram_matches"] += 1
            details["trigrams_evaluated"].append({
                "context": f"({w_prev}, {candidate}, {w_next})",
                "raw_count": raw_count,
                "prob": round(prob, 6)
            })

        # Calculate combined contextual score using linear interpolation:
        # Heavily reward exact trigram and bigram hits from training data
        avg_trigram = sum(trigram_probs) / len(trigram_probs) if trigram_probs else uni_prob
        avg_bigram = sum(bigram_probs) / len(bigram_probs) if bigram_probs else uni_prob

        # Bonus for raw n-gram co-occurrences in the corpus
        corpus_bonus = (details["raw_trigram_matches"] * 0.5) + (details["raw_bigram_matches"] * 0.2)

        # Standard linear interpolation: 0.5 Trigram + 0.35 Bigram + 0.15 Unigram + corpus_bonus
        score = (0.50 * avg_trigram) + (0.35 * avg_bigram) + (0.15 * uni_prob) + corpus_bonus
        details["context_score"] = round(score, 6)

        match_info = []
        if details["raw_trigram_matches"] > 0:
            match_info.append(f"{details['raw_trigram_matches']} trigram match(es)")
        if details["raw_bigram_matches"] > 0:
            match_info.append(f"{details['raw_bigram_matches']} bigram match(es)")
        if not match_info:
            match_info.append("smoothed transition probabilities + unigram prior")

        details["explanation"] = f"Context score {details['context_score']} based on {', '.join(match_info)}"
        return details

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the model into a dictionary suitable for JSON."""
        return {
            "metadata": {
                "vocab_size": self.vocab_size,
                "total_words": self.total_words,
                "unigram_count": len(self.unigram_counts),
                "bigram_count": len(self.bigram_counts),
                "trigram_count": len(self.trigram_counts)
            },
            "vocab": self.vocab,
            "unigram_counts": dict(self.unigram_counts),
            "bigram_counts": dict(self.bigram_counts),
            "trigram_counts": dict(self.trigram_counts)
        }

    def from_dict(self, data: Dict[str, Any]) -> None:
        """Loads model counts and vocabulary from dictionary."""
        self.vocab = data.get("vocab", [])
        self.vocab_set = set(self.vocab)
        self.total_words = data.get("metadata", {}).get("total_words", 0)
        self.unigram_counts = Counter(data.get("unigram_counts", {}))
        self.bigram_counts = Counter(data.get("bigram_counts", {}))
        self.trigram_counts = Counter(data.get("trigram_counts", {}))

    def save(self, filepath: str) -> None:
        """Saves model to a JSON file."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    def load(self, filepath: str) -> None:
        """Loads model from a JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.from_dict(data)
