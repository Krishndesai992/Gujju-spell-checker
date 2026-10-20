"""
spell_checker.py - Core Gujarati Spell Checker Module

Combines:
1. Visible Preprocessing (Unicode, Whitespace, Punctuation, Tokenization, Stopwords Retained)
2. Vocabulary Verification (against trained N-gram model corpus)
3. Candidate Generation (Filtering by Levenshtein Distance <= 1, <= 2, <= 3)
4. Candidate Ranking (Levenshtein Distance + Prefix/Suffix Matra Matching + N-gram Context)
5. Multi-candidate suggestions (minimum 6 suggestions ordered from most to least relevant)
"""

import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from preprocessing import preprocess_with_details
from levenshtein import levenshtein_distance
from ngram_model import NGramLanguageModel


class GujaratiSpellChecker:
    """
    Gujarati Spell Checker using a statistical N-gram Language Model
    and dynamic-programming Levenshtein Distance.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model = NGramLanguageModel()
        self.is_loaded = False
        base_dir = Path(__file__).resolve().parent
        default_path = base_dir / "model" / "ngram_model.json"
        if model_path:
            self.load_model(model_path)
        elif default_path.exists():
            self.load_model(str(default_path))

    def load_model(self, model_path: str) -> None:
        """Loads the trained statistical N-gram model from JSON."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        self.model.load(model_path)
        self.is_loaded = True

    def generate_candidates(self, word: str, min_needed: int = 6) -> List[Tuple[str, int]]:
        """
        Generates candidate words from vocabulary with minimum edit distances.
        Evaluates distance 1, distance 2, and distance 3 (if needed)
        to guarantee at least min_needed (default 6) suggestions.
        """
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded. Train or load the model first.")

        d1_list: List[Tuple[str, int]] = []
        d2_list: List[Tuple[str, int]] = []
        d3_list: List[Tuple[str, int]] = []

        w_len = len(word)

        for vocab_word in self.model.vocab:
            v_len = len(vocab_word)
            len_diff = abs(v_len - w_len)
            if len_diff > 3:
                continue

            dist = levenshtein_distance(word, vocab_word)
            if dist == 1:
                d1_list.append((vocab_word, 1))
            elif dist == 2:
                d2_list.append((vocab_word, 2))
            elif dist == 3:
                d3_list.append((vocab_word, 3))

        candidates = d1_list + d2_list
        if len(candidates) < min_needed:
            candidates += d3_list

        return candidates

    def rank_candidates(
        self,
        word: str,
        candidates: List[Tuple[str, int]],
        token_index: int,
        tokens: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Ranks candidate words using Google Keyboard style relevance:
        1. Levenshtein Distance (1 is better than 2 is better than 3)
        2. Prefix & Suffix match length (rewards words sharing start & end characters)
        3. Character set overlap (rewards words sharing letters/matras)
        4. N-gram Contextual Probability Score (or unigram frequency prior)

        Returns sorted candidate analysis dictionaries.
        """
        w_len = len(word)
        ranked = []

        for cand, dist in candidates:
            c_len = len(cand)

            # Common prefix length
            p_len = 0
            while p_len < min(w_len, c_len) and word[p_len] == cand[p_len]:
                p_len += 1

            # Common suffix length
            s_len = 0
            while s_len < min(w_len, c_len) and word[-1 - s_len] == cand[-1 - s_len]:
                s_len += 1

            # Character set overlap
            overlap = len(set(word) & set(cand))

            # Context probability
            context_info = self.model.calculate_context_probability(cand, token_index, tokens)
            score = context_info["context_score"]

            ranked.append({
                "word": cand,
                "levenshtein_distance": dist,
                "prefix_len": p_len,
                "suffix_len": s_len,
                "match_len": p_len + s_len,
                "overlap": overlap,
                "context_score": score,
                "unigram_prob": context_info["unigram_prob"],
                "raw_bigram_matches": context_info["raw_bigram_matches"],
                "raw_trigram_matches": context_info["raw_trigram_matches"],
                "explanation": context_info["explanation"]
            })

        # Multi-factor ranking:
        # 1. Lower edit distance (dist ascending)
        # 2. Higher prefix+suffix match length (match_len descending)
        # 3. Higher character overlap (overlap descending)
        # 4. Higher N-gram contextual likelihood (context_score descending)
        ranked.sort(key=lambda x: (x["levenshtein_distance"], -x["match_len"], -x["overlap"], -x["context_score"]))
        return ranked

    def check_word(
        self,
        word: str,
        token_index: int = 0,
        tokens: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Checks a single word against vocabulary.
        If correct: marks is_correct = True.
        If incorrect: generates at least 6 relevant suggestions ordered by relevance.
        """
        if tokens is None:
            tokens = [word]

        in_vocab = word in self.model.vocab_set

        if in_vocab:
            return {
                "word": word,
                "in_vocabulary": True,
                "is_correct": True,
                "suggested_correction": word,
                "levenshtein_distance": 0,
                "message": "Correct spelling",
                "context_info": f"Word is valid. Unigram frequency: {self.model.unigram_counts.get(word, 0)}",
                "top_candidates": []
            }

        # Misspelled or unknown word
        candidates = self.generate_candidates(word, min_needed=6)

        if not candidates:
            return {
                "word": word,
                "in_vocabulary": False,
                "is_correct": False,
                "suggested_correction": None,
                "levenshtein_distance": None,
                "message": "Possible spelling error: No close candidate found.",
                "context_info": "No matching candidate within distance threshold.",
                "top_candidates": []
            }

        ranked = self.rank_candidates(word, candidates, token_index, tokens)
        best = ranked[0]

        # Return at least 6 suggestions (up to 8)
        top_suggestions = ranked[:8]

        return {
            "word": word,
            "in_vocabulary": False,
            "is_correct": False,
            "suggested_correction": best["word"],
            "levenshtein_distance": best["levenshtein_distance"],
            "message": f"Possible spelling error. Suggested: '{best['word']}'",
            "context_info": best["explanation"],
            "top_candidates": top_suggestions
        }

    def check_text(self, text: str) -> Dict[str, Any]:
        """
        Main entry point for checking raw text (word or sentence).
        Performs full visible preprocessing, checks all tokens,
        and constructs a complete structured report.
        """
        if not self.is_loaded:
            return {
                "error": "Trained model not found. Please run train_ngram.py first.",
                "has_error": False,
                "tokens": [],
                "corrected_text": text
            }

        details = preprocess_with_details(text)
        tokens = details["final_tokens"]

        if not tokens:
            return {
                "original_text": text,
                "has_error": False,
                "is_empty": True,
                "message": "Please enter a valid Gujarati word or sentence.",
                "tokens": [],
                "corrected_text": "",
                "words_analysis": [],
                "preprocessing_details": details
            }

        words_analysis = []
        corrected_tokens = []
        any_misspelled = False

        for idx, token in enumerate(tokens):
            analysis = self.check_word(token, token_index=idx, tokens=tokens)
            words_analysis.append(analysis)

            if not analysis["is_correct"]:
                any_misspelled = True
                if analysis["suggested_correction"]:
                    corrected_tokens.append(analysis["suggested_correction"])
                else:
                    corrected_tokens.append(token)
            else:
                corrected_tokens.append(token)

        corrected_text = " ".join(corrected_tokens)

        # First error analysis for quick summary
        first_error_analysis = None
        for a in words_analysis:
            if not a["is_correct"]:
                first_error_analysis = a
                break

        return {
            "original_text": text,
            "has_error": any_misspelled,
            "is_empty": False,
            "tokens": tokens,
            "corrected_text": corrected_text,
            "first_error": first_error_analysis,
            "words_analysis": words_analysis,
            "preprocessing_details": details
        }


if __name__ == "__main__":
    import sys
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            try:
                stream.reconfigure(encoding='utf-8', errors='replace')
            except Exception:
                pass

    checker = GujaratiSpellChecker()
    print("Testing GujaratiSpellChecker:")

    sample_word = "ગરવ"
    res1 = checker.check_text(sample_word)
    print(f"\nInput: {sample_word}")
    print(f"Corrected: {res1['corrected_text']}")
    print(f"Top 6 Suggestions for '{sample_word}':")
    for i, c in enumerate(res1['words_analysis'][0]['top_candidates'][:6]):
        print(f"  {i+1}. {c['word']} (dist={c['levenshtein_distance']})")

    # Correct word test: આવકાર
    sample_word2 = "આવકાર"
    res2 = checker.check_text(sample_word2)
    print(f"\nInput: {sample_word2}")
    print(f"Corrected: {res2['corrected_text']}")
    print(f"Has error: {res2['has_error']}")
