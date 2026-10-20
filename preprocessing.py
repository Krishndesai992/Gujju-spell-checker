"""
preprocessing.py - Text Preprocessing Module for Gujarati Spell Checker

This module provides visible, academic-standard preprocessing stages for Gujarati text:
1. Unicode Normalization (NFC)
2. Whitespace Normalization
3. Punctuation Handling (preserving Gujarati characters while removing punctuation)
4. Tokenization
5. Stopword Handling (EXPLICITLY DISABLED to preserve N-gram context)
"""

import re
import unicodedata
from typing import Dict, List, Any


def unicode_normalize(text: str) -> str:
    """
    Normalizes Gujarati text using Unicode Normalization Form C (NFC).
    Ensures composite characters (vowel matras, conjuncts) are canonical.
    """
    if not text:
        return ""
    return unicodedata.normalize('NFC', text)


def whitespace_normalize(text: str) -> str:
    """
    Normalizes whitespace by replacing tabs, newlines, and multiple spaces
    with a single space, and strips leading and trailing whitespace.
    """
    if not text:
        return ""
    return re.sub(r'\s+', ' ', text).strip()


def handle_punctuation(text: str) -> str:
    """
    Handles punctuation marks by replacing them with spaces,
    keeping Gujarati letters, matras, digits, and word characters intact.
    Includes Gujarati danda (।), double danda (॥), quotes, and standard punctuation.
    """
    if not text:
        return ""
    # Punctuation set including ASCII and Indic punctuation
    # \u0964 = danda (।), \u0965 = double danda (॥)
    punct_pattern = r'[।॥.,!?:;"\'\(\)\[\]\{\}\\\/<>@#$%^&*_+=~`|–—\-]'
    cleaned = re.sub(punct_pattern, ' ', text)
    # Collapse any newly formed multiple spaces
    return re.sub(r'\s+', ' ', cleaned).strip()


def tokenize(text: str) -> List[str]:
    """
    Tokenizes preprocessed text into individual words.
    Splits by whitespace.
    """
    if not text:
        return []
    return [token for token in text.split(' ') if token]


def handle_stopwords(tokens: List[str]) -> Dict[str, Any]:
    """
    Handles stopwords.
    
    IMPORTANT ARCHITECTURAL DECISION:
    Stopword removal is intentionally DISABLED.
    In N-gram language modeling and spell-checking, common words (હું, છે, અને, માં, etc.)
    provide critical grammatical and transition context between adjacent words.
    Removing them would distort n-gram transition probabilities and break context.
    """
    return {
        "status": "DISABLED",
        "reason": "Stopwords are retained to preserve syntactic context for the N-gram language model.",
        "tokens": list(tokens)
    }


def preprocess(text: str) -> List[str]:
    """
    Performs full preprocessing on raw text and returns clean word tokens.
    """
    norm = unicode_normalize(text)
    ws = whitespace_normalize(norm)
    punct = handle_punctuation(ws)
    tokens = tokenize(punct)
    return tokens


def preprocess_with_details(text: str) -> Dict[str, Any]:
    """
    Performs full preprocessing while recording each visible stage.
    Used for web UI pipeline inspection and viva demonstration.
    """
    raw_input = text if text is not None else ""
    step1_unicode = unicode_normalize(raw_input)
    step2_whitespace = whitespace_normalize(step1_unicode)
    step3_punctuation = handle_punctuation(step2_whitespace)
    step4_tokens = tokenize(step3_punctuation)
    step5_stopwords = handle_stopwords(step4_tokens)

    return {
        "original_text": raw_input,
        "unicode_normalized": step1_unicode,
        "whitespace_normalized": step2_whitespace,
        "punctuation_handled": step3_punctuation,
        "tokens": step4_tokens,
        "stopword_handling": {
            "status": step5_stopwords["status"],
            "description": "Stopword removal: DISABLED. All words and grammatical particles are preserved.",
            "reason": step5_stopwords["reason"]
        },
        "final_tokens": step5_stopwords["tokens"]
    }
