# Gujarati Spell Checker (ગુજરાતી જોડણી તપાસક)

An educational, explainable, lightweight, and production-ready Gujarati Spell Checker developed as a Semester 7 Natural Language Processing (NLP) mini-project.

Built entirely on classical NLP foundations:
**Text Preprocessing + Unicode NFC Normalization + Tokenization + Statistical N-Gram Language Modeling + Dynamic Programming Levenshtein Distance**.

> **No deep learning black-boxes, no neural networks, zero external spell-checking APIs, and 100% offline.**

---

## Quick Start (Run in 2 Commands)

```bash
# 1. Train the statistical N-Gram model
python train_ngram.py

# 2. Launch the Flask web server
python app.py
```
Open **[http://127.0.0.1:5000](http://127.0.0.1:5000)** in your browser.

---

## Table of Contents
1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Technology Stack](#technology-stack)
4. [Datasets & Data Leakage Prevention](#datasets--data-leakage-prevention)
5. [Installation & Setup](#installation--setup)
6. [Training the N-Gram Model](#training-the-n-gram-model)
7. [Running the Web Application](#running-the-web-application)
8. [API Endpoints](#api-endpoints)
9. [Standalone Preprocessing Demo](#standalone-preprocessing-demo)
10. [Quantitative Evaluation Benchmark](#quantitative-evaluation-benchmark)
11. [Automated Unit Testing](#automated-unit-testing)
12. [Repository Directory Structure](#repository-directory-structure)
13. [Algorithm Walkthrough](#algorithm-walkthrough)
14. [End-to-End Example](#end-to-end-example)
15. [Academic Limitations](#academic-limitations)
16. [Viva Examination Guide](#viva-examination-guide)
17. [Documentation](#documentation)

---

## Overview

Natural language processing for Indian regional languages like Gujarati poses unique orthographic challenges due to complex characters, vowel matras, and conjunct consonants (*jodakshar*).

This project provides an end-to-end, fully visible spell-checking pipeline that allows students, evaluators, and developers to trace every stage of spell correction:
1. Cleaning and normalizing Gujarati Unicode text via **Unicode NFC**.
2. Normalizing whitespace and stripping Western and Indic punctuation (`।`, `॥`).
3. Tokenizing sentences into words while **deliberately keeping stopwords** for grammatical context.
4. Checking words against a statistical corpus vocabulary.
5. Generating candidates with small edit distances using **Dynamic Programming Levenshtein Distance**.
6. Ranking suggestions using Unigram, Bigram, and Trigram probabilities with **Laplace (add-one) smoothing**.
7. Displaying the decision process in a modern, responsive web interface with Google Keyboard style chip suggestions.

---

## Key Features

- **Single Word & Sentence Processing**: Corrects individual words (e.g. `શાલામાં` → `શાળામાં`) as well as multi-word sentences (e.g. `હું શાલામાં જાઉં છું.` → `હું શાળામાં જાઉં છું.`).
- **100% Explainable NLP**: Every formula (Levenshtein matrix, N-gram transition probabilities, Laplace smoothing) is transparently calculated in readable Python code.
- **Visible Preprocessing**: Intermediate results for Unicode normalization, whitespace collapsing, punctuation stripping, and tokenization are viewable in real-time.
- **Google Keyboard Style Suggestions**: Returns at least 6 ranked candidate chips with click-to-apply functionality.
- **Contextual Awareness**: Utilizes Bigrams and Trigrams from the training corpus to select candidates that make sense in context.
- **Zero Data Leakage**: Training strictly uses `data/gujarati_training.csv`. Evaluation strictly uses `data/gujarati_test.csv`.
- **Zero Heavy Frameworks**: Requires only `Flask`. No TensorFlow, PyTorch, Scikit-learn, or Streamlit.

---

## Technology Stack

- **Language**: Python 3.11+
- **Web Backend**: Flask
- **Frontend**: Vanilla HTML5, CSS3, JavaScript (ES6+)
- **Storage**: CSV (datasets), JSON (trained statistical model)
- **Standard Library Modules**: `re`, `unicodedata`, `collections`, `csv`, `json`, `math`, `pathlib`, `sys`, `os`, `unittest`

---

## Datasets & Data Leakage Prevention

The project operates with **EXACTLY TWO** datasets in `data/`:

| Dataset | Path | Records | Description |
|---|---|---|---|
| **Training Corpus** | `data/gujarati_training.csv` | 829 sentences | Used by `train_ngram.py` to extract 550 vocabulary words and compute Unigram, Bigram, and Trigram frequencies. |
| **Held-Out Test Set** | `data/gujarati_test.csv` | 109 test cases | Used exclusively by `evaluate.py`. Contains 89 real spelling errors (matra substitution, character substitution, insertion, deletion) and 20 error-free cases. |

### Strict Zero Data Leakage Rule:
The model is **never** trained or tuned on `data/gujarati_test.csv`. The evaluation benchmark strictly measures genuine out-of-sample error detection and correction performance.

---

## Installation & Setup

### 1. Clone or Navigate to the Repository
```bash
git clone https://github.com/your-username/NLP_eazy.git
cd NLP_eazy
```

### 2. (Recommended) Create a Virtual Environment
```bash
# On Windows
python -m venv venv
.\venv\Scripts\activate

# On Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Minimal Dependencies
```bash
pip install -r requirements.txt
```
*(Only installs Flask; all NLP and testing logic uses the Python Standard Library)*

### 4. (Optional) Configure Environment Variables
Copy `.env.example` to `.env` if custom port or debug settings are needed:
```bash
# Windows PowerShell
Copy-Item .env.example .env

# Linux / macOS
cp .env.example .env
```

---

## Training the N-Gram Model

Before launching the web app for the first time, train the statistical model:

```bash
python train_ngram.py
```

### Training Pipeline:
1. Loads training sentences from `data/gujarati_training.csv`.
2. Normalizes, cleans, and tokenizes every sentence.
3. Automatically builds the unique vocabulary (550 unique words).
4. Counts unigram tokens (5,476 total), distinct bigrams (1,531), and distinct trigrams (1,899).
5. Serializes the statistical model to `model/ngram_model.json`.

---

## Running the Web Application

Launch the Flask development server:

```bash
python app.py
```

Access the interface in your browser:
```
http://127.0.0.1:5000
```

### Web UI Highlights:
- **Clean Input Card**: Enter any Gujarati word or sentence, or click quick viva sample cards.
- **Verdict Banner**: Immediate visual feedback indicating whether spelling errors were detected.
- **Suggestion Chips**: Google Keyboard style suggestion chips for misspelled words; click any chip to apply the correction instantly.
- **Visual Token Breakdown**: Colored pill badges indicating each word's correctness status.
- **Pipeline Inspector**: Accordion view displaying Unicode normalization, whitespace collapsing, punctuation handling, tokenization, stopword preservation rationale, and Levenshtein DP calculation details.

---

## API Endpoints

The Flask application exposes JSON endpoints for integration and automation:

| Endpoint | Method | Description | Sample Request / Response |
|---|---|---|---|
| `/` | `GET` | Main web interface | Returns HTML UI |
| `/health` | `GET` | Health check endpoint | `{"status": "healthy", "model_loaded": true, "vocab_size": 550}` |
| `/api/status` | `GET` | Model operational metrics | `{"status": "ready", "vocab_size": 550, "unigrams": 550, "bigrams": 1531, "trigrams": 1899}` |
| `/check` | `POST` | Check spelling of word or sentence | Body: `{"text": "શાલામાં"}`<br>Returns corrected text, candidate rankings, and pipeline traces |

---

## Standalone Preprocessing Demo

For interactive demonstration during a viva or presentation:

```bash
python preprocessing_demo.py
```
Prompts for any Gujarati text and prints the step-by-step transformation:
- Original Text
- Step 1: Unicode NFC Normalization
- Step 2: Whitespace Normalization
- Step 3: Punctuation Handling (Indic & ASCII)
- Step 4: Tokenization
- Step 5: Stopword Preservation Rationale
- Final Preprocessed Tokens

---

## Quantitative Evaluation Benchmark

Run the rigorous evaluation script against the 109 held-out test cases:

```bash
python evaluate.py
```

### Benchmark Results:
```text
============================================
   GUJARATI SPELL CHECKER EVALUATION   
============================================

Total Test Cases:            109
Spelling Error Cases:        89
Correct Word Cases:          20

Correctly Detected:          109 (89 error cases + 20 correct cases)
Error Detection Sensitivity: 100.00%

Exact Corrections:           88 / 89
Correction Accuracy:         98.88%

False Positives:             0 / 20 (0.00%)

--------------------------------------------
Breakdown by Error Type:
--------------------------------------------
Error Type           Count    Detected   Exact Match
deletion             20       20         19
insertion            28       28         28
matra_substitution   21       21         21
substitution         20       20         20
============================================
```

---

## Automated Unit Testing

The repository includes a comprehensive unit test suite using Python's built-in `unittest` runner:

```bash
python -m unittest discover tests
```

Tests cover:
- `tests/test_preprocessing.py`: Unicode NFC, whitespace normalization, punctuation handling, tokenization, and stopword preservation.
- `tests/test_levenshtein.py`: Dynamic programming matrix base cases, substitutions, insertions, deletions, and symmetry.
- `tests/test_ngram_model.py`: Frequency counting, Laplace smoothing, vocab extraction, model serialization and deserialization.
- `tests/test_spell_checker.py`: Candidate generation (at least 6 chips), multi-factor ranking, error correction, and preserving correct words.
- `tests/test_app.py`: Flask endpoints (`/`, `/health`, `/api/status`, `/check`).

---

## Repository Directory Structure

```text
NLP_eazy/
├── data/
│   ├── gujarati_training.csv      # 829 training sentences (vocabulary & N-grams)
│   └── gujarati_test.csv          # 109 held-out test cases (evaluation benchmark)
│
├── docs/
│   └── ARCHITECTURE.md            # Technical architecture, mathematical formulas & PRD
│
├── model/
│   └── ngram_model.json           # Serialized statistical N-Gram model
│
├── static/
│   ├── script.js                  # Asynchronous UI controller & chip handler
│   └── style.css                  # Responsive CSS design system
│
├── templates/
│   └── index.html                 # Main web application view template
│
├── tests/
│   ├── __init__.py                # Test package initialization
│   ├── test_app.py                # Flask route and API endpoint tests
│   ├── test_levenshtein.py        # Levenshtein distance unit tests
│   ├── test_ngram_model.py        # Statistical model and Laplace smoothing tests
│   ├── test_preprocessing.py      # Text preprocessing and tokenization tests
│   └── test_spell_checker.py      # Spell-checking and candidate ranking tests
│
├── .env.example                   # Safe template for environment variables
├── .gitignore                     # Git ignore rules for Python, IDEs, and OS files
├── AGENTS.md                      # Operational guidelines and coding rules
├── app.py                         # Flask web application entry point
├── evaluate.py                    # Evaluation benchmark runner
├── levenshtein.py                 # Dynamic-programming Levenshtein distance
├── ngram_model.py                 # Statistical N-Gram Language Model
├── preprocessing.py               # Reusable text preprocessing pipeline
├── preprocessing_demo.py          # Standalone terminal preprocessing demo
├── README.md                      # Primary project documentation (this file)
├── requirements.txt               # Production dependencies (Flask)
├── spell_checker.py               # Spell checker engine & candidate ranker
└── train_ngram.py                 # Model training and serialization script
```

---

## Algorithm Walkthrough

1. **Preprocessing**:
   - `unicodedata.normalize('NFC', text)` canonicalizes Gujarati vowel matras and conjuncts.
   - `re.sub(r'\s+', ' ', text).strip()` removes excess whitespace.
   - Punctuation cleaner replaces western symbols and Indic dandas (`।`, `॥`) with spaces.
2. **Tokenization**:
   - Segments cleaned text into whitespace-delimited word tokens.
3. **Stopword Policy (Preserved)**:
   - Functional words (`હું`, `છે`, `અને`, `માં`) are explicitly retained because removing them destroys bigram and trigram contextual links.
4. **Vocabulary Check**:
   - Words are looked up in $O(1)$ time in the model vocabulary hash set.
5. **Candidate Generation**:
   - For unknown words, candidates within Levenshtein distance $\le 1$ and $\le 2$ (and $\le 3$ if needed) are extracted to provide at least 6 suggestions.
6. **Levenshtein Distance**:
   - Evaluated using a dynamic programming matrix $DP[i][j]$ with unit costs for insertions, deletions, and substitutions.
7. **N-Gram Contextual Scoring**:
   - Candidates are scored using Trigram likelihood $P(c \mid w_{i-2}, w_{i-1})$, Bigram likelihood $P(c \mid w_{i-1})$, and Unigram prior $P(c)$, smoothed using Laplace add-one smoothing.
8. **Ranking**:
   - Candidate chips are ordered by minimum edit distance, maximum prefix/suffix match, character set overlap, and contextual transition probability.

---

## End-to-End Example

### Input Sentence:
```gujarati
હું શાલામાં જાઉં છું.
```

### Execution Trace:
1. **Preprocessing**: Normalized, punctuation cleaned, tokens: `['હું', 'શાલામાં', 'જાઉં', 'છું']`.
2. **Vocabulary Lookup**:
   - `હું`: Found (Valid)
   - `શાલામાં`: Not found in vocabulary (Flagged as misspelled)
   - `જાઉં`: Found (Valid)
   - `છું`: Found (Valid)
3. **Candidate Generation for `શાલામાં`**:
   - `શાળામાં`: Levenshtein distance = 1 (substituting `લ` with `ળ`).
   - Other candidates: Distance $\ge 2$.
4. **Contextual Evaluation**:
   - Trigram `('હું', 'શાળામાં', 'જાઉં')` and Bigram `('હું', 'શાળામાં')` have high probability in the corpus.
5. **Result**:
   - Suggested correction: `શાળામાં`
   - Reconstructed text: `હું શાળામાં જાઉં છું.`

---

## Academic Limitations

1. **Vocabulary Scope**: The vocabulary comprises 550 unique words extracted from `data/gujarati_training.csv`. Unseen valid Gujarati words outside this corpus are flagged as out-of-vocabulary.
2. **N-Gram Context Window**: Trigrams capture local context of two preceding words; long-range semantic dependencies beyond 3 words are not modeled.
3. **Real-Word Errors**: If a typographical error accidentally results in another valid Gujarati word in the vocabulary, it is recognized as valid unless conflicting with contextual n-gram thresholds.

---

## Viva Examination Guide

| Question | Authoritative Answer |
| :--- | :--- |
| **Why not remove stopwords?** | In spell checking and N-gram language modeling, functional words (e.g., "હું", "છે", "અને", "માં") define syntactic transitions. Removing them breaks bigram and trigram co-occurrences (e.g., "અભ્યાસ કરે છે" becomes disjoint words). |
| **How does Levenshtein Distance work?** | It uses a Dynamic Programming table of size $(m+1) \times (n+1)$ where each cell $DP[i][j]$ computes the minimum operations (deletion $+1$, insertion $+1$, substitution $+1$ if mismatch, $+0$ if match) to transform prefix $A[1..i]$ to $B[1..j]$. |
| **What is Laplace Smoothing and why is it needed?** | Maximum Likelihood Estimation assigns $0$ probability to any unseen word sequence. Laplace (add-one) smoothing adds $1$ to the frequency count and $V$ (vocabulary size) to the denominator, ensuring non-zero probabilities for unseen transitions. |
| **What is the difference between Bigram and Trigram probabilities?** | Bigram calculates $P(w_i \mid w_{i-1}) = \frac{\text{Count}(w_{i-1}, w_i)}{\text{Count}(w_{i-1})}$. Trigram calculates $P(w_i \mid w_{i-2}, w_{i-1}) = \frac{\text{Count}(w_{i-2}, w_{i-1}, w_i)}{\text{Count}(w_{i-2}, w_{i-1})}$, capturing richer two-word grammatical context. |
| **Why not use Deep Learning (BERT / LSTMs)?** | Classical statistical NLP provides 100% explainability, requires zero GPU acceleration, trains in under 1 second, and allows line-by-line inspection of mathematical decisions—ideal for academic evaluation and lightweight offline deployment. |

---

## Documentation

- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**: In-depth technical specifications, mathematical equations, zero data leakage policy, and detailed PRD constraints.
- **[AGENTS.md](AGENTS.md)**: Coding rules and operational guidelines for developers and AI agents working on this codebase.
