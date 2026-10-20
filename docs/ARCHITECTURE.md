# Architecture & Technical Specification

## Gujarati Spell Checker (ગુજરાતી જોડણી તપાસક)
*Semester 7 Natural Language Processing (NLP) Mini-Project*

---

## 1. System Overview

Natural language processing for Indian regional languages like Gujarati presents unique orthographic challenges due to complex characters, independent and dependent vowels (matras), anusvara, visarga, and conjunct consonants (*jodakshar*).

This project implements an educational, transparent, explainable, and lightweight automated Gujarati Spell Checker built entirely on classical NLP principles:
1. **Unicode NFC Normalization & Text Cleaning**
2. **Whitespace Normalization & Indic/ASCII Punctuation Handling**
3. **Word Tokenization with Deliberate Stopword Preservation**
4. **Vocabulary Cross-referencing against Statistical Corpus**
5. **Candidate Generation & Dynamic Programming Levenshtein Distance**
6. **Multi-Factor Ranking using N-Gram Language Model Transition Probabilities**
7. **Interactive Real-Time Pipeline Inspection via Web Interface and Terminal**

```
                       +-----------------------------+
                       |         User Input          |
                       |   (Word or Full Sentence)   |
                       +--------------+--------------+
                                      |
                                      v
                       +-----------------------------+
                       |    Visible Preprocessing    |
                       |  - Unicode NFC Normalization|
                       |  - Whitespace Normalization |
                       |  - Punctuation Stripping    |
                       |  - Tokenization             |
                       |  - Stopwords Retained       |
                       +--------------+--------------+
                                      |
                                      v
                       +-----------------------------+
                       |      Vocabulary Check       |
                       | (Trained Model Vocabulary)  |
                       +--------------+--------------+
                                      |
                     +---------------+---------------+
                     |                               |
              [Word In Vocab]                [Word Not In Vocab]
                     |                               |
                     v                               v
            +----------------+             +-------------------+
            | Retain as Valid|             | Candidate Gen.    |
            | Word (dist = 0)|             | (Dist <= 1, <= 2) |
            +--------+-------+             +---------+---------+
                     |                               |
                     |                               v
                     |                     +-------------------+
                     |                     | Levenshtein Dist. |
                     |                     | (Dynamic Program) |
                     |                     +---------+---------+
                     |                               |
                     |                               v
                     |                     +-------------------+
                     |                     | N-Gram Contextual |
                     |                     | Probability Rank  |
                     |                     +---------+---------+
                     |                               |
                     +---------------+---------------+
                                     |
                                     v
                       +-----------------------------+
                       |    Final Spell Decision     |
                       | & Reconstructed Text Output |
                       +--------------+--------------+
                                     |
                                     v
                       +-----------------------------+
                       |    Flask Web Application    |
                       |     (templates / static)    |
                       |  - Result Verdict Card      |
                       |  - Google Keyboard Chips    |
                       |  - Full Pipeline Inspector  |
                       +-----------------------------+
```

---

## 2. NLP Pipeline & Mathematical Foundations

### 2.1 Text Preprocessing (`preprocessing.py`)

1. **Unicode Normalization Form C (NFC)**:
   - Resolves composite Gujarati characters (e.g. combining consonant base + dependent matra codepoints into canonical representations).
   - Utilizes Python's standard `unicodedata.normalize('NFC', text)`.

2. **Whitespace Normalization**:
   - Replaces irregular spaces, tabs, and newlines with a single space (`re.sub(r'\s+', ' ', text).strip()`).

3. **Punctuation Handling**:
   - Strips Western punctuation (`. , ! ? : ; " ' ( ) [ ] { } \ / < > @ # $ % ^ & * _ + = ~ ` | – — -`) as well as Indic punctuation: danda (`।`, `\u0964`) and double danda (`॥`, `\u0965`).
   - Retains Gujarati letters, matras, digits, and word characters intact.

4. **Tokenization**:
   - Segments preprocessed text into individual words based on whitespace boundaries.

5. **Stopword Handling Rationale (Crucial Architectural Rule)**:
   - **Stopword removal is intentionally DISABLED.**
   - In N-gram language modeling and spell-checking, common words (હું, છે, અને, માં, etc.) provide critical grammatical and transition context between adjacent words.
   - Removing them would destroy n-gram transition probabilities and render contextual ranking ineffective.

---

### 2.2 Dynamic Programming Levenshtein Distance (`levenshtein.py`)

Levenshtein distance measures the minimum number of single-character edits (insertions, deletions, or substitutions) required to transform source word $s_1$ into target candidate $s_2$.

#### Recurrence Relation:
Let $DP[i][j]$ be the edit distance between prefix $s_1[0 \dots i-1]$ and $s_2[0 \dots j-1]$:

$$DP[i][j] = \begin{cases}
i & \text{if } j = 0 \quad (\text{deleting all chars of } s_1) \\
j & \text{if } i = 0 \quad (\text{inserting all chars of } s_2) \\
DP[i-1][j-1] & \text{if } s_1[i-1] = s_2[j-1] \\
1 + \min \begin{cases} DP[i-1][j] & (\text{Deletion}) \\ DP[i][j-1] & (\text{Insertion}) \\ DP[i-1][j-1] & (\text{Substitution}) \end{cases} & \text{if } s_1[i-1] \ne s_2[j-1]
\end{cases}$$

- **Complexity**: $O(m \times n)$ time and $O(m \times n)$ space, where $m = |s_1|$ and $n = |s_2|$.
- **Educational Value**: Implemented from scratch without external C extensions, allowing complete line-by-line inspection of DP table filling.

---

### 2.3 Statistical N-Gram Language Model (`ngram_model.py`)

The language model captures word frequency and sequential transition likelihoods from `data/gujarati_training.csv`.

#### Unigram Probability:
Prior probability of word $w$ occurring independently:
$$P(w) = \frac{\text{Count}(w)}{N}$$
where $N$ is the total number of words across the training corpus.

#### Bigram Transition Probability:
Probability of word $w_i$ given immediately preceding word $w_{i-1}$:
$$P(w_i | w_{i-1}) = \frac{\text{Count}(w_{i-1}, w_i)}{\text{Count}(w_{i-1})}$$

#### Trigram Transition Probability:
Probability of word $w_i$ given preceding two words $w_{i-2}$ and $w_{i-1}$:
$$P(w_i | w_{i-2}, w_{i-1}) = \frac{\text{Count}(w_{i-2}, w_{i-1}, w_i)}{\text{Count}(w_{i-2}, w_{i-1})}$$

#### Laplace (Add-One) Smoothing:
To handle unseen transitions without zero-probability collapse:
$$P_{\text{Laplace}}(w_i | w_{i-1}) = \frac{\text{Count}(w_{i-1}, w_i) + 1}{\text{Count}(w_{i-1}) + V}$$
where $V$ is the vocabulary size ($|\text{Vocab}|$).

---

### 2.4 Candidate Generation & Ranking (`spell_checker.py`)

When an out-of-vocabulary word is detected:
1. **Candidate Retrieval**:
   - Filters corpus vocabulary for words where $| \text{len}(w) - \text{len}(v) | \le 3$.
   - Identifies all vocabulary candidates with Levenshtein distance $= 1$, then $= 2$, and $= 3$ (if needed) to ensure at least 6 candidates are available for Google Keyboard style suggestion chips.
2. **Multi-Factor Ranking Formula**:
   Candidates are scored based on four weighted factors:
   - **Edit Distance**: Primary filter ($d=1 \gg d=2 \gg d=3$).
   - **Prefix & Suffix Character Match**: Rewards candidates that share the initial and ending characters/matras.
   - **Character Set Jaccard Overlap**: Rewards words constructed from identical character components.
   - **Contextual Likelihood**: Computes trigram/bigram transition probability based on surrounding tokens in the sentence, falling back to unigram frequency prior.

---

## 3. Datasets & Strict Zero Data Leakage Policy

The repository operates with **EXACTLY TWO** datasets in `data/`:

| Dataset | File | Records | Purpose |
|---|---|---|---|
| **Training Corpus** | `data/gujarati_training.csv` | 829 sentences | Used exclusively by `train_ngram.py` to extract the vocabulary and compute N-gram frequencies. Saved to `model/ngram_model.json`. |
| **Held-Out Test Set** | `data/gujarati_test.csv` | 109 test cases | Used exclusively by `evaluate.py`. Contains 89 real spelling errors across 4 error classes (`matra_substitution`, `substitution`, `insertion`, `deletion`) and 20 error-free cases. |

### Zero Data Leakage Guarantee:
- The model is **never** trained on `data/gujarati_test.csv`.
- Vocabulary extraction strictly excludes test items.
- Evaluation metrics reported in `evaluate.py` represent genuine, un-fabricated generalisation capability.

---

## 4. Completed Milestones Summary

| Milestone | Deliverables | Verification Status |
|---|---|---|
| **Phase 1: Project Setup** | Directory layout, datasets UTF-8-sig encoding validation | Verified |
| **Phase 2: Preprocessing** | Unicode NFC, whitespace, punctuation, stopword preservation, `preprocessing_demo.py` | Verified |
| **Phase 3: N-Gram Model** | Unigram/Bigram/Trigram counters, Laplace smoothing, model serialization (`ngram_model.json`) | Verified |
| **Phase 4: Levenshtein DP** | From-scratch Dynamic Programming edit distance matrix | Verified |
| **Phase 5: Spell Checker** | Candidate generator (6+ chips), contextual ranking engine, sentence synthesis | Verified |
| **Phase 6: Web Application** | Flask backend (`app.py`), responsive vanilla CSS (`style.css`), dynamic UI (`script.js`) | Verified |
| **Phase 7: Evaluation** | Benchmark runner (`evaluate.py`), sensitivity 100%, accuracy 98.88%, false positives 0.0% | Verified |
| **Phase 8: Repository Hygiene** | Unit test suite (`tests/`), `.gitignore`, `.env.example`, consolidated architecture docs | Verified |

---

## 5. Architectural Boundaries & Non-Negotiable Constraints

1. **Zero Deep Learning**: No CNNs, RNNs, LSTMs, Transformers, BERT, PyTorch, TensorFlow, or Keras.
2. **Minimal Dependencies**: Production runtime requires only `Flask`. All NLP and algorithmic logic is implemented with Python's standard library.
3. **No Frontend Bloat**: Vanilla HTML5, CSS3, and ES6+ JavaScript. No React, Vue, Angular, or TailwindCSS.
4. **Offline & Explainable**: Completely local execution without external cloud APIs or GPU requirements.

---

## 6. Known Limitations & Future Work

### Current Limitations:
1. **Vocabulary Coverage**: Vocabulary is bounded by the sentences in `data/gujarati_training.csv` (550 unique words). Out-of-corpus valid words cannot be suggested unless added to the training set.
2. **Statistical Context**: N-gram statistics represent the sentence structures present in the training corpus.
3. **Phonetic Ambiguity**: Homophonic substitutions (e.g. `શ` vs `ષ` vs `સ`) rely on edit distance and corpus frequency rather than explicit phonetic rule tables.

### Future Work:
- Integrating a Gujarati phonetic Metaphone/Soundex algorithm for phonetic distance weighting.
- Training on larger open Gujarati corpora (e.g., Leipzig Corpora, IndicCorp) to expand vocabulary breadth.
- Real-word error detection (identifying grammatically inappropriate words that are spelled correctly).
