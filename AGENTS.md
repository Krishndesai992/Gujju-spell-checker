# Agent Guidelines & Project Rules

This document governs any AI coding agent, pair programmer, or developer modifying the `NLP_eazy` codebase.

---

## Core Operational Rules

1. **Standalone Project**: This is an independent, standalone project inside `NLP_eazy/`. Never import from, link to, or reference any external/previous project folders.
2. **No Dependency on Previous Implementations**: Do not depend on, preserve, or attempt to recreate any past Gujarati Spell Checker architecture.
3. **Simplicity First**: Keep the architecture simple, clean, and educational. If choosing between complex heuristics and an easily explainable NLP algorithm, always choose the simple, explainable one.
4. **No Deep Learning**: Under no circumstances should deep learning, neural networks, CNNs, RNNs, LSTMs, GRUs, or Transformers be introduced.
5. **No TensorFlow**: Do not install or import TensorFlow.
6. **No Keras**: Do not install or import Keras.
7. **No PyTorch**: Do not install or import PyTorch.
8. **No Streamlit**: Do not use Streamlit. The application uses a standard Flask backend with Vanilla HTML/CSS/JS.
9. **No GujMORPH**: Do not use GujMORPH or complex morphological taggers.
10. **No Scikit-learn**: Do not install or import Scikit-learn or similar high-level machine learning frameworks.
11. **Do Not Remove Stopwords**: Stopword removal is strictly prohibited. Stopwords (હું, છે, અને, માં, etc.) are essential syntactic anchors for N-gram transition modeling.
12. **Exactly Two Datasets**: Do not add a third dataset. The project operates exclusively with `data/gujarati_training.csv` and `data/gujarati_test.csv`.
13. **Strict Separation (No Data Leakage)**: Never train, tune, or extract vocabulary from `data/gujarati_test.csv`. Training is strictly confined to `data/gujarati_training.csv`.
14. **Preserve Gujarati Unicode**: Handle UTF-8 encoding diligently across all file operations and console streams (`sys.stdout.reconfigure(encoding='utf-8')`).
15. **Keep Preprocessing Visible**: Preprocessing stages (Unicode NFC, whitespace, punctuation, tokenization, stopword preservation) must remain explicit and inspectable.
16. **Keep Algorithms Explainable**: All logic (Levenshtein distance, N-gram calculations, Laplace smoothing, candidate ranking) must be understandable for a university semester viva.
17. **Prefer Python Standard Library**: Favor standard modules (`csv`, `json`, `re`, `unicodedata`, `collections`, `pathlib`, `sys`, `os`).
18. **No Hard-Coded Outputs**: Never hard-code spell-checking or evaluation results. All outputs must be computed dynamically.
19. **Minimal Dependencies**: The only external production dependency is `Flask` (`requirements.txt`).
20. **Update README.md**: Whenever functionality, commands, or structure change, update `README.md`.
21. **Update Architecture Documentation**: If architectural specifications or designs evolve, update `docs/ARCHITECTURE.md`.
22. **Maintain Automated Tests**: Keep unit tests under `tests/` passing without regressions (`python -m unittest discover tests`).
23. **Keep Frontend Clean & Intuitive**: Maintain a clean, accessible user interface free of unnecessary dashboards or bloat.
24. **No Frontend Frameworks**: Do not inject React, Vue, Angular, or TailwindCSS. Stick to clean Vanilla HTML, CSS, and JavaScript.
25. **No Architectural Drift**: Do not silently alter the fundamental project architecture.
