"""
app.py - Flask Web Application for Gujarati Spell Checker

Provides:
- Web interface for single word and sentence spell checking.
- Google Keyboard style suggestions (6+ candidate chips).
- Interactive inspection of the full NLP pipeline.
- Automatic reload if model/ngram_model.json is updated.
"""

import os
import sys
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from spell_checker import GujaratiSpellChecker

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "ngram_model.json"
_spell_checker = None
_model_mtime = 0


def get_spell_checker():
    """Returns the cached spell checker, reloading if the model file changed."""
    global _spell_checker, _model_mtime
    if not MODEL_PATH.exists():
        return None

    try:
        current_mtime = os.path.getmtime(MODEL_PATH)
        if _spell_checker is None or current_mtime != _model_mtime:
            _spell_checker = GujaratiSpellChecker(str(MODEL_PATH))
            _model_mtime = current_mtime
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    return _spell_checker


@app.route("/")
def index():
    """Renders the clean single-page UI."""
    checker = get_spell_checker()
    return render_template(
        "index.html",
        model_ready=(checker is not None and checker.is_loaded),
        model_error=None if (checker and checker.is_loaded) else "Trained model not found. Please run train_ngram.py first.",
        vocab_size=checker.model.vocab_size if (checker and checker.is_loaded) else 0
    )


@app.route("/check", methods=["POST"])
def check():
    """
    API endpoint to check spelling of Gujarati word or sentence.
    Accepts JSON: {"text": "..."} or form data: text=...
    Returns comprehensive pipeline analysis and multi-candidate suggestions.
    """
    checker = get_spell_checker()
    if checker is None or not checker.is_loaded:
        return jsonify({
            "success": False,
            "error": "Trained model not found. Please run train_ngram.py first."
        }), 503

    if request.is_json:
        data = request.get_json() or {}
        text = data.get("text", "")
    else:
        text = request.form.get("text", "")

    text = text.strip()
    if not text:
        return jsonify({
            "success": False,
            "error": "Please enter a Gujarati word or sentence to check."
        }), 400

    result = checker.check_text(text)
    result["success"] = True
    return jsonify(result)


@app.route("/health")
def health():
    """Health check endpoint for deployment monitoring."""
    checker = get_spell_checker()
    if checker and checker.is_loaded:
        return jsonify({
            "status": "healthy",
            "model_loaded": True,
            "vocab_size": checker.model.vocab_size
        }), 200
    return jsonify({
        "status": "degraded",
        "model_loaded": False,
        "error": "Model not loaded"
    }), 503


@app.route("/api/status")
def status():
    """Returns the operational status and metadata of the spell checker."""
    checker = get_spell_checker()
    if checker and checker.is_loaded:
        return jsonify({
            "status": "ready",
            "vocab_size": checker.model.vocab_size,
            "total_corpus_words": checker.model.total_words,
            "unigrams": len(checker.model.unigram_counts),
            "bigrams": len(checker.model.bigram_counts),
            "trigrams": len(checker.model.trigram_counts)
        })
    return jsonify({
        "status": "not_ready",
        "error": "Model not loaded"
    }), 503


if __name__ == "__main__":
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            try:
                stream.reconfigure(encoding='utf-8', errors='replace')
            except Exception:
                pass

    checker = get_spell_checker()
    print("=" * 50)
    print("      GUJARATI SPELL CHECKER WEB SERVER      ")
    print("=" * 50)
    if checker and checker.is_loaded:
        print(f"Model loaded: {checker.model.vocab_size} vocabulary words.")
        print("Starting Flask web server on http://127.0.0.1:5000 ...")
    else:
        print("WARNING: Trained model not found. Run train_ngram.py first.")

    app.run(host="127.0.0.1", port=5000, debug=False)
