"""
preprocessing_demo.py - Interactive Preprocessing Demonstration

This standalone script allows students and evaluators to observe each stage
of the NLP text preprocessing pipeline on arbitrary Gujarati text.
"""

import sys
from preprocessing import preprocess_with_details


def main():
    # Ensure Windows console handles UTF-8 Gujarati characters properly
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            try:
                stream.reconfigure(encoding='utf-8', errors='replace')
            except Exception:
                pass

    print("=" * 44)
    print("        GUJARATI PREPROCESSING DEMO        ")
    print("=" * 44)
    print("Example input: હું શાલામાં જાઉં છું.\n")

    try:
        user_input = input("Enter Gujarati text: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nExiting.")
        return

    if not user_input:
        user_input = "હું શાલામાં જાઉં છું."
        print(f"No input detected. Using default example: {user_input}")

    details = preprocess_with_details(user_input)

    print("\n" + "=" * 44)
    print("PREPROCESSING PIPELINE OUTPUT")
    print("=" * 44)

    print("\nOriginal Text:")
    print(details["original_text"])

    print("\nStep 1 - Unicode Normalization:")
    print(details["unicode_normalized"])

    print("\nStep 2 - Whitespace Normalization:")
    print(details["whitespace_normalized"])

    print("\nStep 3 - Punctuation Handling:")
    print(details["punctuation_handled"])

    print("\nStep 4 - Tokenization:")
    print(details["tokens"])

    print("\nStep 5 - Stopword Handling:")
    print(f"Stopword Removal: {details['stopword_handling']['status']}")
    print("Stopwords are retained.")
    print(f"Reason: {details['stopword_handling']['reason']}")

    print("\nFinal Preprocessed Output:")
    print(details["final_tokens"])

    print("=" * 44)


if __name__ == "__main__":
    main()
