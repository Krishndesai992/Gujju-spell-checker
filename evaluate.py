"""
evaluate.py - Quantitative Evaluation Script for Gujarati Spell Checker

Evaluates the spell checker strictly on the held-out data/gujarati_test.csv.
Adheres strictly to the NO DATA LEAKAGE rule:
- Uses only the test dataset.
- Never modifies or retrains the model.
- Calculates and reports real, un-fabricated metrics.
"""

import sys
import csv
from pathlib import Path
from collections import defaultdict
from spell_checker import GujaratiSpellChecker


def run_evaluation():
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            try:
                stream.reconfigure(encoding='utf-8', errors='replace')
            except Exception:
                pass

    base_dir = Path(__file__).resolve().parent
    test_csv_path = base_dir / "data" / "gujarati_test.csv"
    if not test_csv_path.exists():
        print(f"Error: Test dataset not found at {test_csv_path}")
        sys.exit(1)

    checker = GujaratiSpellChecker()
    if not checker.is_loaded:
        print("Error: Model is not trained. Please run 'python train_ngram.py' first.")
        sys.exit(1)

    print("=" * 44)
    print("   GUJARATI SPELL CHECKER EVALUATION   ")
    print("=" * 44)

    test_cases = []
    with open(test_csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_cases.append(row)

    total_test_cases = len(test_cases)
    spelling_error_cases = 0
    correct_word_cases = 0

    detected_errors = 0          # Error cases correctly identified as misspelled
    detected_correct = 0         # Correct cases correctly identified as correct
    exact_corrections = 0        # Cases where suggestion == expected_correction
    false_positives = 0          # Correct words mistakenly flagged as errors

    by_error_type = defaultdict(lambda: {"total": 0, "detected": 0, "exact_corrected": 0})

    for row in test_cases:
        input_word = row["input_word"].strip()
        correct_word = row["correct_word"].strip()
        error_type = row["error_type"].strip()
        expected = row.get("expected_correction", correct_word).strip()

        is_actual_error = (error_type != "no_error")

        if is_actual_error:
            spelling_error_cases += 1
            by_error_type[error_type]["total"] += 1
        else:
            correct_word_cases += 1

        # Run spell checker
        res = checker.check_word(input_word)
        suggested = res["suggested_correction"]
        is_flagged_error = not res["is_correct"]

        if is_actual_error:
            if is_flagged_error:
                detected_errors += 1
                by_error_type[error_type]["detected"] += 1

            if suggested == expected:
                exact_corrections += 1
                by_error_type[error_type]["exact_corrected"] += 1
        else:
            if not is_flagged_error:
                detected_correct += 1
            else:
                false_positives += 1

    # Total correctly detected (detected errors + detected correct)
    correctly_detected = detected_errors + detected_correct
    detection_rate = (detected_errors / spelling_error_cases * 100) if spelling_error_cases else 0.0
    correction_accuracy = (exact_corrections / spelling_error_cases * 100) if spelling_error_cases else 0.0

    print(f"\nTotal Test Cases:")
    print(total_test_cases)

    print(f"\nSpelling Error Cases:")
    print(spelling_error_cases)

    print(f"\nCorrect Word Cases:")
    print(correct_word_cases)

    print(f"\nCorrectly Detected:")
    print(f"{correctly_detected} ({detected_errors} error cases + {detected_correct} correct cases)")
    print(f"Error Detection Sensitivity: {detection_rate:.2f}%")

    print(f"\nExact Corrections:")
    print(exact_corrections)

    print(f"\nCorrection Accuracy:")
    print(f"{correction_accuracy:.2f}%")

    print(f"\nFalse Positives:")
    print(f"{false_positives} (out of {correct_word_cases} correct word cases)")

    print("\n" + "-" * 44)
    print("Breakdown by Error Type:")
    print("-" * 44)
    print(f"{'Error Type':<20} {'Count':<8} {'Detected':<10} {'Exact Match':<12}")
    for etype, stats in sorted(by_error_type.items()):
        print(f"{etype:<20} {stats['total']:<8} {stats['detected']:<10} {stats['exact_corrected']:<12}")

    print("\n" + "=" * 44)
    print("Academic Explanation for Viva:")
    print(f"1. Error Detection Rate is {detection_rate:.1f}% on held-out test words.")
    print(f"2. Exact Correction Accuracy is {exact_corrections}/{spelling_error_cases} ({correction_accuracy:.2f}%).")
    print(f"3. False Positives: {false_positives}/{correct_word_cases} ({((false_positives/correct_word_cases)*100 if correct_word_cases else 0):.1f}%).")
    print("=" * 44)


if __name__ == "__main__":
    run_evaluation()
