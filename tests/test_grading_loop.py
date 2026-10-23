import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.grading_rules import evaluate_quality


def run_verification_loop():
    test_cases = [
        {"defect": 0.0, "expected_grade": "Grade A"},
        {"defect": 3.2, "expected_grade": "Grade A"},
        {"defect": 5.9, "expected_grade": "Grade A"},
        {"defect": 6.1, "expected_grade": "Grade B"},
        {"defect": 14.5, "expected_grade": "Grade B"},
        {"defect": 20.0, "expected_grade": "Grade B"},
        {"defect": 21.9, "expected_grade": "Grade B"},
        {"defect": 22.1, "expected_grade": "Grade C"},
        {"defect": 37.3, "expected_grade": "Grade C"},
        {"defect": 96.2, "expected_grade": "Grade C"},
    ]

    all_passed = True
    print("\n=== RUNNING AGRIGRADE GRADING VERIFICATION LOOP ===")
    for case in test_cases:
        res = evaluate_quality(case["defect"])
        assigned = res["grade"]
        status = "PASS" if assigned == case["expected_grade"] else "FAIL"
        if status == "FAIL":
            all_passed = False
        print(f"[{status}] Defect: {case['defect']:>5.1f}% | Expected: {case['expected_grade']} | Received: {assigned}")

    print("===================================================\n")
    if not all_passed:
        raise ValueError("Grading verification loop failed assertions!")
    print("ALL TEST CASES PASSED SUCCESSFULLY.\n")


if __name__ == "__main__":
    run_verification_loop()
