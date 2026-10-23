"""Grading rules and quality evaluation logic."""

from typing import Any, Dict


def evaluate_quality(defect_percentage: float, fruit_type: str = "tomato") -> Dict[str, Any]:
    """Evaluate fruit quality based on defect percentage and classification rules.

    Args:
        defect_percentage: Percentage of defects detected (0.0 to 100.0).
        fruit_type: Commodity type (default: "tomato").

    Returns:
        Dictionary containing grade, defect percentage, shelf life, salvage value,
        recommended action, and status color.
    """
    if defect_percentage < 5.0:
        return {
            "grade": "Grade A (Export & Supermarket Quality)",
            "defect_percentage": round(defect_percentage, 2),
            "shelf_life_days": "7-10 days",
            "salvage_value": "100%",
            "recommended_action": "Pack for cold chain and premium wholesale retail.",
            "status_color": "green",
        }
    elif defect_percentage <= 22.0:
        return {
            "grade": "Grade B (Secondary Market / Quick Sale)",
            "defect_percentage": round(defect_percentage, 2),
            "shelf_life_days": "2-4 days",
            "salvage_value": "75%",
            "recommended_action": "Discount 15-20% for immediate local retail or street distribution.",
            "status_color": "amber",
        }
    else:
        return {
            "grade": "Grade C (Industrial Processing / Biomass)",
            "defect_percentage": round(defect_percentage, 2),
            "shelf_life_days": "< 24 hours",
            "salvage_value": "25%",
            "recommended_action": "Divert immediately to sauce/puree processing, animal feed, or composting.",
            "status_color": "red",
        }
