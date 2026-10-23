"""Grading rules and quality evaluation logic."""

from typing import Any, Dict, List


def evaluate_quality(
    defect_percentage: float,
    fruit_type: str = "tomato",
    **kwargs: Any,
) -> Dict[str, Any]:
    """Evaluate fruit quality based on defect percentage and classification rules.

    Args:
        defect_percentage: Percentage of defects detected (0.0 to 100.0) or ratio (0.0 to 1.0).
        fruit_type: Commodity type (default: "tomato").
        **kwargs: Optional forward/backward compatibility keyword arguments.

    Returns:
        Dictionary containing grade, category, defect percentage, shelf life,
        salvage rate, commercial actions, status colors, agronomic prevention advisory,
        and alternative repurposing use cases.
    """
    try:
        defect_percentage = float(defect_percentage)
    except (ValueError, TypeError):
        defect_percentage = 0.0

    # Auto-normalize ratio (e.g., 0.14 -> 14.0, 0.37 -> 37.0)
    if 0.0 < defect_percentage <= 1.0:
        defect_percentage = defect_percentage * 100.0

    defect_percentage = round(defect_percentage, 2)

    if defect_percentage <= 6.0:
        return {
            "grade": "Grade A",
            "category": "Export / Supermarket Quality",
            "defect_percentage": defect_percentage,
            "shelf_life": "7–10 days",
            "shelf_life_days": "7–10 days",
            "salvage_rate": "100%",
            "salvage_yield": "100%",
            "salvage_value": "100%",
            "ui_color": "#22C55E",
            "status_color": "#22C55E",
            "commercial_action": "Approved for premium supermarket shelf and export cold chain.",
            "recommended_action": "Approved for premium supermarket shelf and export cold chain.",
            "agronomic_prevention": (
                "Maintain current balanced drip irrigation and calcium monitoring to sustain export-grade skin integrity."
            ),
            "alternative_uses": [
                "Export & Supermarket Fresh Retail",
                "Premium Packaging & Extended Cold Chain Supply",
            ],
        }
    elif defect_percentage <= 22.0:
        return {
            "grade": "Grade B",
            "category": "Secondary Market / Quick Sale / Local Mandi",
            "defect_percentage": defect_percentage,
            "shelf_life": "2–4 days",
            "shelf_life_days": "2–4 days",
            "salvage_rate": "75%",
            "salvage_yield": "75%",
            "salvage_value": "75%",
            "ui_color": "#F59E0B",
            "status_color": "#F59E0B",
            "commercial_action": "Divert to local mandis, hyper-local commerce, or immediate kitchen processing.",
            "recommended_action": "Divert to local mandis, hyper-local commerce, or immediate kitchen processing.",
            "agronomic_prevention": (
                "Early stage pathogen/fungal damage detected. Prune lower canopy foliage, shift irrigation to morning hours, and spray preventative bio-fungicides."
            ),
            "alternative_uses": [
                "Immediate Local Market / Mandi Quick Sale",
                "Supply to Cloud Kitchens, Canteens & Local Restaurants (Same-Day Cooking)",
                "Farmgate Batch Puree / Tomato Paste Extraction",
            ],
        }
    else:
        return {
            "grade": "Grade C",
            "category": "Industrial Repurposing / Biomass",
            "defect_percentage": defect_percentage,
            "shelf_life": "< 24 hours",
            "shelf_life_days": "< 24 hours",
            "salvage_rate": "25%",
            "salvage_yield": "25%",
            "salvage_value": "25%",
            "ui_color": "#EF4444",
            "status_color": "#EF4444",
            "commercial_action": "Unfit for fresh retail. Divert immediately to paste/sauce pulping, composting, or biogas.",
            "recommended_action": "Unfit for fresh retail. Divert immediately to paste/sauce pulping, composting, or biogas.",
            "agronomic_prevention": (
                "Severe rot breakdown. Isolate and destroy diseased field debris, enforce a 3-year Solanaceae crop rotation, and check subsoil drainage."
            ),
            "alternative_uses": [
                "Industrial Sauce / Ketchup Processing Units (post-trimming)",
                "Nutrient-Rich Organic Vermicomposting",
                "Anaerobic Digestion / Biogas Energy Generation Feedstock",
                "Cattle / Livestock Feed Supplement (subject to local biosafety)",
            ],
        }
