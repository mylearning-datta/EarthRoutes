from __future__ import annotations

import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from .report_utils import (
        ensure_directories,
        get_project_root,
        save_figure,
        write_markdown,
    )
except Exception:
    # Fallback for direct script execution
    import sys
    from pathlib import Path as _Path

    sys.path.append(str(_Path(__file__).resolve().parent))
    from report_utils import (
        ensure_directories,
        get_project_root,
        save_figure,
        write_markdown,
    )


def parse_total_reviews(raw: pd.Series) -> pd.Series:
    cleaned = raw.fillna(0).astype(str).str.replace(
        r"[^0-9]", "", regex=True
    )
    as_int = pd.to_numeric(cleaned, errors="coerce").fillna(0).astype(int)
    return as_int


def extract_city(place: pd.Series) -> pd.Series:
    # If value contains a comma, assume last token is city (e.g., "Taj Ganj, Agra" -> "Agra")
    def _city(val: str) -> str:
        if not isinstance(val, str) or not val:
            return ""
        parts = [p.strip() for p in val.split(",") if p.strip()]
        return parts[-1] if parts else val.strip()

    return place.apply(_city)


def run() -> None:
    project_root = get_project_root(__file__)
    data_path = project_root / "data" / "hotel_details.csv"
    figures_dir = project_root / "eda" / "figures"
    reports_dir = project_root / "eda" / "reports"
    ensure_directories([figures_dir, reports_dir])

    df = pd.read_csv(data_path)

    # Coerce fields
    df["Rating_num"] = pd.to_numeric(df.get("Rating"), errors="coerce")
    df["Total_Reviews_num"] = parse_total_reviews(df.get("Total Reviews"))
    df["City"] = extract_city(df.get("Place"))
    df["Condition_norm"] = (
        df.get("Condition").fillna("").astype(str).str.strip().str.title()
    )

    # Figures
    fig1, ax1 = plt.subplots(figsize=(7, 4))
    ax1.hist(df["Rating_num"].dropna(), bins=np.arange(0, 10.5, 0.5), color="#4e79a7")
    ax1.set_xlabel("Hotel rating (0–10)")
    ax1.set_ylabel("Count")
    ax1.set_title("Distribution of hotel ratings")
    f1 = figures_dir / "hotels_rating_hist.png"
    save_figure(fig1, f1)
    plt.close(fig1)

    # Scatter: rating vs log(total reviews)
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    x = df["Rating_num"]
    y = df["Total_Reviews_num"].clip(lower=1)
    ax2.scatter(x, np.log10(y), s=8, alpha=0.5, color="#f28e2b")
    ax2.set_xlabel("Hotel rating (0–10)")
    ax2.set_ylabel("log10(total reviews)")
    ax2.set_title("Rating vs popularity (reviews)")
    f2 = figures_dir / "hotels_rating_vs_reviews.png"
    save_figure(fig2, f2)
    plt.close(fig2)

    # Condition counts
    fig3, ax3 = plt.subplots(figsize=(7, 4))
    cond_counts = (
        df["Condition_norm"].replace({"": "Unknown"}).value_counts().head(12)
    )
    cond_counts.plot(kind="bar", color="#59a14f", ax=ax3)
    ax3.set_xlabel("Condition label")
    ax3.set_ylabel("Count")
    ax3.set_title("Hotel condition labels (top)")
    f3 = figures_dir / "hotels_condition_counts.png"
    save_figure(fig3, f3)
    plt.close(fig3)

    # Insights (concise)
    num_rows = len(df)
    rating_valid = df["Rating_num"].notna().sum()
    rating_mean = df["Rating_num"].mean()
    rating_median = df["Rating_num"].median()
    missing_rating = num_rows - rating_valid
    top_cities = df["City"].value_counts().head(5)

    sections = [
        {
            "heading": "Dataset snapshot",
            "content": (
                f"Rows: {num_rows}. Ratings present: {rating_valid} (missing {missing_rating}).\n"
                f"Mean rating: {rating_mean:.2f}; median: {rating_median:.2f}.\n"
                f"Top cities by listings: {', '.join([f'{c} ({n})' for c, n in top_cities.items()])}.\n\n"
                "### Why this matters for EarthRoutes\n"
                "- Coverage and missingness determine confidence in rankings; with ~13.5% ratings missing,\n"
                "  down‑weight or caveat sparse items.\n"
                "- City skew (e.g., Jaipur/Bangalore/Mumbai) may induce popularity bias; diversify within cities\n"
                "  so overrepresented metros don’t crowd out others.\n"
                "- Evaluation readiness: track missingness and skew as data‑quality metrics and surface brief disclaimers when needed."
            ),
            "images": [
                str(f1.relative_to(project_root)),
            ],
            "image_captions": [
                "Hotel ratings concentrate between 7–9, with a long tail of lower scores—useful to set expectation and calibrate filters.",
            ],
        },
        {
            "heading": "Popularity vs quality",
            "content": (
                "Most hotels cluster between ratings 7–9; higher-rated hotels generally have more reviews,\n"
                "but there are notable high-rating, low-review outliers worth spotlighting.\n\n"
                "### Why this matters for EarthRoutes\n"
                "- Distinguishes ‘hidden gems’ (high rating, low reviews) from ‘tourist staples’ (high rating, high reviews)\n"
                "  to present transparent trade‑offs.\n"
                "- When ratings are comparable, prefer hotels nearer low‑CO₂ POIs or public transport and state this tie‑breaker\n"
                "  explicitly in recommendations."
            ),
            "images": [
                str(f2.relative_to(project_root)),
            ],
            "image_captions": [
                "Points toward the upper-right are ‘tourist staples’; upper-left suggests ‘hidden gems’ worth proposing with clear caveats.",
            ],
        },
        {
            "heading": "Condition labels",
            "content": (
                "Condition categories like ‘Exceptional’, ‘Very Good’, and ‘Good’ dominate; normalize these\n"
                "to an ordered scale for consistent UX and filtering.\n\n"
                "### Why this matters for EarthRoutes\n"
                "- Raw labels are inconsistent; normalizing to a strict ordinal prevents ranking artifacts and improves\n"
                "  conversational filters (e.g., ‘Very Good or better’).\n"
                "- Retrieval: store both raw and normalized labels; use normalized for filtering and raw for display/citation."
            ),
            "images": [
                str(f3.relative_to(project_root)),
            ],
            "image_captions": [
                "Label frequency is uneven and partly overlapping—reinforces the need for a strict ordinal mapping before ranking.",
            ],
        },
        {
            "heading": "Cross‑cutting implications",
            "content": (
                "- Retrieval/ranking: use proximity to sustainable POIs as a tie‑breaker when ratings are similar.\n"
                "- Explainability: call out popularity vs quality trade‑offs (e.g., ‘less crowded but similarly rated’).\n"
                "- Data governance: monitor missing ratings and duplicates by (Hotel Name, Place)."
            ),
            "images": [],
        },
    ]

    report_path = reports_dir / "hotels_report.md"
    write_markdown(report_path, "Hotels EDA", sections)


if __name__ == "__main__":
    run()


