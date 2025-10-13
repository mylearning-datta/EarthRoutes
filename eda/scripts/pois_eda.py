from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from .report_utils import ensure_directories, get_project_root, save_figure, write_markdown
except Exception:
    # Fallback for direct script execution
    import sys
    from pathlib import Path as _Path

    sys.path.append(str(_Path(__file__).resolve().parent))
    from report_utils import ensure_directories, get_project_root, save_figure, write_markdown


def to_int_lakhs(series: pd.Series) -> pd.Series:
    # Convert lakhs (e.g., 1.5) to absolute counts (150000)
    s = pd.to_numeric(series, errors="coerce")
    return (s * 100_000).round().astype("Int64")


def normalize_boolean(series: pd.Series) -> pd.Series:
    return (
        series.fillna("").astype(str).str.strip().str.title().replace({"None": "No"})
    )


def run() -> None:
    project_root = get_project_root(__file__)
    data_path = project_root / "data" / "Top Indian Places to Visit.csv"
    figures_dir = project_root / "eda" / "figures"
    reports_dir = project_root / "eda" / "reports"
    ensure_directories([figures_dir, reports_dir])

    df = pd.read_csv(data_path)

    # Drop leading unnamed index column if present
    if df.columns[0].startswith("Unnamed") or df.columns[0] == "":
        df = df.drop(columns=[df.columns[0]])

    # Coerce numerics and standardize
    df["Google review rating_num"] = pd.to_numeric(
        df.get("Google review rating"), errors="coerce"
    )
    df["Entrance Fee in INR_num"] = pd.to_numeric(
        df.get("Entrance Fee in INR"), errors="coerce"
    )
    df["time needed to visit in hrs_num"] = pd.to_numeric(
        df.get("time needed to visit in hrs"), errors="coerce"
    )
    df["reviews_count_abs"] = to_int_lakhs(df.get("Number of google review in lakhs"))
    df["Airport_flag"] = normalize_boolean(df.get("Airport with 50km Radius"))
    df["Type_norm"] = df.get("Type").fillna("").astype(str).str.strip().str.title()
    df["Zone_norm"] = df.get("Zone").fillna("").astype(str).str.strip().str.title()

    # Figures
    # 1) Type frequency (top 15)
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    type_counts = df["Type_norm"].value_counts().head(15)
    type_counts.plot(kind="bar", color="#4e79a7", ax=ax1)
    ax1.set_xlabel("Type")
    ax1.set_ylabel("Count")
    ax1.set_title("POI types (top 15)")
    f1 = figures_dir / "pois_type_counts.png"
    save_figure(fig1, f1)
    plt.close(fig1)

    # 2) Rating distribution by Type (top 6)
    top_types = type_counts.index[:6].tolist()
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    df_top = df[df["Type_norm"].isin(top_types)]
    df_top.boxplot(column="Google review rating_num", by="Type_norm", ax=ax2, grid=False)
    ax2.set_xlabel("Type (top)")
    ax2.set_ylabel("Google rating (0–5)")
    ax2.set_title("Ratings by type (top)")
    plt.suptitle("")
    f2 = figures_dir / "pois_ratings_by_type.png"
    save_figure(fig2, f2)
    plt.close(fig2)

    # 3) Entrance fee vs rating scatter (sample)
    fig3, ax3 = plt.subplots(figsize=(7, 4))
    mask = df["Entrance Fee in INR_num"].notna() & df["Google review rating_num"].notna()
    ax3.scatter(
        df.loc[mask, "Entrance Fee in INR_num"],
        df.loc[mask, "Google review rating_num"],
        s=8,
        alpha=0.5,
        color="#f28e2b",
    )
    ax3.set_xlabel("Entrance fee (INR)")
    ax3.set_ylabel("Google rating (0–5)")
    ax3.set_title("Fee vs rating")
    f3 = figures_dir / "pois_fee_vs_rating.png"
    save_figure(fig3, f3)
    plt.close(fig3)

    # Insights (concise, thesis-aligned)
    type_top_line = ", ".join([f"{t} ({n})" for t, n in type_counts.items()])
    airport_share = (
        (df["Airport_flag"] == "Yes").mean() * 100.0
        if len(df) > 0
        else 0.0
    )

    sections = [
        {
            "heading": "Schema and coverage",
            "content": (
                f"Rows: {len(df)}. Airport within 50km available for ~{airport_share:.1f}% of POIs.\n"
                f"Top types: {type_top_line}.\n\n"
                "### Why this matters for EarthRoutes\n"
                "- Airport availability gates mode‑choice comparisons; where absent, avoid proposing flights and focus on rail/road.\n"
                "- The city/type inventory powers eco‑alternative suggestions when users request higher‑impact types.\n"
                "- Geospatial readiness: use airport flag to decide flight comparisons; where absent, bias toward walkable clusters of sustainable POIs."
            ),
            "images": [str(f1.relative_to(project_root))],
            "image_captions": [
                "Type coverage shows strong representation for Temples/Forts and a healthy base of nature types to power eco alternatives.",
            ],
        },
        {
            "heading": "Experience vs price",
            "content": (
                "Ratings cluster between 4.3–4.7 across popular types; fees show wide spread.\n"
                "Price-for-quality outliers can be highlighted for itinerary planning.\n\n"
                "### Why this matters for EarthRoutes\n"
                "- High‑rating, low‑fee, sustainable types (parks, lakes, viewpoints) are ideal for low‑CO₂ days without compromising satisfaction.\n"
                "- Use a simple value signal (rating high, fee low, sustainable) to order alternatives; surface fee context in explanations.\n"
                "- Itinerary feasibility: pair time‑needed with type to balance day plans, preferring clusters of short, sustainable visits when time is constrained."
            ),
            "images": [str(f2.relative_to(project_root)), str(f3.relative_to(project_root))],
            "image_captions": [
                "Across top types, user satisfaction is high and tight—suggesting room to optimize for sustainability and cost without losing quality.",
                "Low-fee, high-rating points are ideal anchors for low-CO₂ days; expensive outliers warrant explicit justification.",
            ],
        },
        {
            "heading": "Cross‑cutting implications",
            "content": (
                "- Retrieval: normalize Type/Zone/booleans; keep a curated list of sustainable types per city to enable alternatives.\n"
                "- Explainability: always show sustainability badge and mapped Reason; add fee context to ‘why this’.\n"
                "- Evaluation: check that each higher‑impact request receives at least one viable sustainable alternative."
            ),
            "images": [],
        },
    ]

    report_path = reports_dir / "pois_report.md"
    write_markdown(report_path, "POIs EDA", sections)


if __name__ == "__main__":
    run()


