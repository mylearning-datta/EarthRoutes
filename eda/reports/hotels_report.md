# Hotels EDA

## Analysis 1: Dataset snapshot
Rows: 9235. Ratings present: 7988 (missing 1247).
Mean rating: 6.95; median: 7.30.
Top cities by listings: Jaipur (970), Bangalore (935), Mumbai (795), Chennai (705), Kolkata (605).

### Figure: Distribution of hotel ratings
![Dataset snapshot](eda/figures/hotels_rating_hist.png)

### Why this matters for EarthRoutes
- Coverage and missingness determine how confidently we rank hotels; with ~13.5% ratings missing, the agent should down‑weight or explicitly caveat such items.
- City skew (Jaipur/Bangalore/Mumbai) reveals potential popularity bias; retrieval should diversify within cities so overrepresented metros don’t crowd out others.
- Evaluation readiness: track rating-missingness and city skew as data quality metrics; surface a brief disclaimer when a recommendation relies on sparse signals.

---

## Analysis 2: Popularity vs quality
Most hotels cluster between ratings 7–9; higher-rated hotels generally have more reviews,
but there are notable high-rating, low-review outliers worth spotlighting.

### Figure: Rating vs popularity (log reviews)
![Popularity vs quality](eda/figures/hotels_rating_vs_reviews.png)

### Why this matters for EarthRoutes
- Distinguishes “hidden gems” (high rating, low reviews) from “tourist staples” (high rating, high reviews) to present transparent trade‑offs.
- When hotels are comparable on rating, prefer those nearer low‑CO₂ POIs or public transport; explain the tie‑breaker in recommendations.
- Ranking logic: add a tie‑breaker that scores proximity to sustainable POIs and transit; keep both raw (rating, reviews) and derived (tie‑breaker) signals visible in explanations.

---

## Analysis 3: Condition labels
Condition categories like ‘Exceptional’, ‘Very Good’, and ‘Good’ dominate; normalize these
to an ordered scale for consistent UX and filtering.

### Figure: Condition label frequency
![Condition labels](eda/figures/hotels_condition_counts.png)

### Why this matters for EarthRoutes
- Raw labels are inconsistent across sources; normalizing to a strict ordinal prevents ranking artifacts and improves conversational filters (e.g., “Very Good or better”).
- Retrieval: store both raw and normalized labels; use normalized for filters and raw for display to preserve source fidelity in RAG citations.

---

## Cross‑cutting implications
- Retrieval and ranking: normalize and store `condition (ordinal)`, and use proximity to sustainable POIs as a tie‑breaker when ratings are similar.
- Explainability: explicitly call out popularity vs quality trade‑offs (e.g., “less crowded but similarly rated”).
- Data governance: keep a lightweight QC dashboard for missing ratings and duplicated hotel entries by `(Hotel Name, Place)`.
