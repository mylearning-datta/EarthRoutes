# Hotels EDA

## Dataset snapshot
Rows: 9235. Ratings present: 7988 (missing 1247).
Mean rating: 6.95; median: 7.30.
Top cities by listings: Jaipur (970), Bangalore (935), Mumbai (795), Chennai (705), Kolkata (605).

### Why this matters for EarthRoutes
- Coverage and missingness determine confidence in rankings; with ~13.5% ratings missing,
  down‑weight or caveat sparse items.
- City skew (e.g., Jaipur/Bangalore/Mumbai) may induce popularity bias; diversify within cities
  so overrepresented metros don’t crowd out others.
- Evaluation readiness: track missingness and skew as data‑quality metrics and surface brief disclaimers when needed.

<figure style="text-align: center;">
<img src="eda/figures/hotels_rating_hist.png" alt="Dataset snapshot" style="max-width: 100%; width: 720px; height: auto;" />
<figcaption><em>Figure: Hotel ratings concentrate between 7–9, with a long tail of lower scores—useful to set expectation and calibrate filters.</em></figcaption>
</figure>

## Popularity vs quality
Most hotels cluster between ratings 7–9; higher-rated hotels generally have more reviews,
but there are notable high-rating, low-review outliers worth spotlighting.

### Why this matters for EarthRoutes
- Distinguishes ‘hidden gems’ (high rating, low reviews) from ‘tourist staples’ (high rating, high reviews)
  to present transparent trade‑offs.
- When ratings are comparable, prefer hotels nearer low‑CO₂ POIs or public transport and state this tie‑breaker
  explicitly in recommendations.

<figure style="text-align: center;">
<img src="eda/figures/hotels_rating_vs_reviews.png" alt="Popularity vs quality" style="max-width: 100%; width: 720px; height: auto;" />
<figcaption><em>Figure: Points toward the upper-right are ‘tourist staples’; upper-left suggests ‘hidden gems’ worth proposing with clear caveats.</em></figcaption>
</figure>

## Condition labels
Condition categories like ‘Exceptional’, ‘Very Good’, and ‘Good’ dominate; normalize these
to an ordered scale for consistent UX and filtering.

### Why this matters for EarthRoutes
- Raw labels are inconsistent; normalizing to a strict ordinal prevents ranking artifacts and improves
  conversational filters (e.g., ‘Very Good or better’).
- Retrieval: store both raw and normalized labels; use normalized for filtering and raw for display/citation.

<figure style="text-align: center;">
<img src="eda/figures/hotels_condition_counts.png" alt="Condition labels" style="max-width: 100%; width: 720px; height: auto;" />
<figcaption><em>Figure: Label frequency is uneven and partly overlapping—reinforces the need for a strict ordinal mapping before ranking.</em></figcaption>
</figure>

## Cross‑cutting implications
- Retrieval/ranking: use proximity to sustainable POIs as a tie‑breaker when ratings are similar.
- Explainability: call out popularity vs quality trade‑offs (e.g., ‘less crowded but similarly rated’).
- Data governance: monitor missing ratings and duplicates by (Hotel Name, Place).
