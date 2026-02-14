# POIs EDA

## Schema and coverage
Rows: 325. Airport within 50km available for ~69.8% of POIs.
Top types: Temple (59), Beach (25), Fort (22), Lake (16), National Park (14), Palace (12), Museum (11), Waterfall (11), Monument (9), Cave (8), Park (7), Zoo (7), Valley (7), Monastery (7), Mall (7).

### Why this matters for EarthRoutes
- When there's no nearby airport, we can skip flight options and recommend trains or buses instead.
- Knowing what types of places exist in each city helps us suggest eco-friendly alternatives when travelers ask for less sustainable options.
- We can use airport proximity to decide when to offer flight comparisons, and when a place has no airport, we focus on recommending clusters of sustainable attractions that are easy to walk between.

<figure style="text-align: center;">
<img src="eda/figures/pois_type_counts.png" alt="Schema and coverage" style="max-width: 100%; width: 720px; height: auto;" />
<figcaption><em>Figure: Type coverage shows strong representation for Temples/Forts and a healthy base of nature types to power eco alternatives.</em></figcaption>
</figure>

## Experience vs price
Ratings cluster between 4.3–4.7 across popular types; fees show wide spread.
Price-for-quality outliers can be highlighted for itinerary planning.

### Why this matters for EarthRoutes
- High‑rating, low‑fee, sustainable types (parks, lakes, viewpoints) are ideal for low‑CO₂ days without compromising satisfaction.
- Use a simple value signal (rating high, fee low, sustainable) to order alternatives; surface fee context in explanations.
- Itinerary feasibility: pair time‑needed with type to balance day plans, preferring clusters of short, sustainable visits when time is constrained.

<figure style="text-align: center;">
<img src="eda/figures/pois_ratings_by_type.png" alt="Experience vs price" style="max-width: 100%; width: 720px; height: auto;" />
<figcaption><em>Figure: Across top types, user satisfaction is high and tight—suggesting room to optimize for sustainability and cost without losing quality.</em></figcaption>
</figure>

<figure style="text-align: center;">
<img src="eda/figures/pois_fee_vs_rating.png" alt="Experience vs price" style="max-width: 100%; width: 720px; height: auto;" />
<figcaption><em>Figure: Low-fee, high-rating points are ideal anchors for low-CO₂ days; expensive outliers warrant explicit justification.</em></figcaption>
</figure>

## Cross‑cutting implications
- Retrieval: normalize Type/Zone/booleans; keep a curated list of sustainable types per city to enable alternatives.
- Explainability: always show sustainability badge and mapped Reason; add fee context to ‘why this’.
- Evaluation: check that each higher‑impact request receives at least one viable sustainable alternative.
