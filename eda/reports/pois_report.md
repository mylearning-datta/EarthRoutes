# POIs EDA

## Analysis 1: Schema and coverage
We have 325 places in our dataset. About 70% of these places have an airport within 50km.
The most common types are: Temples (59), Beaches (25), Forts (22), Lakes (16), National Parks (14), Palaces (12), Museums (11), Waterfalls (11), Monuments (9), Caves (8), Parks (7), Zoos (7), Malls (7), Valleys (7), and Monasteries (7).

### Why this matters for EarthRoutes
- When there's no nearby airport, we can skip flight options and recommend trains or buses instead.
- Knowing what types of places exist in each city helps us suggest eco-friendly alternatives when travelers ask for less sustainable options.
- We can use airport proximity to decide when to offer flight comparisons, and when a place has no airport, we focus on recommending clusters of sustainable attractions that are easy to walk between.

### Figure: Top POI types by count
![Schema and coverage](eda/figures/pois_type_counts.png)

_Figure: Type coverage shows strong representation for Temples/Forts and a healthy base of nature types to power eco alternatives._

---

## Analysis 2: Experience vs price
Ratings cluster between 4.3–4.7 across popular types; fees show wide spread.
Price-for-quality outliers can be highlighted for itinerary planning.

### Why this matters for EarthRoutes
- High‑rating, low‑fee, sustainable types (parks, lakes, viewpoints) are ideal for low‑CO₂ days without compromising satisfaction.
- Use a simple value signal (rating high, fee low, sustainable) to order alternatives; surface fee context in explanations.
- Itinerary feasibility: pair time‑needed with type to balance day plans, preferring clusters of short, sustainable visits when time is constrained.

### Figure: Ratings distribution by top POI types
![Experience vs price](eda/figures/pois_ratings_by_type.png)

_Figure: Across top types, user satisfaction is high and tight—suggesting room to optimize for sustainability and cost without losing quality._

### Figure: Entrance fee vs rating scatter
![Experience vs price](eda/figures/pois_fee_vs_rating.png)

_Figure: Low-fee, high-rating points are ideal anchors for low-CO₂ days; expensive outliers warrant explicit justification._

---

## Cross‑cutting implications
- Retrieval: normalize Type/Zone/booleans; keep a curated list of sustainable types per city to enable alternatives.
- Explainability: always show sustainability badge and mapped Reason; add fee context to ‘why this’.
- Evaluation: check that each higher‑impact request receives at least one viable sustainable alternative.
