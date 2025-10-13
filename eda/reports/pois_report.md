# POIs EDA

## Analysis 1: Schema and coverage
Rows: 325. Airport within 50km available for ~69.8% of POIs.
Top types: Temple (59), Beach (25), Fort (22), Lake (16), National Park (14), Palace (12), Museum (11), Waterfall (11), Monument (9), Cave (8), Park (7), Zoo (7), Mall (7), Valley (7), Monastery (7).

### Why this matters for EarthRoutes
- Airport availability gates mode‑choice comparisons; where absent, avoid proposing flights and focus on rail/road.
- The city/type inventory powers eco‑alternative suggestions when users request higher‑impact types.
- Geospatial readiness: use airport flag to decide flight comparisons; where absent, bias toward walkable clusters of sustainable POIs.

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
