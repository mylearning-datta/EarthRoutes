# Sustainability Types EDA

## Analysis 1: Join coverage
Sustainability label coverage over POIs: 100.0%.

### Why this matters for EarthRoutes
- Guarantees every POI can carry a sustainability badge and concise rationale, enabling consistent, transparent nudges.
- Evaluation readiness: with 100% mapping, measure changes in selections when labeled alternatives are shown vs hidden.

### Figure: POIs by sustainability label
![Join coverage](eda/figures/sust_label_counts.png)

_Figure: Every POI is labeled sustainable/non-sustainable, enabling consistent badges and rationales in recommendations._

---

## Analysis 2: Top unmapped types
None

### Why this matters for EarthRoutes
- No blind spots in the mapping; alternatives and city‑level ‘sustainable share’ metrics remain reliable.

---

## Cross‑cutting implications
- Data pipeline: enforce type normalization (title‑case, dedupe synonyms like Tomb/Tombs) and unit test for 100% mapping.
- UX: include the mapped Reason verbatim in explanations to anchor claims; avoid generic ‘eco’ phrasing.
- Analytics: maintain a city‑level ‘sustainable share’ metric to target cities where nudges matter most.
