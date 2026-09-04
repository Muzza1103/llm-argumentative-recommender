# Hotel scripts

This directory contains the executable scripts for the cold-start hotel recommendation pipeline.

The pipeline does not rely on an individual user history. It interprets a request for the current session, evaluates hotel facts and review opinions, generates traceable support and attack arguments, and aggregates them with DF-QuAD. Mandatory constraints determine eligibility separately from the argumentative score.

Run all commands from the project root.

## Required data

The main input is a prepared `hotel_profiles_complete.json` file containing hotel metadata, normalized facilities, policies, review annotations and aspect-level statistics.

This file and the original company data are not included in the repository. Extraction, cleaning, exploration, annotation and profile construction were performed separately in notebooks delivered to Jinko.

The deterministic facility ontology is available at:

```text
configs/hotel_facility_ontology.json
```

## Main scripts

| Script | Role |
|---|---|
| `validate_hotel_data.py` | Validates prepared hotel profiles and, optionally, their review-annotation file. |
| `evaluate_hotel_session.py` | Evaluates one hotel from structured preferences or a natural-language request. |
| `render_hotel_graph.py` | Converts an evaluation JSON file into a standalone HTML explanation. |
| `rank_hotel_session.py` | Interprets one request and evaluates several hotels with the same session profile. |
| `contest_hotel_evaluation.py` | Modifies existing soft preferences and recomputes one evaluation without another LLM call. |
| `inventory_hotel_facilities.py` | Compares observed facility names and identifiers with the canonical ontology. |
| `build_hotel_subset_from_rows.py` | Converts compatible exported rows into the project's historical JSONL format. |

## Model usage

Two execution modes are available:

- `baseline`: deterministic argument construction from structured preferences;
- `hybrid`: LLM-generated proposals followed by deterministic validation and scoring.

When `--preference-text` is used in hybrid mode, Gemini first converts the natural-language request into a structured session profile. A second Gemini call generates argument proposals from the closed source registry.

Gemini never computes the argument strengths, eligibility or DF-QuAD score. These stages remain deterministic.

## 1. Validate prepared data

Validate hotel profiles only:

```bash
python -m scripts.hotel.validate_hotel_data \
  --profiles PATH/TO/hotel_profiles_complete.json \
  --output-summary data/processed/hotel_validation_summary.json
```

Cross-check profiles against review annotations:

```bash
python -m scripts.hotel.validate_hotel_data \
  --profiles PATH/TO/hotel_profiles_complete.json \
  --reviews PATH/TO/hotel_review_annotations.jsonl \
  --output-summary data/processed/hotel_validation_summary.json
```

## 2. Evaluate one hotel

### Hybrid evaluation from natural language

```bash
python -m scripts.hotel.evaluate_hotel_session \
  --profiles PATH/TO/hotel_profiles_complete.json \
  --hotel-id HOTEL_ID \
  --preference-text "I need a quiet hotel in central London with reliable Wi-Fi. Parking would be a plus." \
  --argument-mode hybrid \
  --gemini-model gemini-2.5-flash \
  --gcp-project YOUR_PROJECT_ID \
  --gcp-location global \
  --output data/processed/hotel_evaluation.json
```

This execution usually makes two LLM calls: one for request interpretation and one for argument generation.

### Deterministic evaluation from a structured profile

```bash
python -m scripts.hotel.evaluate_hotel_session \
  --profiles PATH/TO/hotel_profiles_complete.json \
  --hotel-id HOTEL_ID \
  --preferences configs/hotel_session_example.json \
  --argument-mode baseline \
  --output data/processed/hotel_evaluation_baseline.json
```

This mode does not require an LLM call.

## 3. Render the explanation

```bash
python -m scripts.hotel.render_hotel_graph \
  --input data/processed/hotel_evaluation.json \
  --output data/processed/hotel_evaluation.html
```

The resulting HTML file presents the request, structured preferences, eligibility, arguments, evidence, strengths, aggregations and final score.

## 4. Rank several hotels

The request is interpreted once and the resulting session profile is reused for every candidate.

```bash
python -m scripts.hotel.rank_hotel_session \
  --profiles PATH/TO/hotel_profiles_complete.json \
  --preference-text "The hotel must be in London. I prefer a quiet central hotel with good Wi-Fi." \
  --candidate-count 10 \
  --argument-mode hybrid \
  --gemini-model gemini-2.5-flash \
  --gcp-project YOUR_PROJECT_ID \
  --gcp-location global \
  --output-dir data/processed/hotel_ranking
```

Specific candidates can be supplied instead of sampling them:

```bash
python -m scripts.hotel.rank_hotel_session \
  --profiles PATH/TO/hotel_profiles_complete.json \
  --preferences configs/hotel_session_example.json \
  --hotel-ids HOTEL_ID_1 HOTEL_ID_2 HOTEL_ID_3 \
  --argument-mode baseline \
  --output-dir data/processed/hotel_ranking_selected
```

The output directory contains:

- the reused structured session profile;
- one JSON evaluation and one HTML report per evaluated hotel;
- `ranking.json`;
- `ranking.csv`.

Only hotels whose mandatory constraints are explicitly satisfied receive a rank. Hotels marked `unknown` or `ineligible` remain in the explanatory outputs but are not ranked.

## 5. Contest an existing evaluation

Contestation reuses a hybrid evaluation and its validated Gemini proposals. It modifies only existing soft preferences or soft constraints, then recomputes weights, argument strengths, aggregations and the DF-QuAD score.

Change the importance of one aspect:

```bash
python -m scripts.hotel.contest_hotel_evaluation \
  --profiles PATH/TO/hotel_profiles_complete.json \
  --evaluation data/processed/hotel_evaluation.json \
  --set-importance wifi_internet=5 \
  --output data/processed/hotel_evaluation_contested.json \
  --html-output data/processed/hotel_evaluation_contested.html
```

Disable one or more existing preferences:

```bash
python -m scripts.hotel.contest_hotel_evaluation \
  --profiles PATH/TO/hotel_profiles_complete.json \
  --evaluation data/processed/hotel_evaluation.json \
  --disable-aspect parking_voiture \
  --disable-constraint CONSTRAINT_ID \
  --output data/processed/hotel_evaluation_contested.json
```

At least one contestation edit must be supplied. The source evaluation must have been produced in `hybrid` mode. No new LLM call is made.

## 6. Inspect facility coverage

```bash
python -m scripts.hotel.inventory_hotel_facilities \
  --profiles PATH/TO/hotel_profiles_complete.json \
  --ontology configs/hotel_facility_ontology.json \
  --output data/processed/hotel_facility_inventory.json
```

This utility reports recognized facilities, unmapped values and identifier/name conflicts.

## Eligibility and scoring

The pipeline distinguishes three eligibility states:

- `eligible`: every mandatory constraint is explicitly satisfied;
- `unknown`: at least one mandatory constraint cannot be verified and none is violated;
- `ineligible`: at least one mandatory constraint is explicitly violated.

Soft preferences and soft constraints contribute to the argumentative score. Mandatory constraints affect eligibility only and are not included in DF-QuAD.

## Notes

- Missing information is treated as `unknown`, not as proof of violation.
- Opinion arguments use the lower bound of the Wilson score to reduce the influence of small numbers of reviews.
- Fact arguments rely on normalized metadata, facilities or policies.
- Generated arguments must reference entries from the closed source registry before they can contribute to the score.
- Execution time depends mainly on Gemini and Vertex AI latency; deterministic recomputation and HTML rendering are substantially faster.

