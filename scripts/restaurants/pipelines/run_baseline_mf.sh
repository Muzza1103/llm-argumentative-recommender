#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 0 ]]; then
  echo "Usage: $0" >&2
  exit 1
fi

DATASET="data/processed/yelp_subset_500_with_review_aspects_nli.jsonl"
RANKING_DATASET="data/processed/yelp_ranking_candidates_100_neg9.jsonl"
MF_DATASET="data/processed/mf_item_dataset_500.csv"
MF_RATING_PREDICTIONS="data/processed/mf_item_predictions_500.json"
MF_RANKING_PREDICTIONS="data/processed/mf_item_ranking_predictions_100_neg9.json"

for required_file in "$DATASET" "$RANKING_DATASET"; do
  if [[ ! -f "$required_file" ]]; then
    echo "Missing required file: $required_file" >&2
    exit 1
  fi
done

# Build the user-item training data used by the MF-only baseline.
python -m scripts.restaurants.build_mf_dataset \
  --input "$DATASET" \
  --output "$MF_DATASET"

# Rating prediction.
python -m scripts.restaurants.train_mf \
  --mf-data "$MF_DATASET" \
  --source-dataset "$DATASET" \
  --output "$MF_RATING_PREDICTIONS"

python -m scripts.restaurants.evaluate_mf_predictions \
  --predictions "$MF_RATING_PREDICTIONS" \
  --dataset "$DATASET" \
  --output-csv data/processed/evaluation_mf_item_500.csv \
  --output-summary data/processed/evaluation_mf_item_500_summary.json

# Ranking. RANKING_DATASET must be the corrected seen-item benchmark.
python -m scripts.restaurants.train_mf \
  --mf-data "$MF_DATASET" \
  --source-dataset "$RANKING_DATASET" \
  --output "$MF_RANKING_PREDICTIONS"

python -m scripts.restaurants.evaluate_ranking \
  --input "$MF_RANKING_PREDICTIONS" \
  --dataset "$RANKING_DATASET" \
  --output-summary data/processed/evaluation_mf_ranking_100_neg9_summary.json \
  --score-source direct \
  --score-key score \
  --k 1 3 5 10 \
  --require-full-groups

