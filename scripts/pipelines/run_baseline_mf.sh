#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${1:-YOUR_PROJECT_ID}"
DATASET="data/processed/yelp_subset_500_with_review_aspects_nli.jsonl"
RANKING_DATASET="data/processed/yelp_ranking_candidates_100_neg9.jsonl"

python -m scripts.build_mf_dataset \
  --input "$DATASET" \
  --output data/processed/mf_item_dataset_500.csv

python -m scripts.train_mf \
  --mf-data data/processed/mf_item_dataset_500.csv \
  --source-dataset "$DATASET" \
  --output data/processed/mf_item_predictions_500.json

python -m scripts.evaluate_mf_predictions \
  --predictions data/processed/mf_item_predictions_500.json \
  --dataset "$DATASET" \
  --output-csv data/processed/evaluation_mf_item_500.csv \
  --output-summary data/processed/evaluation_mf_item_500_summary.json

python -m scripts.train_mf \
  --mf-data data/processed/mf_item_dataset_500.csv \
  --source-dataset "$RANKING_DATASET" \
  --output data/processed/mf_item_ranking_predictions_100_neg9.json

python -m scripts.evaluate_ranking \
  --input data/processed/mf_item_ranking_predictions_100_neg9.json \
  --dataset "$RANKING_DATASET" \
  --output-summary data/processed/evaluation_mf_ranking_100_neg9_summary.json \
  --score-source direct \
  --score-key score \
  --k 1 3 5 10 \
  --require-full-groups
