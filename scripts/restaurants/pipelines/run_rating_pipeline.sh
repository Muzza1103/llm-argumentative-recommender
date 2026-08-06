#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${1:-YOUR_PROJECT_ID}"
DATASET="data/processed/yelp_subset_500_with_review_aspects_nli.jsonl"

python -m scripts.restaurants.generate_batch \
  --input "$DATASET" \
  --output data/processed/generated_arguments_gemini_flash_500_unbalanced.jsonl \
  --backend gemini \
  --gemini-model gemini-2.5-flash \
  --gcp-project "$PROJECT_ID" \
  --gcp-location global \
  --max-new-tokens 5000 \
  --batch-size 5 \
  --num-examples 500 \
  --argument-mode unbalanced

python -m scripts.restaurants.score_batch \
  --dataset "$DATASET" \
  --input data/processed/generated_arguments_gemini_flash_500_unbalanced_valid.jsonl \
  --output data/processed/scored_arguments_gemini_flash_500_unbalanced.jsonl \
  --mf-predictions data/processed/aspect_mf_predictions_nli_500.json \
  --backend gemini \
  --gemini-model gemini-2.5-flash \
  --gcp-project "$PROJECT_ID" \
  --gcp-location global \
  --batch-size 5 \
  --max-new-tokens 4000

python -m scripts.restaurants.dfquad_batch \
  --input data/processed/scored_arguments_gemini_flash_500_unbalanced.jsonl \
  --output data/processed/dfquad_original_500_unbalanced.jsonl \
  --dataset "$DATASET" \
  --aggregation-method dfquad \
  --combination-method dfquad \
  --save-graph

python -m scripts.restaurants.evaluate_dfquad_scores \
  --input data/processed/dfquad_original_500_unbalanced.jsonl \
  --dataset "$DATASET" \
  --output-csv data/processed/evaluation_dfquad_original_500_unbalanced.csv \
  --output-summary data/processed/evaluation_dfquad_original_500_unbalanced_summary.json