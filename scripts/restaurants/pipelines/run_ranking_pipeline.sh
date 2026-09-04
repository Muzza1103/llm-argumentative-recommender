#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${1:-YOUR_PROJECT_ID}"
RANKING_DATASET="data/processed/yelp_ranking_candidates_100_neg9.jsonl"

python -m scripts.restaurants.generate_batch \
  --input "$RANKING_DATASET" \
  --output data/processed/generated_arguments_gemini_flash_100_neg9_unbalanced.jsonl \
  --backend gemini \
  --gemini-model gemini-2.5-flash \
  --gcp-project "$PROJECT_ID" \
  --gcp-location global \
  --max-new-tokens 5000 \
  --batch-size 5 \
  --num-examples 1000 \
  --argument-mode unbalanced

python -m scripts.restaurants.score_batch \
  --dataset "$RANKING_DATASET" \
  --input data/processed/generated_arguments_gemini_flash_100_neg9_unbalanced_valid.jsonl \
  --output data/processed/scored_arguments_gemini_flash_100_neg9_unbalanced.jsonl \
  --mf-predictions data/processed/aspect_mf_predictions_nli_500.json \
  --backend gemini \
  --gemini-model gemini-2.5-flash \
  --gcp-project "$PROJECT_ID" \
  --gcp-location global \
  --batch-size 5 \
  --max-new-tokens 4000

python -m scripts.restaurants.dfquad_batch \
  --input data/processed/scored_arguments_gemini_flash_100_neg9_unbalanced.jsonl \
  --output data/processed/dfquad_ranking_original_100_neg9_unbalanced.jsonl \
  --dataset "$RANKING_DATASET" \
  --aggregation-method dfquad \
  --combination-method dfquad \
  --save-graph

python -m scripts.restaurants.evaluate_ranking \
  --input data/processed/dfquad_ranking_original_100_neg9_unbalanced.jsonl \
  --dataset "$RANKING_DATASET" \
  --output-summary data/processed/evaluation_ranking_original_100_neg9_unbalanced_summary.json \
  --score-source dfquad \
  --score-key final_score \
  --k 1 3 5 10 \
  --require-full-groups