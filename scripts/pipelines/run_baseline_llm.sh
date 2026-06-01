#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${1:-YOUR_PROJECT_ID}"
DATASET="data/processed/yelp_subset_500_with_review_aspects_nli.jsonl"
RANKING_DATASET="data/processed/yelp_ranking_candidates_100_neg9.jsonl"

# LLM-only rating prediction
python -m scripts.llm_direct_score \
  --input "$DATASET" \
  --output data/processed/llm_direct_scores_gemini_flash_500.jsonl \
  --gemini-model gemini-2.5-flash \
  --gcp-project "$PROJECT_ID" \
  --gcp-location global \
  --batch-size 5 \
  --max-new-tokens 500 \
  --num-examples 500

python -m scripts.evaluate_llm_direct_scores \
  --input data/processed/llm_direct_scores_gemini_flash_500.jsonl \
  --dataset "$DATASET" \
  --output-csv data/processed/evaluation_llm_direct_500.csv \
  --output-summary data/processed/evaluation_llm_direct_500_summary.json

# LLM-only ranking
python -m scripts.llm_direct_score \
  --input "$RANKING_DATASET" \
  --output data/processed/llm_direct_scores_gemini_flash_ranking_100_neg9.jsonl \
  --gemini-model gemini-2.5-flash \
  --gcp-project "$PROJECT_ID" \
  --gcp-location global \
  --batch-size 5 \
  --max-new-tokens 500 \
  --num-examples 1000

python -m scripts.evaluate_ranking \
  --input data/processed/llm_direct_scores_gemini_flash_ranking_100_neg9.jsonl \
  --dataset "$RANKING_DATASET" \
  --output-summary data/processed/evaluation_llm_direct_ranking_100_neg9_summary.json \
  --score-source direct \
  --score-key score \
  --k 1 3 5 10 \
  --require-full-groups