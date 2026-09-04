#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 GCP_PROJECT_ID" >&2
  exit 1
fi

GCP_PROJECT_ID="$1"
DATASET="data/processed/yelp_subset_500_with_review_aspects_nli.jsonl"
ASPECT_MF_PREDICTIONS="data/processed/aspect_mf_predictions_nli_500.json"
MF_ITEM_PREDICTIONS="data/processed/mf_item_predictions_500.json"
GENERATED_ARGUMENTS="data/processed/generated_arguments_gemini_flash_500_unbalanced.jsonl"
VALID_ARGUMENTS="data/processed/generated_arguments_gemini_flash_500_unbalanced_valid.jsonl"
SCORED_ARGUMENTS="data/processed/scored_arguments_gemini_flash_500_unbalanced.jsonl"

for required_file in "$DATASET" "$ASPECT_MF_PREDICTIONS" "$MF_ITEM_PREDICTIONS"; do
  if [[ ! -f "$required_file" ]]; then
    echo "Missing required file: $required_file" >&2
    echo "Prepare Yelp and Aspect-MF data, then run run_baseline_mf.sh first." >&2
    exit 1
  fi
done

# Generate four arguments per example with a free support/attack distribution.
python -m scripts.restaurants.generate_batch \
  --input "$DATASET" \
  --output "$GENERATED_ARGUMENTS" \
  --backend gemini \
  --gemini-model gemini-2.5-flash \
  --gcp-project "$GCP_PROJECT_ID" \
  --gcp-location global \
  --max-new-tokens 5000 \
  --batch-size 5 \
  --num-examples 500 \
  --argument-mode unbalanced

# Compute the combined semantic and empirical strength of validated arguments.
python -m scripts.restaurants.score_batch \
  --dataset "$DATASET" \
  --input "$VALID_ARGUMENTS" \
  --output "$SCORED_ARGUMENTS" \
  --mf-predictions "$ASPECT_MF_PREDICTIONS" \
  --backend gemini \
  --gemini-model gemini-2.5-flash \
  --gcp-project "$GCP_PROJECT_ID" \
  --gcp-location global \
  --batch-size 5 \
  --max-new-tokens 4000

# CoRAL: neutral DF-QuAD root.
python -m scripts.restaurants.dfquad_batch \
  --input "$SCORED_ARGUMENTS" \
  --output data/processed/coral_500_unbalanced.jsonl \
  --dataset "$DATASET" \
  --root-base-score 0.5 \
  --root-base-source constant \
  --argument-score-source combined \
  --aggregation-method dfquad \
  --combination-method dfquad \
  --save-graph

python -m scripts.restaurants.evaluate_dfquad_scores \
  --input data/processed/coral_500_unbalanced.jsonl \
  --dataset "$DATASET" \
  --output-csv data/processed/evaluation_coral_500_unbalanced.csv \
  --output-summary data/processed/evaluation_coral_500_unbalanced_summary.json

# CoRAL (MF-init): the global MF prediction initializes the DF-QuAD root.
python -m scripts.restaurants.dfquad_batch \
  --input "$SCORED_ARGUMENTS" \
  --output data/processed/coral_mf_init_500_unbalanced.jsonl \
  --dataset "$DATASET" \
  --root-base-source mf_item \
  --mf-item-predictions "$MF_ITEM_PREDICTIONS" \
  --argument-score-source combined \
  --aggregation-method dfquad \
  --combination-method dfquad \
  --save-graph

python -m scripts.restaurants.evaluate_dfquad_scores \
  --input data/processed/coral_mf_init_500_unbalanced.jsonl \
  --dataset "$DATASET" \
  --output-csv data/processed/evaluation_coral_mf_init_500_unbalanced.csv \
  --output-summary data/processed/evaluation_coral_mf_init_500_unbalanced_summary.json

# CoRAL-corr: MF score corrected by lambda * (aggregated support - attack).
python -m scripts.restaurants.dfquad_batch \
  --input "$SCORED_ARGUMENTS" \
  --output data/processed/coral_corr_lambda_0_5_500_unbalanced.jsonl \
  --dataset "$DATASET" \
  --mf-item-predictions "$MF_ITEM_PREDICTIONS" \
  --mf-combination-mode mf_correction \
  --mf-lambda 0.5 \
  --argument-score-source combined \
  --aggregation-method dfquad \
  --combination-method dfquad \
  --save-graph

python -m scripts.restaurants.evaluate_dfquad_scores \
  --input data/processed/coral_corr_lambda_0_5_500_unbalanced.jsonl \
  --dataset "$DATASET" \
  --output-csv data/processed/evaluation_coral_corr_lambda_0_5_500_unbalanced.csv \
  --output-summary data/processed/evaluation_coral_corr_lambda_0_5_500_unbalanced_summary.json

