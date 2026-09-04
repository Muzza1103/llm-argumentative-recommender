#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 GCP_PROJECT_ID" >&2
  exit 1
fi

GCP_PROJECT_ID="$1"
RANKING_DATASET="data/processed/yelp_ranking_candidates_100_neg9.jsonl"
ASPECT_MF_PREDICTIONS="data/processed/aspect_mf_predictions_nli_500.json"
MF_ITEM_PREDICTIONS="data/processed/mf_item_ranking_predictions_100_neg9.json"
GENERATED_ARGUMENTS="data/processed/generated_arguments_gemini_flash_100_neg9_unbalanced.jsonl"
VALID_ARGUMENTS="data/processed/generated_arguments_gemini_flash_100_neg9_unbalanced_valid.jsonl"
SCORED_ARGUMENTS="data/processed/scored_arguments_gemini_flash_100_neg9_unbalanced.jsonl"

for required_file in "$RANKING_DATASET" "$ASPECT_MF_PREDICTIONS" "$MF_ITEM_PREDICTIONS"; do
  if [[ ! -f "$required_file" ]]; then
    echo "Missing required file: $required_file" >&2
    echo "Build the corrected seen-item benchmark and run run_baseline_mf.sh first." >&2
    exit 1
  fi
done

# RANKING_DATASET must contain one held-out target and nine seen-item candidates
# per group. Ties must be resolved independently of candidate insertion order.

python -m scripts.restaurants.generate_batch \
  --input "$RANKING_DATASET" \
  --output "$GENERATED_ARGUMENTS" \
  --backend gemini \
  --gemini-model gemini-2.5-flash \
  --gcp-project "$GCP_PROJECT_ID" \
  --gcp-location global \
  --max-new-tokens 5000 \
  --batch-size 5 \
  --num-examples 1000 \
  --argument-mode unbalanced

python -m scripts.restaurants.score_batch \
  --dataset "$RANKING_DATASET" \
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
  --output data/processed/coral_ranking_100_neg9_unbalanced.jsonl \
  --dataset "$RANKING_DATASET" \
  --root-base-score 0.5 \
  --root-base-source constant \
  --argument-score-source combined \
  --aggregation-method dfquad \
  --combination-method dfquad \
  --save-graph

python -m scripts.restaurants.evaluate_ranking \
  --input data/processed/coral_ranking_100_neg9_unbalanced.jsonl \
  --dataset "$RANKING_DATASET" \
  --output-summary data/processed/evaluation_coral_ranking_100_neg9_unbalanced_summary.json \
  --score-source dfquad \
  --score-key final_score \
  --k 1 3 5 10 \
  --require-full-groups

# CoRAL (MF-init): the global MF prediction initializes the DF-QuAD root.
python -m scripts.restaurants.dfquad_batch \
  --input "$SCORED_ARGUMENTS" \
  --output data/processed/coral_mf_init_ranking_100_neg9_unbalanced.jsonl \
  --dataset "$RANKING_DATASET" \
  --root-base-source mf_item \
  --mf-item-predictions "$MF_ITEM_PREDICTIONS" \
  --argument-score-source combined \
  --aggregation-method dfquad \
  --combination-method dfquad \
  --save-graph

python -m scripts.restaurants.evaluate_ranking \
  --input data/processed/coral_mf_init_ranking_100_neg9_unbalanced.jsonl \
  --dataset "$RANKING_DATASET" \
  --output-summary data/processed/evaluation_coral_mf_init_ranking_100_neg9_unbalanced_summary.json \
  --score-source dfquad \
  --score-key final_score \
  --k 1 3 5 10 \
  --require-full-groups

# CoRAL-corr, lambda = 0.5. Scores are not clamped during ranking to avoid
# creating additional ties at the boundaries of the [0, 1] interval.
python -m scripts.restaurants.dfquad_batch \
  --input "$SCORED_ARGUMENTS" \
  --output data/processed/coral_corr_lambda_0_5_ranking_100_neg9_unbalanced.jsonl \
  --dataset "$RANKING_DATASET" \
  --mf-item-predictions "$MF_ITEM_PREDICTIONS" \
  --mf-combination-mode mf_correction \
  --mf-lambda 0.5 \
  --argument-score-source combined \
  --aggregation-method dfquad \
  --combination-method dfquad \
  --no-clamp-final-score \
  --save-graph

python -m scripts.restaurants.evaluate_ranking \
  --input data/processed/coral_corr_lambda_0_5_ranking_100_neg9_unbalanced.jsonl \
  --dataset "$RANKING_DATASET" \
  --output-summary data/processed/evaluation_coral_corr_lambda_0_5_ranking_100_neg9_unbalanced_summary.json \
  --score-source dfquad \
  --score-key final_score \
  --k 1 3 5 10 \
  --require-full-groups

# CoRAL-corr, lambda = 1.0.
python -m scripts.restaurants.dfquad_batch \
  --input "$SCORED_ARGUMENTS" \
  --output data/processed/coral_corr_lambda_1_ranking_100_neg9_unbalanced.jsonl \
  --dataset "$RANKING_DATASET" \
  --mf-item-predictions "$MF_ITEM_PREDICTIONS" \
  --mf-combination-mode mf_correction \
  --mf-lambda 1.0 \
  --argument-score-source combined \
  --aggregation-method dfquad \
  --combination-method dfquad \
  --no-clamp-final-score \
  --save-graph

python -m scripts.restaurants.evaluate_ranking \
  --input data/processed/coral_corr_lambda_1_ranking_100_neg9_unbalanced.jsonl \
  --dataset "$RANKING_DATASET" \
  --output-summary data/processed/evaluation_coral_corr_lambda_1_ranking_100_neg9_unbalanced_summary.json \
  --score-source dfquad \
  --score-key final_score \
  --k 1 3 5 10 \
  --require-full-groups
