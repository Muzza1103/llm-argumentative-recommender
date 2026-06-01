# Scripts

This folder contains runnable utility scripts used throughout the project.

The scripts cover the full experimental pipeline:
- Yelp subset construction
- aspect extraction from reviews
- aspect-based collaborative filtering
- LLM-based argument generation
- argument validation
- semantic and empirical scoring
- argument graph construction
- DF-QuAD and contrastive aggregation
- rating prediction evaluation
- ranking evaluation
- MF-only and LLM-only baselines
- debugging and visualization

Run all commands from the project root.

---

## Execution modes

The project supports both:
- local Hugging Face models
- API-based LLMs such as Gemini through Google Cloud / Vertex AI

Most recent experiments use Gemini for argument generation and scoring.

---

## Google Cloud / Gemini setup

For Gemini-based generation and scoring, authenticate with Google Cloud:

```bash
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
gcloud auth application-default set-quota-project YOUR_PROJECT_ID
```

You can check the active project with:

```bash
gcloud config get-value project
```

Typical Gemini arguments used in scripts:

```bash
--backend gemini
--gemini-model gemini-2.5-flash
--gcp-project YOUR_PROJECT_ID
--gcp-location global
```

When using Gemini in batch mode, keep `--batch-size` moderate to avoid quota errors.
A batch size around `5` has been used as a stable setting in preliminary experiments.

---

## Input file conventions

Several scripts use similar argument names. Their meaning depends on the stage of the pipeline.

| Argument | Meaning |
| `--input` | Main input file for the script. This can be a dataset, generated arguments, scored arguments, or DF-QuAD output depending on the script. |
| `--dataset` | Source dataset used to recover user history, target ratings, or ranking metadata. |
| `--results` | Generated or scored argument file used by inspection/debug scripts. |
| `--mf-data` | CSV file used to train a Matrix Factorization model. |
| `--mf-predictions` | Precomputed MF predictions used during argument scoring. |
| `--source-dataset` | JSONL dataset containing target items to score with MF. |
| `--output` | Main output file produced by the script. |
| `--output-csv` | Per-example evaluation CSV. |
| `--output-summary` | Aggregated evaluation metrics JSON. |

Typical dataset files:
- `data/processed/yelp_subset_500_with_review_aspects_nli.jsonl`
- `data/processed/yelp_ranking_candidates_100_neg9.jsonl`

Typical generated files:
- `generated_arguments_*.jsonl`
- `scored_arguments_*.jsonl`
- `dfquad_*.jsonl`
- `evaluation_*_summary.json`

---

## Full argumentative pipeline

This is the main pipeline used to generate arguments, score them, aggregate them and evaluate the final score.


### 1. Dataset construction 

Builds a processed subset from the Yelp Open Dataset.

- Load raw Yelp files
- Filter restaurant businesses
- Filter informative reviews
- Group reviews by user
- Construct examples with:
  - user history
  - target item using leave-one-out
- Save the result as JSONL in `data/processed/`

### Usage

```bash
python -m scripts.build_yelp_subset \
  --business-file data/raw/business.json \
  --review-file data/raw/review.json \
  --user-file data/raw/user.json \
  --output data/processed/yelp_subset_500.jsonl \
  --num-users 500 \
  --history-size 5 \
  --min-user-reviews 6 \
  --seed 42
```

---

### 2. Extract review aspects

Extracts review aspects using an NLI model and a predefined aspect vocabulary.

The output dataset enriches user history and target items with review-level aspects.

### Usage

```bash
python -m scripts.extract_review_aspects_nli \
  --input data/processed/yelp_subset_500.jsonl \
  --output data/processed/yelp_subset_500_with_review_aspects_nli.jsonl \
  --aspect-vocab configs/aspect_vocabulary.json \
  --aspect-threshold 0.45
```

Main inputs:
- `--input`: Yelp JSONL dataset
- `--aspect-vocab`: aspect vocabulary file
- `--aspect-threshold`: minimum NLI confidence for keeping an aspect


### 3. Train aspect-based MF

### build_aspect_mf_dataset.py

Builds a user-aspect-rating dataset from extracted review aspects.

The output is a CSV used to train aspect-based Matrix Factorization.

### Usage

```bash
python -m scripts.build_aspect_mf_dataset \
  --input data/processed/yelp_subset_500_with_review_aspects_nli.jsonl \
  --output data/processed/aspect_mf_dataset_nli_500.csv
```

---

## train_aspect_mf.py

Trains an aspect-based MF model and produces predictions for user-aspect pairs.

### Usage

```bash
python -m scripts.train_aspect_mf \
  --input data/processed/aspect_mf_dataset_nli_500.csv \
  --output data/processed/aspect_mf_predictions_nli_500.json
```

Output:
- `aspect_mf_predictions_*.json`
- `aspect_mf_predictions_*_summary.json`

These predictions are used by `score_batch.py` as empirical argument-level signals.


### 4. Generate arguments

```bash
python -m scripts.generate_batch \
  --input data/processed/yelp_subset_500_with_review_aspects_nli.jsonl \
  --output data/processed/generated_arguments_gemini_flash_500_unbalanced.jsonl \
  --backend gemini \
  --gemini-model gemini-2.5-flash \
  --gcp-project YOUR_PROJECT_ID \
  --gcp-location global \
  --max-new-tokens 5000 \
  --batch-size 5 \
  --num-examples 500 \
  --argument-mode unbalanced
```

Main inputs:
- `--input`: Yelp JSONL dataset containing `history` and `target_item`
- `--argument-mode balanced`: enforce equal support and attack arguments
- `--argument-mode unbalanced`: allow the model to generate support/attack arguments more freely

Outputs:
- `generated_arguments_*.jsonl`
- `generated_arguments_*_valid.jsonl`
- `generated_arguments_*_invalid.jsonl`
- `generated_arguments_*_summary.json`


### 5. Score generated arguments

```bash
python -m scripts.score_batch \
  --dataset data/processed/yelp_subset_500_with_review_aspects_nli.jsonl \
  --input data/processed/generated_arguments_gemini_flash_500_unbalanced_valid.jsonl \
  --output data/processed/scored_arguments_gemini_flash_500_unbalanced.jsonl \
  --mf-predictions data/processed/aspect_mf_predictions_nli_500.json \
  --backend gemini \
  --gemini-model gemini-2.5-flash \
  --gcp-project YOUR_PROJECT_ID \
  --gcp-location global \
  --batch-size 5 \
  --max-new-tokens 4000
```

Main inputs:
- `--dataset`: original Yelp dataset used to recover user/item context
- `--input`: valid generated arguments
- `--mf-predictions`: aspect-MF predictions used as empirical signal

Output:
- scored arguments enriched with:
  - LLM score
  - MF score
  - combined score

---

### 6. Apply DF-QuAD / contrastive aggregation

Applies argumentative aggregation methods to scored arguments.

This script transforms scored support and attack arguments into a final recommendation score.

Supported aggregation methods include:
- DF-QuAD
- contrastive variants

### DF-QuAD

DF-QuAD is a gradual argumentation semantics used to aggregate support and attack relations within an argument graph.

Each argument contributes to the final recommendation score according to:
- its polarity (support or attack)
- its strength
- its position within the graph

DF-QuAD propagates supporting and attacking evidence through the graph in order to compute a structured recommendation score.

### Contrastive aggregation

The project also explores contrastive aggregation variants.

These variants aim to increase the separation between support and attack signals during aggregation.

The parameter `--contrastive-gamma` controls the amplification strength.

### Outputs

The script produces:
- final recommendation scores
- optional graph information
- optional graph visualisation data

DF-QuAD original:

```bash
python -m scripts.dfquad_batch \
  --input data/processed/scored_arguments_gemini_flash_500_unbalanced.jsonl \
  --output data/processed/dfquad_original_500_unbalanced.jsonl \
  --dataset data/processed/yelp_subset_500_with_review_aspects_nli.jsonl \
  --aggregation-method dfquad \
  --combination-method dfquad \
  --save-graph
```

Contrastive variant:

```bash
python -m scripts.dfquad_batch \
  --input data/processed/scored_arguments_gemini_flash_500_unbalanced.jsonl \
  --output data/processed/dfquad_mean_contrastive_gamma2_500_unbalanced.jsonl \
  --dataset data/processed/yelp_subset_500_with_review_aspects_nli.jsonl \
  --aggregation-method mean \
  --combination-method contrastive_power \
  --contrastive-gamma 2 \
  --save-graph
```

Main inputs:
- `--input`: scored arguments
- `--dataset`: source Yelp dataset
- `--save-graph`: save argument graph details for analysis/debugging

---

### 7. Evaluate rating prediction

```bash
python -m scripts.evaluate_dfquad_scores \
  --input data/processed/dfquad_original_500_unbalanced.jsonl \
  --dataset data/processed/yelp_subset_500_with_review_aspects_nli.jsonl \
  --output-csv data/processed/evaluation_dfquad_original_500_unbalanced.csv \
  --output-summary data/processed/evaluation_dfquad_original_500_unbalanced_summary.json
```

Metrics:
- MAE
- MSE
- RMSE

---

## Pipeline execution scripts

For reproducibility, the project provides shell scripts that automate the main experimental workflows:

- `scripts/pipelines/run_rating_pipeline.sh`
- `scripts/pipelines/run_ranking_pipeline.sh`
- `scripts/pipelines/run_baseline_mf.sh`
- `scripts/pipelines/run_baseline_llm.sh`

Before using them, make sure they are executable:

```bash
chmod +x scripts/pipelines/*.sh
```

### Usage

Argumentative rating prediction pipeline:

```bash
bash scripts/pipelines/run_rating_pipeline.sh YOUR_PROJECT_ID
```

Argumentative ranking pipeline:

```bash
bash scripts/pipelines/run_ranking_pipeline.sh YOUR_PROJECT_ID
```

MF-only and LLM-only baselines:

```bash
bash scripts/pipelines/run_baseline_mf.sh YOUR_PROJECT_ID
```

```bash
bash scripts/pipelines/run_baseline_llm.sh YOUR_PROJECT_ID
```

These scripts automatically execute the successive stages of the pipeline and generate all intermediate and final outputs in `data/processed/`.


## Baselines

## llm_direct_score.py

Runs the LLM-only direct scoring baseline.

The LLM directly predicts a score between 0 and 1 from:
- user history
- target item

No argument generation or DF-QuAD aggregation is used.

### Rating prediction usage

```bash
python -m scripts.llm_direct_score \
  --input data/processed/yelp_subset_500_with_review_aspects_nli.jsonl \
  --output data/processed/llm_direct_scores_gemini_flash_500.jsonl \
  --gemini-model gemini-2.5-flash \
  --gcp-project YOUR_PROJECT_ID \
  --gcp-location global \
  --batch-size 5 \
  --max-new-tokens 5000 \
  --num-examples 500
```

### Ranking usage

```bash
python -m scripts.llm_direct_score \
  --input data/processed/yelp_ranking_candidates_100_neg9.jsonl \
  --output data/processed/llm_direct_scores_gemini_flash_ranking_100_neg9.jsonl \
  --gemini-model gemini-2.5-flash \
  --gcp-project YOUR_PROJECT_ID \
  --gcp-location global \
  --batch-size 5 \
  --max-new-tokens 5000 \
  --num-examples 1000
```

---

## MF-only baseline scripts

## build_mf_dataset.py

Builds a user-item-rating dataset for the MF-only baseline.

By default, target items are not included in the training data.

### Usage

```bash
python -m scripts.build_mf_dataset \
  --input data/processed/yelp_subset_500_with_review_aspects_nli.jsonl \
  --output data/processed/mf_item_dataset_500.csv
```

---

## train_mf.py

Trains an MF model using Surprise SVD and generates predictions for target items.

### Rating prediction usage

```bash
python -m scripts.train_mf \
  --mf-data data/processed/mf_item_dataset_500.csv \
  --source-dataset data/processed/yelp_subset_500_with_review_aspects_nli.jsonl \
  --output data/processed/mf_item_predictions_500.json
```

### Ranking usage

```bash
python -m scripts.train_mf \
  --mf-data data/processed/mf_item_dataset_500.csv \
  --source-dataset data/processed/yelp_ranking_candidates_100_neg9.jsonl \
  --output data/processed/mf_item_ranking_predictions_100_neg9.json
```

---

## Ranking scripts

## build_ranking_candidates.py

Builds a ranking evaluation dataset.

Each ranking group contains:
- one positive target item
- several negative candidate items

The same user history is reused for all candidates in the group.

### Usage

```bash
python -m scripts.build_ranking_candidates \
  --input data/processed/yelp_subset_500_with_review_aspects_nli.jsonl \
  --output data/processed/yelp_ranking_candidates_100_neg9.jsonl \
  --num-examples 100 \
  --num-negatives 9
```

Output:
- `100` groups
- `10` candidates per group
- `1000` total candidate records

---

## evaluate_ranking.py

Evaluates ranking metrics from scored candidate records.

Supported inputs:
- DF-QuAD output files
- MF-only prediction files
- LLM-only direct score files

### DF-QuAD ranking evaluation

```bash
python -m scripts.evaluate_ranking \
  --input data/processed/dfquad_ranking_original_100_neg9_balanced.jsonl \
  --dataset data/processed/yelp_ranking_candidates_100_neg9.jsonl \
  --output-summary data/processed/evaluation_ranking_original_100_neg9_balanced_summary.json \
  --score-source dfquad \
  --score-key final_score \
  --k 1 3 5 10 \
  --require-full-groups
```

### MF-only ranking evaluation

```bash
python -m scripts.evaluate_ranking \
  --input data/processed/mf_item_ranking_predictions_100_neg9.json \
  --dataset data/processed/yelp_ranking_candidates_100_neg9.jsonl \
  --output-summary data/processed/evaluation_mf_ranking_100_neg9_summary.json \
  --score-source direct \
  --score-key score \
  --k 1 3 5 10 \
  --require-full-groups
```

### LLM-only ranking evaluation

```bash
python -m scripts.evaluate_ranking \
  --input data/processed/llm_direct_scores_gemini_flash_ranking_100_neg9.jsonl \
  --dataset data/processed/yelp_ranking_candidates_100_neg9.jsonl \
  --output-summary data/processed/evaluation_llm_direct_ranking_100_neg9_summary.json \
  --score-source direct \
  --score-key score \
  --k 1 3 5 10 \
  --require-full-groups
```

Metrics:
- HitRate@K
- NDCG@K
- MRR

---

## Evaluation scripts for prediction

## evaluate_dfquad_scores.py

Evaluates DF-QuAD output against normalized target ratings.

### Usage

```bash
python -m scripts.evaluate_dfquad_scores \
  --input data/processed/dfquad_original_500_unbalanced.jsonl \
  --dataset data/processed/yelp_subset_500_with_review_aspects_nli.jsonl \
  --output-csv data/processed/evaluation_dfquad_original_500_unbalanced.csv \
  --output-summary data/processed/evaluation_dfquad_original_500_unbalanced_summary.json
```

---

## evaluate_mf_predictions.py

Evaluates MF-only target-item predictions against normalized target ratings.

### Usage

```bash
python -m scripts.evaluate_mf_predictions \
  --predictions data/processed/mf_item_predictions_500.json \
  --dataset data/processed/yelp_subset_500_with_review_aspects_nli.jsonl \
  --output-csv data/processed/evaluation_mf_item_500.csv \
  --output-summary data/processed/evaluation_mf_item_500_summary.json
```

---

## evaluate_llm_direct_scores.py

Evaluates LLM-only direct scores against normalized target ratings.

### Usage

```bash
python -m scripts.evaluate_llm_direct_scores \
  --input data/processed/llm_direct_scores_gemini_flash_500.jsonl \
  --dataset data/processed/yelp_subset_500_with_review_aspects_nli.jsonl \
  --output-csv data/processed/evaluation_llm_direct_500.csv \
  --output-summary data/processed/evaluation_llm_direct_500_summary.json
```

## Unitary test, enabling precise visualisation

## inspect_jsonl.py

Inspects a JSONL file and displays examples with summary statistics.

### Usage

```bash
python -m scripts.inspect_jsonl \
  --file data/processed/yelp_subset_500_with_review_aspects_nli.jsonl \
  --n 3
```

## test_prompt.py

Builds and displays the LLM prompt for a given example.

### Usage

```bash
python -m scripts.test_prompt \
  --file data/processed/yelp_subset_500_with_review_aspects_nli.jsonl \
  --index 0
```

## test_generation.py

Runs local LLM-based argument generation on one JSONL example.

### Usage

```bash
python -m scripts.test_generation \
  --input data/processed/yelp_subset_500_with_review_aspects_nli.jsonl \
  --index 0
```

Use a different local Hugging Face model:

```bash
python -m scripts.test_generation \
  --input data/processed/yelp_subset_500_with_review_aspects_nli.jsonl \
  --index 3 \
  --model Qwen/Qwen2.5-3B-Instruct
```

## inspect_generation_results.py

Inspects JSONL files produced by batch generation and validation.

### Usage

```bash
python -m scripts.inspect_generation_results \
  --file data/processed/generated_arguments_gemini_flash_500_balanced.jsonl \
  --n 3
```

Show only invalid generations:

```bash
python -m scripts.inspect_generation_results \
  --file data/processed/generated_arguments_gemini_flash_500_balanced.jsonl \
  --only invalid \
  --n 3 \
  --show-prompt
```

## test_scoring.py

Tests argument scoring on a single example.

### Usage

```bash
python -m scripts.test_scoring \
  --input data/processed/yelp_subset_500_with_review_aspects_nli.jsonl \
  --results data/processed/generated_arguments_gemini_flash_500_balanced_valid.jsonl \
  --index 0
```

## test_dfquad.py

Tests DF-QuAD aggregation on a single example.

### Usage

```bash
python -m scripts.test_dfquad \
  --input data/processed/yelp_subset_500_with_review_aspects_nli.jsonl \
  --results data/processed/scored_arguments_gemini_flash_500_balanced.jsonl \
  --index 0
```

## test_graph.py

Displays an interactive argument graph.

### Usage

```bash
python -m scripts.test_graph \
  --input data/processed/yelp_subset_500_with_review_aspects_nli.jsonl \
  --results data/processed/scored_arguments_gemini_flash_500_balanced.jsonl \
  --index 0
```

## Notes

- Run scripts from the project root.
- Raw Yelp files should be placed in `data/raw/`.
- Processed outputs are stored in `data/processed/`.
- Sample files are available in `data/sample/`.
- Keep large generated outputs out of git.
- For Gemini, use moderate batch sizes to avoid quota errors.
