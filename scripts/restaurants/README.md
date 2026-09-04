# Restaurant scripts

This directory contains the scripts used by CoRAL, the argumentative restaurant recommendation pipeline evaluated on Yelp.

CoRAL uses a user's past ratings and reviews to generate support and attack arguments about a target restaurant. Argument strengths combine a semantic score produced by an LLM with an empirical user–aspect signal learned by Aspect-MF. The resulting arguments are aggregated with DF-QuAD.

Run all commands from the project root.

## Pipeline overview

```text
Yelp data
   → subset construction
   → aspect extraction
   → Aspect-MF training
   → argument generation and validation
   → semantic and empirical strength computation
   → DF-QuAD aggregation
   → rating or ranking evaluation
```

## Main scripts

### Data preparation

| Script | Role |
|---|---|
| `build_yelp_subset.py` | Builds leave-one-out user histories and target restaurants from the Yelp Open Dataset. |
| `extract_review_aspects_nli.py` | Detects review aspects with an NLI model and the configured aspect vocabulary. |
| `build_aspect_mf_dataset.py` | Converts detected aspects into a user–aspect–rating dataset. |
| `train_aspect_mf.py` | Trains Aspect-MF and exports user–aspect predictions. |
| `build_mf_dataset.py` | Builds the user–restaurant dataset used by the MF-only baseline. |
| `train_mf.py` | Trains matrix factorization and predicts scores for target restaurants. |
| `build_ranking_candidates.py` | Creates ranking groups containing one held-out target and comparison candidates. |

### Argumentative recommendation

| Script | Role |
|---|---|
| `generate_batch.py` | Generates support and attack arguments, validates their structure and separates valid and invalid outputs. |
| `score_batch.py` | Computes semantic and empirical components and assigns a combined strength to each valid argument. |
| `dfquad_batch.py` | Builds the flat bipolar argument graph and computes the recommendation score. |

### Baselines and evaluation

| Script | Role |
|---|---|
| `llm_direct_score.py` | Produces the LLM-only score without generating arguments. |
| `evaluate_dfquad_scores.py` | Evaluates argumentative rating predictions with MAE and RMSE. |
| `evaluate_mf_predictions.py` | Evaluates MF-only rating predictions. |
| `evaluate_llm_direct_scores.py` | Evaluates LLM-only rating predictions. |
| `evaluate_ranking.py` | Computes Hit Rate, MRR and NDCG for ranking outputs. |

### Inspection and visualization

| Script | Role |
|---|---|
| `inspect_prompt.py` | Displays the argument-generation prompt for one example. |
| `run_generation_example.py` | Runs argument generation on one example. |
| `inspect_generation_results.py` | Inspects valid or rejected generated outputs. |
| `inspect_scoring.py` | Displays the strength computation for one example. |
| `inspect_dfquad.py` | Inspects DF-QuAD inputs and results. |
| `render_graph.py` | Produces a visual representation of an argumentative result. |
| `explore_review_aspects.py` | Summarizes extracted review aspects. |
| `analyze_mf_fallback_aspects.py` | Diagnoses missing Aspect-MF predictions and fallback cases. |

Additional exploratory or compatibility scripts remain available in the directory. Use `python -m MODULE --help` to inspect their current options.

## 1. Build the Yelp subset

Place the following Yelp source files in `data/raw/`:

```text
data/raw/yelp_academic_dataset_business.json
data/raw/yelp_academic_dataset_review.json
```

Then run:

```bash
python -m scripts.restaurants.build_yelp_subset \
  --output data/processed/yelp_subset_500.jsonl \
  --nb-users 500 \
  --history-size 5 \
  --min-user-reviews 6 \
  --seed 42
```

Each resulting record contains a user history and one held-out target restaurant.

## 2. Extract review aspects

```bash
python -m scripts.restaurants.extract_review_aspects_nli \
  --input data/processed/yelp_subset_500.jsonl \
  --output data/processed/yelp_subset_500_with_review_aspects_nli.jsonl \
  --aspect-vocab configs/aspect_vocabulary.json \
  --aspect-threshold 0.45
```

## 3. Train Aspect-MF

```bash
python -m scripts.restaurants.build_aspect_mf_dataset \
  --input data/processed/yelp_subset_500_with_review_aspects_nli.jsonl \
  --output data/processed/aspect_mf_dataset_nli_500.csv
```

```bash
python -m scripts.restaurants.train_aspect_mf \
  --input data/processed/aspect_mf_dataset_nli_500.csv \
  --output data/processed/aspect_mf_predictions_nli_500.json
```

## 4. Run the argumentative rating pipeline

The provided shell script runs argument generation, argument scoring, DF-QuAD and rating evaluation in sequence:

```bash
bash scripts/restaurants/pipelines/run_rating_pipeline.sh YOUR_PROJECT_ID
```

The same stages can be run individually. The main generation command is:

```bash
python -m scripts.restaurants.generate_batch \
  --input data/processed/yelp_subset_500_with_review_aspects_nli.jsonl \
  --output data/processed/generated_arguments_gemini_flash_500_unbalanced.jsonl \
  --backend gemini \
  --gemini-model gemini-2.5-flash \
  --gcp-project YOUR_PROJECT_ID \
  --gcp-location global \
  --batch-size 5 \
  --num-examples 500 \
  --argument-mode unbalanced
```

Generation creates valid, invalid and summary files next to the requested output. Only validated arguments should be passed to the scoring stage.

```bash
python -m scripts.restaurants.score_batch \
  --dataset data/processed/yelp_subset_500_with_review_aspects_nli.jsonl \
  --input data/processed/generated_arguments_gemini_flash_500_unbalanced_valid.jsonl \
  --output data/processed/scored_arguments_gemini_flash_500_unbalanced.jsonl \
  --mf-predictions data/processed/aspect_mf_predictions_nli_500.json \
  --backend gemini \
  --gemini-model gemini-2.5-flash \
  --gcp-project YOUR_PROJECT_ID \
  --gcp-location global \
  --batch-size 5
```

```bash
python -m scripts.restaurants.dfquad_batch \
  --input data/processed/scored_arguments_gemini_flash_500_unbalanced.jsonl \
  --output data/processed/dfquad_original_500_unbalanced.jsonl \
  --dataset data/processed/yelp_subset_500_with_review_aspects_nli.jsonl \
  --aggregation-method dfquad \
  --combination-method dfquad \
  --save-graph
```

```bash
python -m scripts.restaurants.evaluate_dfquad_scores \
  --input data/processed/dfquad_original_500_unbalanced.jsonl \
  --dataset data/processed/yelp_subset_500_with_review_aspects_nli.jsonl \
  --output-csv data/processed/evaluation_dfquad_original_500_unbalanced.csv \
  --output-summary data/processed/evaluation_dfquad_original_500_unbalanced_summary.json
```

## 5. Run the ranking pipeline

Create ranking candidates before running the complete ranking workflow:

```bash
python -m scripts.restaurants.build_ranking_candidates \
  --input data/processed/yelp_subset_500_with_review_aspects_nli.jsonl \
  --output data/processed/yelp_ranking_candidates_100_neg9.jsonl \
  --num-examples 100 \
  --num-negatives 9 \
  --candidate-source history \
  --min-candidate-occurrences 1 \
  --require-positive-seen
```

```bash
bash scripts/restaurants/pipelines/run_ranking_pipeline.sh YOUR_PROJECT_ID
```

Each group contains one held-out target restaurant and nine comparison candidates. The evaluation measures whether the target is recovered near the top of the ranking; comparison candidates are not explicit user rejections.

## Baselines

The complete MF-only and LLM-only workflows can be launched with:

```bash
bash scripts/restaurants/pipelines/run_baseline_mf.sh YOUR_PROJECT_ID
```

```bash
bash scripts/restaurants/pipelines/run_baseline_llm.sh YOUR_PROJECT_ID
```

## Local model support

Scripts exposing a `--backend` option can also use a local Hugging Face model. Qwen2.5-3B-Instruct was used during early exploratory work:

```bash
python -m scripts.restaurants.run_generation_example \
  --input data/processed/yelp_subset_500_with_review_aspects_nli.jsonl \
  --index 0 \
  --model Qwen/Qwen2.5-3B-Instruct
```

## Notes

- The full Yelp dataset and large generated outputs are not versioned.
- Use moderate Gemini batch sizes to reduce quota and transient API errors.
- LLM outputs that fail schema or reference validation are rejected rather than included in the argumentative calculation.
- File names in the examples reflect the experiments reported in the internship work and can be changed as long as subsequent stages use the corresponding paths.
