# Scripts

This folder contains runnable utility scripts used throughout the project.

These scripts cover the full pipeline:
- dataset construction
- prompt generation
- argument generation
- validation
- scoring (LLM + MF)
- argument graph construction
- DF-QuAD aggregation
- visualization

---

## build_yelp_subset.py

Builds a processed subset from the Yelp Open Dataset.

### Responsibilities
- Load raw Yelp files
- Filter restaurant businesses
- Filter informative reviews
- Group reviews by user
- Construct examples with:
  - user history
  - target item (leave-one-out)
- Save the result as a JSONL file in `data/processed/`

### Usage

```bash
python -m scripts.build_yelp_subset
```

---

## inspect_jsonl.py

Inspects a JSONL file and displays examples along with summary statistics.

### Features
- Display first `n` examples
- Show history size
- Show target rating and review length
- Show sentiment distribution (positive / negative / neutral)

### Usage

```bash
python -m scripts.inspect_jsonl --file data/processed/yelp_subset.jsonl --n 3
```

---

## test_prompt.py

Builds and displays the LLM prompt for a given example.


### Usage

```bash
python -m scripts.test_prompt --file data/processed/yelp_subset.jsonl --index 0
```

---

## test_generation.py

Runs local LLM-based argument generation on one JSONL example.

### Features
- Load one example from a JSONL file
- Select an example by index or sample one randomly
- Format user history and target item
- Build the final generation prompt
- Load a local Hugging Face instruction model
- Generate structured recommendation arguments
- Display:
  - the selected example index
  - the final prompt
  - the raw model output
  - the parsed JSON output when valid

### Usage

Run on a specific example:

```bash
python -m scripts.test_generation --input data/processed/yelp_subset.jsonl --index 0
```

Run on a random example:

```bash
python -m scripts.test_generation --input data/processed/yelp_subset.jsonl --random
```

Use a different model:

```bash
python -m scripts.test_generation \
  --input data/processed/yelp_subset.jsonl \
  --index 3 \
  --model Qwen/Qwen2.5-3B-Instruct
```

### Main arguments

- `--input`: path to the input JSONL file
- `--index`: example index to load from the JSONL file
- `--random`: select a random example instead of using `--index`
- `--model`: Hugging Face model name
- `--max-new-tokens`: maximum number of generated tokens
- `--temperature`: generation temperature
- `--top-p`: top-p sampling parameter
- `--do-sample`: enable sampling during generation

## generate_batch.py

Runs LLM-based argument generation on multiple examples with validation.

### Features
- Load multiple examples from an input JSONL file
- Select examples sequentially or randomly
- Build a prompt for each example using the current prompting pipeline
- Run local argument generation with a Hugging Face model
- Parse the generated JSON output
- Validate each generated result
- Save:
  - all generated results
  - valid results only
  - invalid results only
  - a summary file with validation statistics and error counts

### Outputs

Given an output path such as:

```bash
data/processed/generated_arguments_batch.jsonl
```

the script also creates:

```
data/processed/generated_arguments_batch_valid.jsonl
data/processed/generated_arguments_batch_invalid.jsonl
data/processed/generated_arguments_batch_summary.json
```

### Usage

Run generation on the first 10 examples:

```bash
python -m scripts.generate_batch --input data/processed/yelp_subset.jsonl --output data/processed/generated_arguments_batch.jsonl --num-examples 10 --save-prompt
```



## inspect_generation_results.py

Inspects JSONL files produced by batch generation and validation.

### Features
- Load generation result files in JSONL format
- Display the first `n` records
- Filter displayed records by validation status: (all, valid, invalid)
- Show, for each record:
  - dataset index
  - user id
  - target item name
  - validation status
  - validation errors
  - parsed JSON output
- Optionally display:
  - the full prompt
  - the raw model output

### Usage

Display the first 3 records from a batch result file:

```bash
python -m scripts.inspect_generation_results --file data/processed/generated_arguments_batch.jsonl --n 3
```

Display only invalid records:

```bash
python -m scripts.inspect_generation_results --file data/processed/generated_arguments_batch.jsonl --only invalid --n 3
```

Display invalid records with the full prompt:

```bash
python -m scripts.inspect_generation_results --file data/processed/generated_arguments_batch.jsonl --only invalid --n 2 --show-prompt
```

## test_scoring.py

Tests scoring on a single example.

### Features
- LLM scoring
- MF scoring
- combined score
- optional prompt + raw

### Usage

```bash
python -m scripts.test_scoring \
  --input data/processed/yelp_subset.jsonl \
  --results data/processed/generated_arguments_batch.jsonl \
  --index 0
```

---

## score_batch.py

Runs scoring on multiple generated examples.

### Features
- LLM + MF scoring
- save enriched JSONL
- generate summary stats

### Usage

```bash
python -m scripts.score_batch \
  --dataset data/processed/yelp_subset.jsonl \
  --input data/processed/generated_arguments_batch.jsonl \
  --output data/processed/scored_arguments_batch.jsonl
```

---

## build_mf_dataset.py

Builds a dataset for Matrix Factorization.

### Output
- `mf_dataset.csv`

### Usage

```bash
python -m scripts.build_mf_dataset \
  --input data/processed/yelp_subset.jsonl \
  --output data/processed/mf_dataset.csv
```

---

## train_mf.py

Trains an MF model (SVD) and generates predictions.

### Output
- `mf_predictions.json`

### Usage

```bash
python -m scripts.train_mf \
  --mf-data data/processed/mf_dataset.csv \
  --source-dataset data/processed/yelp_subset.jsonl \
  --output data/processed/mf_predictions.json
```

---

## test_dfquad.py

Tests DF-QuAD aggregation.

### Features
- build argument graph
- compute aggregated scores
- optional graph display

### Usage

```bash
python -m scripts.test_dfquad \
  --input data/processed/yelp_subset.jsonl \
  --results data/processed/scored_arguments_batch.jsonl \
  --index 0
```

---

## test_graph.py

Displays an interactive argument graph.

### Features
- nodes: arguments + item
- edges: support / attack
- zoom + hover
- display history + attributes

### Usage

```bash
python -m scripts.test_graph \
  --input data/processed/yelp_subset.jsonl \
  --results data/processed/scored_arguments_batch.jsonl \
  --index 0
```

---

### Notes
- `--random` overrides the fixed example selection logic
- the script prints the actual example index used
- Run scripts from the project root
- Raw data should be placed in `data/raw/`
- Processed outputs are stored in `data/processed/`
- Sample files are available in `data/examples/`
