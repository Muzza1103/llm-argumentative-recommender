# Scripts

This folder contains runnable utility scripts used throughout the project.

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

Example with sample data:

```bash
python -m scripts.inspect_jsonl --file data/examples/sample_input.jsonl --n 2
```

---

## test_prompt.py

Builds and displays the LLM prompt for a given example.

### Features
- Load an example from JSONL
- Format user history
- Format target item
- Generate the final prompt used for argument generation

### Usage

```bash
python -m scripts.test_prompt --file data/processed/yelp_subset.jsonl --index 0
```

Example with sample data:

```bash
python -m scripts.test_prompt --file data/examples/sample_input.jsonl --index 0
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

### Notes
- `--random` overrides the fixed example selection logic
- the script prints the actual example index used
- Run scripts from the project root
- Raw data should be placed in `data/raw/`
- Processed outputs are stored in `data/processed/`
- Sample files are available in `data/examples/`
