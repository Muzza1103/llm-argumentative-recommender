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
python scripts/build_yelp_subset.py
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
python scripts/inspect_jsonl.py --file data/processed/yelp_subset.jsonl --n 3
```

Example with sample data:

```bash
python scripts/inspect_jsonl.py --file data/examples/sample_input.jsonl --n 2
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
python scripts/test_prompt.py --file data/processed/yelp_subset.jsonl --index 0
```

Example with sample data:

```bash
python scripts/test_prompt.py --file data/examples/sample_input.jsonl --index 0
```

---

## Notes

- Run scripts from the project root
- Raw data should be placed in `data/raw/`
- Processed outputs are stored in `data/processed/`
- Sample files are available in `data/examples/`
