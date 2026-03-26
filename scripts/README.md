# Scripts

This folder contains runnable utility scripts used throughout the project.

## Current scripts

### `build_yelp_subset.py`
Builds a small processed subset from the Yelp Open Dataset.

Main responsibilities:
- load raw Yelp files
- filter restaurant businesses
- filter informative reviews
- construct user history / target item examples
- save the result as a JSONL file in `data/processed/`

Example usage:

```bash
python scripts/build_yelp_subset.py
```

### `inspect_jsonl.py`
Inspects a JSONL file and displays a few formatted examples.

Main responsibilities:
- load a JSONL file
- display the first `n` examples
- show lightweight summary information for quick debugging

Example usage:

```bash
python scripts/inspect_jsonl.py --file data/processed/yelp_subset.jsonl --n 3
```

Example on the synthetic sample:

```bash
python scripts/inspect_jsonl.py --file data/sample/sample_input.jsonl --n 2
```

## Purpose

These scripts are meant to support:
- dataset preprocessing
- quick inspection of intermediate files
- debugging
- reproducible experimentation

## Notes

- Raw Yelp data is expected in `data/raw/`
- Generated outputs are typically stored in `data/processed/`
- Small synthetic examples can be stored in `data/sample/`
