# LLM Argumentative Recommender

This project explores **explainable** and **contestable recommendation systems** using **local Large Language Models** and **argumentation-based reasoning**.

The main objective is to build a recommendation pipeline in which a language model generates structured arguments from user history and target item information, and these arguments are then used to support **explicit reasoning, scoring, and explanations**.

## Current objective

The current focus is on building a first prototype for:

- preprocessing a small Yelp-based subset
- generating arguments with a local LLM
- structuring these arguments into a support/attack graph
- preparing the ground for explicit reasoning and recommendation scoring

## Planned pipeline

```text
User history + target item
        ↓
LLM-based argument generation
        ↓
Validation and filtering
        ↓
Argument graph construction
        ↓
Argument scoring and reasoning / aggregation
        ↓
Explainable and contestable recommendation
```

## Project structure

```text
.
├── src/        
│   ├── llm/         # LLM loading, generation and validation
│   └── prompting/   # Prompt construction and formatting
├── scripts/         # Runnable scripts (data, prompt, generation)
├── configs/         # Configuration files
├── data/  
│   ├── raw/         # Original datasets (not versioned)
│   ├── processed/   # Generated datasets (not versioned)
│   └── sample/      # Small versioned examples
```

## Data

This repository does **not** include the full Yelp dataset.

- Raw data should be placed in `data/raw/`
- Processed subsets are generated locally in `data/processed/`
- Small synthetic examples are available in `data/sample/`

See `data/README.md` for more details on how to generate the dataset.

The dataset is built from the Yelp Open Dataset.

Each example contains:

- `history`: user past interactions
- `target_item`: item to evaluate

Data is stored in JSONL format.

## Inspecting Data

To inspect dataset samples:

```bash
python scripts/inspect_jsonl.py --file data/processed/yelp_subset.jsonl --n 3
```

---

## Prompting

The project includes a prompt generation module located in `src/prompting/`.

This module is responsible for:
- formatting user history and target items
- building structured prompts for LLM-based argument generation

### Input Design

The prompt uses:
- a **compact user history** (categories + ratings)
- a **filtered subset of item attributes**, including:
  - price range
  - takeout / delivery
  - seating / attire
  - noise level
  - group / kids suitability

This ensures:
- compact inputs (important for local LLMs)
- reduced noise
- better grounding of arguments
- fewer hallucinations

## Scripts

All runnable scripts are located in the `scripts/` directory.

These scripts cover:
- dataset preprocessing
- dataset inspection
- prompt generation
- local LLM-based argument generation
- batch generation and validation
- result inspection

For detailed usage and available commands, see:
`scripts/README.md`

---

## Status

Work in progress.  
This repository is part of a research / internship project on **LLM-based explainable recommendation using argumentation**.

## Notes

- The system is designed to work with **local LLMs**
- Quantization is supported for lightweight inference
- Future work includes:
  - argument scoring
  - integration with QBAF / DF-QuAD
  - potential LoRA fine-tuning

