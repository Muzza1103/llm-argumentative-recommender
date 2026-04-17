# LLM Argumentative Recommender

This project explores **explainable** and **contestable recommendation systems** using **local Large Language Models** and **argumentation-based reasoning**.

The main objective is to build a recommendation pipeline in which a language model generates structured arguments from user history and target item information, and these arguments are then used to support **explicit reasoning, scoring, and explanations**.

## Current objective

The current focus is on building a first prototype for:

- preprocessing a small Yelp-based subset
- generating structured recommendation arguments with a local LLM
- validating and filtering generated arguments
- assigning a semantic score to each argument with an LLM-based scorer
- assigning a first empirical score based on collaborative recommendation signals
- structuring arguments into a support / attack graph
- aggregating argument strengths with a first **DF-QuAD-based reasoning step**

## Planned pipeline

```text
User history + target item
        ↓
LLM-based argument generation
        ↓
Validation and filtering
        ↓
Argument scoring
   - semantic score (LLM)
   - empirical score (MF / fallback)
        ↓
Argument graph construction
        ↓
DF-QuAD aggregation
        ↓
Explainable and contestable recommendation
```

## Project structure

```text
.
├── src/
│   ├── llm/             # LLM loading, generation and local scoring
│   ├── prompting/       # Prompt construction and formatting
│   └── argumentation/   # Argument schema, scoring, graph construction and DF-QuAD aggregation
├── scripts/             # Runnable scripts for data, generation, scoring and inspection
├── configs/             # Configuration files
├── data/
│   ├── raw/             # Original datasets (not versioned)
│   ├── processed/       # Generated datasets and intermediate outputs (not versioned)
│   └── sample/          # Small versioned examples
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
- filtering item attributes to keep only the most useful fields
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

## Argument Scoring and Reasoning

The project currently includes a first hybrid scoring pipeline for generated arguments.

### Semantic scoring

Each generated argument can receive a **semantic score** from a local LLM-based scorer.

This score is intended to reflect:
- coherence with the user history
- compatibility with the target item
- quality and usefulness of the argument
- consistency of the provided evidence


### Empirical scoring

The project also includes a first **empirical scoring** component.

At the current stage, this score is computed at the **user-item level** and can be obtained from:
- a fallback heuristic
- or precomputed Matrix Factorization predictions

This empirical signal is then combined with the semantic score.

### Argument graph and aggregation

Generated and scored arguments are structured into a graph in which:
- **support** arguments strengthen the recommendation claim
- **attack** arguments weaken the recommendation claim

A first **DF-QuAD-based aggregation** step is then applied in order to obtain a structured final score instead of relying on simple averaging.


## Scripts

All runnable scripts are located in the `scripts/` directory.

These scripts cover:
- dataset preprocessing
- dataset inspection
- prompt generation
- local LLM-based argument generation
- batch generation and validation
- scoring
- MF dataset preparation and prediction
- graph construction and DF-QuAD testing
- interactive graph visualization

For detailed usage and available commands, see:
`scripts/README.md`

---

## Status

Work in progress.  
This repository is part of a research / internship project on **LLM-based explainable recommendation using argumentation**.

A first end-to-end prototype is now available, including:
- dataset construction
- argument generation
- validation
- semantic scoring
- first empirical scoring
- argument graph construction
- DF-QuAD aggregation
- interactive graph inspection

## Notes

- The system is designed to work with **local LLMs**
- Quantization is supported for lightweight inference
- The current empirical scorer is a **first prototype**, and finer argument-level empirical scoring remains future work
- Future work includes:
  - improving argument generation
  - grounding arguments more explicitly in attributes and categories
  - refining empirical scoring beyond a single item-level score
  - exploring richer argumentative graph structures
  - potential LoRA fine-tuning

