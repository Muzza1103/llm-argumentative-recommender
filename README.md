# LLM Argumentative Recommender

This project explores **explainable** and **contestable** recommendation using **local LLMs** and **argumentation-based reasoning**.

The main objective is to build a recommendation pipeline in which a language model generates structured arguments from user history and target item information, and these arguments are then used to support explicit reasoning and explanations.

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
Argument graph construction
        ↓
Reasoning / aggregation
        ↓
Explainable and contestable recommendation
```

## Project structure

```text
.
├── src/        # Core project logic
├── scripts/    # Runnable scripts
├── configs/    # Configuration files
├── data/       # Data instructions, samples, and local dataset structure
```

## Data

This repository does **not** include the full Yelp dataset.

- Raw data should be placed in `data/raw/`
- Processed subsets are generated locally in `data/processed/`
- Small synthetic examples are available in `data/sample/`

See `data/README.md` for more details.

## Status

Work in progress.  
This repository is part of a research / internship project on LLM-based explainable recommendation through argumentation.

## Notes

- The repository is designed to work with **local models**
