# LLM Argumentative Recommender

This project explores **explainable** and **contestable recommendation systems** using **Large Language Models (LLMs)**, **collaborative filtering** and **argumentation-based reasoning**.

The main objective is to build a recommendation pipeline in which a language model generates structured arguments from user history and target item information, and these arguments are then used to support **explicit reasoning, scoring, and explanations**.


## Current pipeline

The current system includes:
- Yelp subset construction
- aspect extraction from reviews
- aspect-based MF dataset construction
- aspect-based MF training
- LLM-based argument generation
- balanced (same number of support and attack arguments) and unbalanced argument generation settings
- argument validation
- hybrid semantic + collaborative scoring
- argument graph construction
- DF-QuAD and contrastive aggregation variants
- rating prediction evaluation
- ranking evaluation
- MF-only and LLM-only baselines

## Recommendation pipeline

```text
User history + target item
        ↓
LLM-based argument generation
        ↓
Validation and filtering
        ↓
Argument scoring
   - semantic score (LLM)
   - empirical score (collaborative filtering / aspect-MF)
        ↓
Argument graph construction
        ↓
Argument aggregation
   - DF-QuAD
   - contrastive variants
        ↓
Explainable and contestable recommendation
```

## Project structure

```text
.
├── src/
│   ├── llm/             # LLM loading, generation and scoring
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

---
## Aspect Extraction and Collaborative Signals

Restaurant ratings alone often provide only coarse preference information.
Two users may assign the same global rating to a restaurant for very different reasons:
one may value food quality while another may care more about service, atmosphere or price.

To obtain finer-grained collaborative signals, the project extracts review aspects from user reviews.

### NLI-based aspect extraction

Aspect extraction is performed using Natural Language Inference (NLI) models.

Each review is compared against a predefined aspect vocabulary containing dimensions such as:
- food
- service
- price
- ambiance
- cleanliness
- location

For each review-aspect pair, the NLI model estimates whether the review semantically entails the presence of the aspect.

The extracted aspects are then used to build structured user-aspect interaction signals.

### Aspect-based collaborative filtering

The extracted aspects are transformed into collaborative learning signals.

Instead of learning only:
- user-item interactions

the system also learns:
- user-aspect affinities

using Matrix Factorization (MF).

This allows the system to estimate:
- which aspects are important for a given user
- how strongly a target item matches these aspects

These aspect-level collaborative signals are then reused during argumentative scoring as empirical evidence complementary to semantic LLM reasoning.

---

## Argument Scoring and Reasoning

The project includes a hybrid argumentative scoring pipeline combining:
- semantic LLM-based reasoning
- collaborative recommendation signals
- formal argumentative aggregation

### Semantic scoring

Each generated argument receives a semantic score from a local or API-based LLM scorer.

This score is intended to reflect:
- coherence with the user history
- compatibility with the target item
- quality and usefulness of the argument
- consistency of the provided evidence
- grounding in the provided context

### Empirical scoring

The project also includes an empirical scoring component.

This empirical signal can be obtained from:
- user-item collaborative filtering
- aspect-based Matrix Factorization
- fallback heuristics

The empirical score is combined with the semantic score in order to integrate collaborative recommendation signals into the argumentative framework.

### Argument graph construction

Generated arguments are structured into a graph in which:
- support arguments strengthen the recommendation claim
- attack arguments weaken the recommendation claim

This graph-based structure enables explicit reasoning over conflicting recommendation evidence.

### DF-QuAD aggregation

DF-QuAD is a gradual argumentation semantics used to propagate support and attack relations through the argument graph.

The objective is to compute a final recommendation score while explicitly accounting for:
- supporting evidence
- attacking evidence
- interaction between arguments

This allows the recommendation process to remain interpretable and contestable.

### Contrastive aggregation variants

The project also explores contrastive aggregation variants.

These variants aim to amplify the separation between support and attack signals during aggregation in order to produce sharper recommendation decisions.

The contrastive parameter $\gamma$ controls how strongly support and attack differences are amplified.

### Balanced vs unbalanced generation

The project explores both balanced and unbalanced argument generation settings.

Balanced generation enforces equal numbers of support and attack arguments.

Unbalanced generation allows the LLM to generate arguments more freely depending on the recommendation context.

This setting is studied in order to analyze how attack/support distributions affect recommendation quality and argumentative behavior.

---

## Evaluation

The project currently evaluates two recommendation tasks:
- rating prediction
- ranking recommendation

### Rating prediction

Rating prediction evaluates how accurately the system predicts the target user preference score.

Predictions are normalized to the range:
- [0, 1]

The following metrics are used:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)

### Ranking recommendation

The ranking task evaluates whether the system can correctly identify the positive target item among several negative candidates.

The current ranking setup uses:
- 1 positive target item
- multiple negative candidate items

The following ranking metrics are used:
- HitRate@K
- NDCG@K
- MRR

### Experimental observations

Current experiments indicate that:
- MF-only baselines achieve the strongest predictive performance
- argumentative variants outperform direct LLM-only scoring in several rating prediction settings
- unbalanced argument generation improves DF-QuAD performance over balanced generation
- preserving collaborative recommendation signals during argumentative aggregation remains a key challenge

These results highlight the presence of a trade-off between:
- predictive performance
- explainability
- contestability

---

## Baselines

The project currently includes several baseline systems used for comparison.

### MF-only baseline

A standard collaborative filtering baseline based on Matrix Factorization (SVD) is used for:
- rating prediction
- ranking recommendation

This baseline directly predicts user-item compatibility scores without argumentative reasoning.

### LLM-only baseline

A direct LLM-based scoring baseline is also evaluated.

In this setting, the LLM directly predicts a recommendation score from:
- the user history
- the target item

without generating argumentative structures.

### Argumentative variants

The main proposed system evaluates multiple argumentative variants, including:
- DF-QuAD aggregation
- contrastive aggregation variants
- balanced argument generation
- unbalanced argument generation

## Notes

- The system is designed to work with local or API-based LLMs
- Quantization is supported for lightweight inference
- Future work includes:
  - improving argument generation
  - grounding arguments more explicitly in attributes and categories
  - refining empirical scoring beyond a single item-level score
  - exploring richer argumentative graph structures
  - potential LoRA fine-tuning

