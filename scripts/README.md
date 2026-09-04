# Scripts

This directory contains the executable scripts for the two recommendation pipelines implemented in the project.

Run commands from the project root so that Python can resolve the `scripts` and `src` packages correctly.

## Directory organization

```text
scripts/
├── README.md
├── inspect_jsonl.py
├── restaurants/
│   ├── README.md
│   └── ...
└── hotel/
    ├── README.md
    └── ...
```

## Restaurant recommendation

The [`restaurants`](restaurants/) directory contains the scripts associated with the Yelp and CoRAL pipeline. They cover:

- Yelp subset construction;
- aspect extraction from reviews;
- user–aspect and user–restaurant matrix factorization;
- support and attack argument generation;
- argument validation and strength computation;
- DF-QuAD aggregation;
- rating-prediction and ranking evaluation;
- MF-only and LLM-only baselines;
- result inspection and graph rendering.

See [`restaurants/README.md`](restaurants/README.md) for the pipeline order, the role of each script and the main commands.

## Hotel recommendation

The [`hotel`](hotel/) directory contains the scripts associated with the cold-start hotel recommendation pipeline. They cover:

- validation of prepared hotel profiles;
- individual hotel evaluation;
- HTML argument-graph rendering;
- multi-hotel ranking;
- structured preference contestation;
- facility inventory and data-conversion utilities.

See [`hotel/README.md`](hotel/README.md) for the expected inputs and the main commands.

## Shared utility

`inspect_jsonl.py` displays records from a JSONL file and can be used to inspect intermediate outputs from either pipeline.

```bash
python -m scripts.inspect_jsonl --file PATH/TO/FILE.jsonl --n 3
```

## LLM configuration

The final experiments use Gemini 2.5 Flash through Google Cloud and Vertex AI. Earlier local exploration used Qwen2.5-3B-Instruct.

For Gemini-based execution, authenticate and select the appropriate Google Cloud project before running the scripts:

```bash
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
gcloud auth application-default set-quota-project YOUR_PROJECT_ID
```

Credentials, private datasets and generated experimental outputs are not included in the repository.

