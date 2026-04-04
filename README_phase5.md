# Phase 5 (Minimal CI + Evaluation)

This repository now includes a deliberately reduced-scope **Phase 5** evaluation system for the Human Design AWS stack.

## What this minimal Phase 5 does

- Adds a 5-sample golden dataset for Vision extraction checks.
- Adds a 5-sample golden dataset for RAG reading checks.
- Runs two simple script-based evaluations (no pytest framework).
- Computes lightweight metrics and writes them to `reports/latest_metrics.json`.
- Adds GitHub Actions CI to run evaluations on every push and pull request.
- Fails CI when metrics fall below thresholds.

## Dataset format

### Vision dataset: `datasets/vision_golden.json`

Each sample includes:

- `id`
- `image_path`
- `expected_chart_data`
  - `type`
  - `authority`
  - `profile`
  - `defined_centers`
  - `active_channels`

### RAG dataset: `datasets/rag_golden.json`

Supports two forms:

1. Query only:

- `id`
- `query`
- `expected_keywords`

2. Chart data + query:

- `id`
- `chart_data`
- `query`
- `expected_keywords`

Keyword matching is case-insensitive and whitespace-normalized via simple substring matching.

## Required environment variables

### Vision script

- `VISION_API_URL` (API Gateway endpoint URL, e.g. `https://.../vision`)

### RAG script

- `RAG_LAMBDA_NAME` (Lambda function name)
- `AWS_REGION` (defaults to `us-east-1` if not set)
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_SESSION_TOKEN` (optional; include if your lab credentials provide it)

## Local usage

Install dependencies:

```bash
pip install -r requirements.txt
```

Run Vision evaluation:

```bash
python tests/test_vision.py
```

Run RAG evaluation:

```bash
python tests/test_rag.py
```

## CI thresholds

- Vision: `vision_average_score >= 0.80`
- RAG: `rag_success_rate == 1.0`
- RAG: `rag_average_keyword_coverage >= 0.60`

## Important notes / assumptions

- Vision evaluation calls API Gateway (`VISION_API_URL`).
- RAG evaluation uses **direct Lambda invoke** via boto3 because API Gateway can time out for long cold starts.
- The sample data in `datasets/vision_golden.json` is a placeholder starter set.
- **TODO:** replace placeholder image files in `datasets/images/` with real BodyGraph images and update corresponding expected chart data values for reliable production metrics.
- Scripts are intentionally simple and easy to extend later (for example, expanding from 5 to 20 samples).
