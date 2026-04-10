# Week 5 Status Update — Phase 5

---

## Phase 5: CI/CD Pipeline + Golden Dataset Evaluation

### What was completed

- Built a **Golden Dataset** of 10 Human Design BodyGraph chart images covering all 5 types: Generator (3), Manifestor (2), Projector (2), Reflector (3)
  - Images uploaded to S3 (`human-design-knowledge-base/golden-dataset/`)
  - Each test case includes: expected `chart_data` JSON, a test query, and expected key facts for RAG evaluation
  - Ground truth stored in `tests/golden_dataset.json`

- Built an automated **evaluation script** (`tests/evaluate.py`) with two test components:
  - **Vision test**: calls `/vision` API with each golden image, compares returned `chart_data` against ground truth using normalized field-by-field matching (exact match for strings, F1 score for lists)
  - **RAG test**: calls `/reading` API with golden `chart_data` + test query, then uses **LLM-as-Judge** (Claude Sonnet) to score the response on three dimensions (1-5 scale)

- Deployed **MLflow Tracking Server** (v3.1.4) on the existing Streamlit EC2 instance
  - Streamlit runs on port 8501, MLflow runs on port 5000 (no conflict)
  - Security Group opened for port 5000
  - Records prompt version (commit SHA), per-case scores, and aggregate metrics for each evaluation run

- Configured **GitHub Actions** workflow (`.github/workflows/evaluate.yml`):
  - Triggers automatically on push to `main` when `lambda/`, `prompts/`, or `tests/` files change
  - Also supports manual trigger via `workflow_dispatch`
  - Runs on GitHub-hosted Ubuntu runner with Python 3.11
  - Injects AWS credentials and API keys from GitHub Secrets

- Configured **GitHub Secrets** (7 total):
  - `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN` (need refresh each Learner Lab session)
  - `ANTHROPIC_API_KEY`
  - `API_BASE` (API Gateway invoke URL)
  - `MLFLOW_TRACKING_URI` (`http://<EC2-IP>:5000`, needs update if EC2 restarts)
  - `S3_BUCKET` (`human-design-knowledge-base`)

### Evaluation metrics

Four quantitative metrics are tracked per evaluation run:

| Metric | What it measures | Method | Scale |
|--------|-----------------|--------|-------|
| Vision Accuracy | Chart data extraction correctness | Field-by-field comparison with normalized matching | 0 – 1.0 |
| Faithfulness | Are all claims grounded in chart_data? No hallucinated channels/centers? | LLM-as-Judge (Claude Sonnet) | 1 – 5 |
| Relevance | Does the response address the user's question? | LLM-as-Judge (Claude Sonnet) | 1 – 5 |
| Completeness | How many expected key facts are covered? | LLM-as-Judge (Claude Sonnet) | 1 – 5 |

**LLM-as-Judge** is the current industry standard for evaluating LLM-generated text. The Judge does not verify Human Design knowledge correctness — it checks whether the response is consistent with the provided `chart_data` and covers the expected key facts.

If any metric falls below its threshold, the pipeline fails and deployment is blocked (red X in GitHub Actions).

### Test results

| Metric | Result | Threshold | Status |
|--------|--------|-----------|--------|
| Vision Accuracy | 0.29 | 0.80 | **FAIL** |
| Faithfulness | 4.90 | 3.0 | PASS |
| Relevance | 5.00 | 3.0 | PASS |
| Completeness | 4.20 | 3.0 | PASS |

Pipeline status: **FAILED** — Vision accuracy below threshold. RAG quality metrics all passed with high scores.

### Key findings

- **RAG quality is strong**: Faithfulness (4.9/5) and Relevance (5.0/5) indicate the RAG pipeline produces accurate, on-topic readings with minimal hallucination
- **Vision accuracy needs improvement**: Claude Vision API struggles with Reflector charts (all undefined centers), misidentifying them as Generator or other types with fabricated channels
- **Normalization helps but is insufficient**: Fuzzy matching for authority names (e.g., "Solar Plexus" = "Emotional/Solar Plexus") and channel ordering (e.g., "59-6" = "6-59") improved some scores, but the core Vision prompt needs refinement

### CI/CD pipeline flow

1. Developer pushes code/prompt changes to GitHub (`main` branch)
2. GitHub Actions automatically triggers the evaluation workflow
3. Workflow runs on GitHub-hosted Ubuntu runner, installs dependencies, configures AWS credentials from Secrets
4. `evaluate.py` loads Golden Dataset → calls Vision API (10 images) → calls RAG API (10 queries) → scores responses via LLM-as-Judge
5. All metrics logged to MLflow Tracking Server on EC2
6. If any metric below threshold → `sys.exit(1)` → GitHub Actions marks workflow as failed → deployment blocked

### Learner Lab limitations

- **AWS credentials expire every ~4 hours**: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and `AWS_SESSION_TOKEN` in GitHub Secrets must be manually updated each session. In production, this would be solved with IAM User permanent credentials or GitHub OIDC federation.
- **EC2 Public IP changes on restart**: `MLFLOW_TRACKING_URI` needs updating if EC2 is stopped/started. In production, an Elastic IP would provide a fixed address.
- **CodePipeline/CodeBuild unavailable**: Learner Lab IAM restrictions prevent creating service roles for AWS-native CI/CD services. GitHub Actions was used as the practical alternative.

### Future improvements

- Optimize Vision Lambda prompt with explicit output format constraints and Reflector-specific detection rules
- Add real-time guardrails: LLM-as-Judge check on each user request before displaying the response, with retry/fallback on low-quality outputs
- Implement multi-run averaging (run each test case 3x) to reduce LLM-as-Judge scoring variance
- Migrate Phase 3 to SageMaker Real-time Endpoint to eliminate ~190s cold starts for faster evaluation runs

**Services used:** GitHub Actions (CI/CD), MLflow (experiment tracking on EC2), S3 (Golden Dataset storage), Lambda (Vision + RAG under test), API Gateway, Anthropic Claude Sonnet API (LLM-as-Judge)
