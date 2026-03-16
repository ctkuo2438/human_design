# Human Design RAG System

A multimodal Retrieval-Augmented Generation (RAG) system that parses Human Design BodyGraph chart images and generates personalized readings grounded in authoritative Human Design books.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Phase 1 — Offline Knowledge Base Construction](#phase-1--offline-knowledge-base-construction)
- [Phase 2 — Vision Extraction Lambda](#phase-2--vision-extraction-lambda)
- [Phase 3 — RAG Reading Generation Lambda](#phase-3--rag-reading-generation-lambda)
- [AWS Services Used](#aws-services-used)
- [Key Design Decisions](#key-design-decisions)
- [Technical Challenges & Solutions](#technical-challenges--solutions)
- [Next Steps](#next-steps)

---

## Overview

Human Design is a self-awareness system synthesizing astrology, I Ching, Kabbalah, and chakra principles. Each person has a unique **BodyGraph chart** that encodes their Type, Authority, Profile, defined Centers, and active Channels.

This system enables users to:

1. **Upload** a BodyGraph chart image
2. **Extract** chart properties via Claude's vision capabilities
3. **Retrieve** relevant passages from Human Design reference books
4. **Generate** a personalized reading grounded in authoritative sources

---

## Architecture

```
User uploads BodyGraph image
        │
        ▼
┌───────────────────────┐
│  Lambda: Vision       │  ← Anthropic Claude Sonnet (multimodal)
│  (human-design-vision)│
└──────────┬────────────┘
           │ chart_data (JSON)
           ▼
┌───────────────────────┐     ┌──────────────────┐
│  Lambda: RAG          │────▶│  S3 Bucket       │
│  (human-design-rag)   │     │  faiss_index.bin │
│  Docker Container     │     │  chunks.json     │
└──────────┬────────────┘     └──────────────────┘
           │
           ├─ FAISS similarity search (top 7 chunks)
           ├─ Anthropic Claude Sonnet API (generation)
           ▼
   Personalized Reading
```

---

## Phase 1 — Offline Knowledge Base Construction

### Source Data

Two authoritative Human Design PDF books uploaded to S3 bucket (`human-design-knowledge-base`):

- *The Definitive Book of Human Design* — Ra Uru Hu & Lynda Bunnell (70.2 MB)
- *Human Design: Discover the Person You Were Born to Be* — Chetan Parkyn (2.1 MB)

### Processing Pipeline

Built in a SageMaker Notebook (`ml.t3.large`, instance: `human-design-phase1`):

1. **Text Extraction** — PyMuPDF (installed via conda due to GCC incompatibility with pip)
2. **Chunking** — Fixed-size ~800 character chunks with sentence boundary alignment to avoid breaking mid-sentence
3. **Embedding** — `sentence-transformers/all-MiniLM-L6-v2` generates 384-dimensional vectors
4. **Indexing** — FAISS builds an in-memory vector index over ~2,393 chunks
5. **Storage** — `faiss_index.bin` (3.5 MB) and `chunks.json` (2.3 MB) uploaded to S3

### Services Used

S3, SageMaker Notebook, FAISS (open-source), sentence-transformers (open-source)

---

## Phase 2 — Vision Extraction Lambda

### Function: `human-design-vision`

A standard Lambda function (ZIP deployment) that accepts a base64-encoded BodyGraph chart image and extracts structured chart properties using Claude's vision capabilities.

### Input / Output

**Input:**
```json
{
  "image": "<base64-encoded image>",
  "media_type": "image/png"
}
```

**Output:**
```json
{
  "success": true,
  "chart_data": {
    "type": "Generator",
    "authority": "Emotional/Solar Plexus",
    "profile": "1/3",
    "strategy": "To Respond",
    "definition": "Single",
    "defined_centers": ["Throat", "G/Self", "Sacral", "Solar Plexus/Emotional", "Root"],
    "undefined_centers": ["Head", "Ajna", "Heart/Will/Ego", "Spleen"],
    "active_channels": ["14-2", "34-57", "27-50", "41-30"]
  }
}
```

### Implementation Details

- Uses `urllib` (not the Anthropic SDK) to call the Anthropic API, avoiding dependency packaging complexity
- Anthropic API key stored in Lambda environment variables
- Timeout: 30 seconds
- IAM Role: LabRole

### Services Used

Lambda, Anthropic Claude Sonnet API

---

## Phase 3 — RAG Reading Generation Lambda

### Function: `human-design-rag`

A Docker Container Lambda that performs RAG-based reading generation. Deployed via container image because `sentence-transformers` + `PyTorch` + `faiss-cpu` exceed Lambda's 250 MB ZIP limit.

### Supported Input Modes

**Mode 1 — Query only:**
```json
{
  "query": "What is a Generator type in Human Design?"
}
```

**Mode 2 — chart_data + query (personalized):**
```json
{
  "chart_data": {
    "type": "Generator",
    "authority": "Sacral",
    "profile": "4/6",
    "defined_centers": ["Sacral", "Throat", "Solar Plexus"],
    "active_channels": ["34-20"]
  },
  "query": "What does my type mean for my career?"
}
```

### Lambda Execution Flow

1. **Cold start:** Downloads `faiss_index.bin` and `chunks.json` from S3 to `/tmp`
2. **Model loading:** Loads FAISS index and pre-packaged sentence-transformers model (`/opt/models`) into memory
3. **Query vectorization:** Converts user query (or chart_data-derived query) into an embedding
4. **Retrieval:** FAISS returns the top 7 most relevant book passages
5. **Generation:** Sends chart_data + retrieved passages + user question to Anthropic Claude Sonnet API
6. **Response:** Claude generates a personalized reading using only the provided passages (prompt enforces no hallucination)

### Configuration

- **Memory:** 3,008 MB
- **Timeout:** 5 minutes
- **Cold start duration:** ~190 seconds (includes S3 download + model loading + API call)
- **Memory used:** ~1,089 MB of 3,008 MB allocated
- **Environment variables:** `ANTHROPIC_API_KEY`, `BUCKET_NAME`

### Build & Deployment

Built on EC2 (`t3.large`, instance: `docker-builder`) and pushed to ECR:

```bash
# Create ECR repository
aws ecr create-repository --repository-name human-design-rag --region us-east-1

# Authenticate Docker to ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin \
  626931581630.dkr.ecr.us-east-1.amazonaws.com

# Build image
docker build -t human-design-rag .

# Tag for ECR
docker tag human-design-rag:latest \
  626931581630.dkr.ecr.us-east-1.amazonaws.com/human-design-rag:latest

# Push to ECR
docker push \
  626931581630.dkr.ecr.us-east-1.amazonaws.com/human-design-rag:latest
```

Lambda function created pointing to the ECR image with LabRole.

### Docker Image Details

- **Base image:** `python:3.11-slim` + `awslambdaric` (AWS Lambda official base image had GLIBC 2.26, too old for dependencies)
- **Image size:** ~4.5 GB
- **Pre-packaged model:** `sentence-transformers/all-MiniLM-L6-v2` downloaded at build time to `/opt/models` (Lambda filesystem is read-only except `/tmp`)

### Services Used

EC2 (build), ECR (image registry), Lambda (Container), S3, FAISS, sentence-transformers, Anthropic Claude Sonnet API

---

## AWS Services Used

| Service | Purpose |
|---|---|
| **S3** | Store PDFs, FAISS index, chunks.json |
| **SageMaker Notebook** | PDF processing, chunking, embedding, FAISS indexing |
| **Lambda** | Vision extraction (ZIP) and RAG generation (Docker Container) |
| **EC2** | Docker image build environment |
| **ECR** | Docker image registry for container Lambda |
| **API Gateway** | *(Phase 4 — upcoming)* |

---

## Key Design Decisions

1. **FAISS over Pinecone / OpenSearch Serverless** — The knowledge base is small (~2,393 chunks). FAISS is free, lightweight, and sufficient for this scale. No need for managed vector database overhead.

2. **Anthropic API over AWS Bedrock** — AWS Academy Learner Lab does not support Amazon Bedrock. The architecture uses the Anthropic API directly with Claude Sonnet for both vision and text generation.

3. **Docker Container Lambda over ZIP** — `sentence-transformers` + `PyTorch` + `faiss-cpu` total over 1 GB, far exceeding Lambda's 250 MB ZIP limit. Docker container deployment supports up to 10 GB.

4. **`urllib` over Anthropic SDK** — Minimizes dependency packaging complexity in Lambda without meaningful functionality loss at this scale.

5. **`python:3.11-slim` + `awslambdaric` over AWS Lambda base image** — AWS Lambda official base image has GLIBC 2.26, which is too old for miniconda and several ML dependencies. Using `python:3.11-slim` with `awslambdaric` as the runtime interface client resolves GLIBC compatibility.

6. **Pre-downloaded embedding model** — sentence-transformers attempts to download models to the home directory at runtime, but Lambda's filesystem is read-only except `/tmp`. The model is pre-downloaded into `/opt/models` during Docker build.

---

## Technical Challenges & Solutions

| Challenge | Solution |
|---|---|
| AWS Academy Learner Lab does not support Bedrock | Pivoted to Anthropic API (Claude Sonnet) for vision and generation |
| Lambda 250 MB ZIP limit for ML dependencies | Docker Container deployment via ECR (10 GB limit) |
| GLIBC 2.26 on AWS Lambda base image too old | Used `python:3.11-slim` + `awslambdaric` as workaround |
| sentence-transformers runtime download on read-only filesystem | Pre-downloaded model to `/opt/models` at Docker build time |
| PyMuPDF pip install fails due to GCC incompatibility on SageMaker | Installed via conda instead |

---

## Next Steps

- **Phase 4:** API Gateway endpoint + Streamlit frontend integration
- **Phase 5:** CI/CD pipeline setup
