import json
import os
import urllib.request
import numpy as np
import faiss
import boto3
from sentence_transformers import SentenceTransformer

# Config
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
BUCKET_NAME = os.environ.get("BUCKET_NAME", "human-design-knowledge-base")
INDEX_PREFIX = "index/"
TOP_K = 7 # number of chunks to retrieve

# Global variables - loaded once when Lambda cold starts
model = None
index = None
chunks = None


def load_resources():
    """
    Load FAISS index, chunks.json, and embedding model.
    Downloads from S3 to /tmp (Lambda's writable directory),
    then loads into memory. Only runs once per cold start.
    """
    global model, index, chunks

    if model is not None:
        return # already loaded

    s3 = boto3.client("s3")

    # Download FAISS index from S3 to /tmp
    # /tmp is the only writable directory in Lambda, max 10GB
    s3.download_file(BUCKET_NAME, f"{INDEX_PREFIX}faiss_index.bin", "/tmp/faiss_index.bin")
    index = faiss.read_index("/tmp/faiss_index.bin")

    # Download chunks.json from S3 to /tmp
    s3.download_file(BUCKET_NAME, f"{INDEX_PREFIX}chunks.json", "/tmp/chunks.json")
    with open("/tmp/chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)

    # Load sentence-transformers model
    # First invocation will be slow (~10s), subsequent ones use cached model
    model = SentenceTransformer("all-MiniLM-L6-v2")


def search_knowledge_base(query, top_k=TOP_K):
    """
    Convert query to vector, search FAISS index, return top-k relevant chunks.
    This is the 'Retrieve' step of RAG.
    """
    # Encode query to vector (same model used to build the index in Phase 1)
    query_vec = model.encode([query])
    faiss.normalize_L2(query_vec)

    # Search FAISS index for nearest neighbors
    distances, indices = index.search(query_vec, top_k)

    results = []
    for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        if idx < len(chunks):
            results.append({
                "rank": i + 1,
                "score": float(1 - dist),
                "text": chunks[idx]["text"],
                "source": chunks[idx]["source"]
            })
    return results


def build_query_from_chart(chart_data):
    """
    Convert chart_data JSON into a natural language query for FAISS search.
    e.g., "Generator type with Emotional/Solar Plexus authority, profile 4/6,
    defined centers: Sacral, Solar Plexus, Throat..."
    """
    parts = []
    if chart_data.get("type"):
        parts.append(f"{chart_data['type']} type in Human Design")
    if chart_data.get("authority"):
        parts.append(f"with {chart_data['authority']} authority")
    if chart_data.get("profile"):
        parts.append(f"profile {chart_data['profile']}")
    if chart_data.get("defined_centers"):
        parts.append(f"defined centers: {', '.join(chart_data['defined_centers'])}")
    if chart_data.get("active_channels"):
        parts.append(f"active channels: {', '.join(chart_data['active_channels'])}")
    return ". ".join(parts)


def call_anthropic_for_reading(chart_data, retrieved_chunks, user_query=None):
    """
    Send chart data + retrieved knowledge base chunks to Anthropic API.
    This is the 'Generate' step of RAG.
    Claude generates a personalized reading grounded in the book passages.
    """
    # Format retrieved chunks as context
    context = "\n\n---\n\n".join([
        f"[Source: {c['source']}]\n{c['text']}" for c in retrieved_chunks
    ])

    # Build the prompt
    system_prompt = """You are an expert Human Design analyst. 
Your task is to provide a personalized Human Design reading based ONLY on the provided reference passages from authoritative Human Design books.
Rules:
- Only use information from the provided passages
- Be specific and personal based on the chart data
- Explain concepts clearly for someone new to Human Design
- Structure the reading with clear sections
- Do not invent or hallucinate information not in the passages"""

    user_message = f"""Here is the person's Human Design chart data:
{json.dumps(chart_data, indent=2)}

Here are relevant passages from Human Design reference books:
{context}

"""
    if user_query:
        user_message += f"The person's specific question: {user_query}\n\n"
        user_message += "Please answer their question based on their chart data and the reference passages above."
    else:
        user_message += "Please provide a comprehensive personalized Human Design reading based on their chart data and the reference passages above."

    payload = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 3000,
        "system": system_prompt,
        "messages": [
            {"role": "user", "content": user_message}
        ],
    }

    headers = {
        "Content-Type": "application/json",
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01"
    }

    req = urllib.request.Request(
        ANTHROPIC_API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST"
    )

    with urllib.request.urlopen(req, timeout=60) as resp:
        result = json.loads(resp.read().decode("utf-8"))

    return result["content"][0]["text"]


def lambda_handler(event, context):
    """
    Main entry point for Lambda.
    Accepts two types of input:
    1. chart_data (from Phase 2) + optional query -> RAG reading
    2. query only (no image) -> RAG answer
    """
    try:
        # Load FAISS index, chunks, and model (only on cold start)
        load_resources()

        # Parse request body (handles both API Gateway and direct invoke)
        if isinstance(event.get("body"), str):
            body = json.loads(event["body"])
        else:
            body = event.get("body") or event

        chart_data = body.get("chart_data")
        user_query = body.get("query")

        if not chart_data and not user_query:
            return {
                "statusCode": 400,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"error": "Provide 'chart_data' and/or 'query'"})
            }

        # Build search query for FAISS
        if chart_data:
            search_query = build_query_from_chart(chart_data)
            if user_query:
                search_query += f". {user_query}"
        else:
            search_query = user_query

        # Retrieve relevant chunks from knowledge base
        retrieved_chunks = search_knowledge_base(search_query)

        # Generate reading using Anthropic API
        reading = call_anthropic_for_reading(
            chart_data or {},
            retrieved_chunks,
            user_query
        )

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({
                "success": True,
                "reading": reading,
                "sources": [{"source": c["source"], "score": c["score"]} for c in retrieved_chunks]
            })
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"success": False, "error": str(e)})
        }