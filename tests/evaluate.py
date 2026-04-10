"""
Phase 5: Golden Dataset Evaluation Script
Runs vision accuracy + RAG quality tests, logs results to MLflow.
Called by GitHub Actions on every push.

This CI/CD pipeline is the auto-test after the developer pushes code.
Not for the real-time guardrails during inference.
"""

import json
import os
import sys
import time
import urllib.request
import boto3
import mlflow

# --- Config ---
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
API_BASE = os.environ.get("API_BASE", "https://vemefih3xa.execute-api.us-east-1.amazonaws.com/prod")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://3.85.93.189:5000")
S3_BUCKET = os.environ.get("S3_BUCKET", "human-design-knowledge-base")
PROMPT_VERSION = os.environ.get("PROMPT_VERSION", "v1.0")

# Thresholds — fail CI if below these
VISION_ACCURACY_THRESHOLD = 0.80
FAITHFULNESS_THRESHOLD = 3.0
RELEVANCE_THRESHOLD = 3.0
COMPLETENESS_THRESHOLD = 3.0


# ============================================================
# Part 1: Vision Accuracy (field-by-field exact match)
# ============================================================

# compare the golden chart_data with the actual chart_data returned by /vision, field by field, and calculate an overall accuracy score. 
# For list fields (centers, channels), use F1 score based on set overlap. 
# For string fields, use exact match (case-insensitive).
def evaluate_vision_field(expected, actual, field):
    """
    Compare a single field between expected and actual chart_data.
    """
    exp_val = expected.get(field)
    act_val = actual.get(field)

    if exp_val is None:
        return None  # skip if not in golden data

    # For list fields (centers, channels): compare as sets
    if isinstance(exp_val, list):
        exp_set = set(exp_val)
        act_set = set(act_val) if isinstance(act_val, list) else set()
        precision = len(exp_set & act_set) / len(act_set) if act_set else 0
        recall = len(exp_set & act_set) / len(exp_set) if exp_set else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return f1

    # For string fields: exact match (case-insensitive)
    return 1.0 if str(exp_val).lower().strip() == str(act_val).lower().strip() else 0.0


def evaluate_vision(expected_chart, actual_chart):
    """Calculate overall vision accuracy across all fields."""
    fields = ["type", "authority", "profile", "strategy", "definition",
              "defined_centers", "undefined_centers", "active_channels"]

    scores = {}
    for field in fields:
        score = evaluate_vision_field(expected_chart, actual_chart, field)
        if score is not None:
            scores[field] = score

    overall = sum(scores.values()) / len(scores) if scores else 0
    return overall, scores


# ============================================================
# Part 2: RAG Quality (LLM-as-Judge)
# ============================================================

# LLM will compare the RAG response against the expected key facts and chart data, and score it on faithfulness, relevance, and completeness.
JUDGE_PROMPT = """You are an evaluator for a Human Design reading system.

Given:
- User question: {question}
- Chart data: {chart_data}
- System response: {response}
- Expected key facts the response should cover:
{expected_facts}

Score each dimension from 1 to 5:

1. **Faithfulness** (1-5): Are all claims in the response grounded in Human Design knowledge? Does it avoid fabricating channels, centers, or types not present in the chart data?
   - 5: Every claim is accurate and supported
   - 3: Mostly accurate with minor unsupported claims
   - 1: Contains significant fabricated information

2. **Relevance** (1-5): Does the response directly address the user's question?
   - 5: Fully addresses the question with specific guidance
   - 3: Partially addresses the question
   - 1: Off-topic or generic response

3. **Completeness** (1-5): How many of the expected key facts are covered?
   - 5: All or nearly all key facts covered
   - 3: About half of key facts covered
   - 1: Very few key facts covered

Return ONLY a JSON object with no other text:
{{"faithfulness": <score>, "relevance": <score>, "completeness": <score>}}"""


def call_anthropic(prompt, max_tokens=200):
    """
    Call Anthropic API using urllib (no SDK dependency).
    """
    payload = json.dumps({
        "model": "claude-sonnet-4-20250514",
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}]
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01"
        }
    )

    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        return data["content"][0]["text"]


def judge_rag_response(question, chart_data, response, expected_facts):
    """
    Use LLM-as-Judge to score RAG response quality.
    """
    facts_str = "\n".join(f"  - {f}" for f in expected_facts)
    prompt = JUDGE_PROMPT.format(
        question=question,
        chart_data=json.dumps(chart_data, indent=2),
        response=response,
        expected_facts=facts_str
    )

    raw = call_anthropic(prompt)

    # Parse JSON from response (handle markdown fences)
    clean = raw.strip().replace("```json", "").replace("```", "").strip()
    scores = json.loads(clean)
    return scores


# ============================================================
# Part 3: Call system endpoints
# ============================================================

def call_vision_api(image_b64, media_type="image/png"):
    """
    Call /vision endpoint via API Gateway.
    Get back the parsed chart_data for the input image.
    """
    import requests
    resp = requests.post(
        f"{API_BASE}/vision",
        json={"image": image_b64, "media_type": media_type},
        timeout=30
    )
    result = resp.json()
    if isinstance(result.get("body"), str):
        body = json.loads(result["body"])
    else:
        body = result
    return body.get("chart_data", {})


def call_reading_api(chart_data, query):
    """
    Call RAG Lambda directly via boto3 (bypasses API Gateway timeout).
    """
    lambda_client = boto3.client("lambda", region_name="us-east-1")
    payload = {"chart_data": chart_data, "query": query}
    resp = lambda_client.invoke(
        FunctionName="human-design-rag",
        InvocationType="RequestResponse",
        Payload=json.dumps(payload)
    )
    result = json.loads(resp["Payload"].read().decode("utf-8"))
    body = json.loads(result["body"]) if isinstance(result.get("body"), str) else result
    return body.get("reading", "")


# ============================================================
# Part 4: Main evaluation loop
# ============================================================

def load_image_from_s3(s3_key):
    """Download test image from S3 and return base64."""
    import base64
    s3 = boto3.client("s3", region_name="us-east-1")
    obj = s3.get_object(Bucket=S3_BUCKET, Key=s3_key)
    img_bytes = obj["Body"].read()
    return base64.b64encode(img_bytes).decode("utf-8")


def main():
    # Load golden dataset
    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(script_dir, "golden_dataset.json")) as f:
        dataset = json.load(f)

    # Setup MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("human-design-evaluation")

    all_vision_scores = []
    all_faithfulness = []
    all_relevance = []
    all_completeness = []

    with mlflow.start_run(run_name=f"eval-{PROMPT_VERSION}-{int(time.time())}"):
        mlflow.log_param("prompt_version", PROMPT_VERSION)
        mlflow.log_param("num_test_cases", len(dataset))
        mlflow.log_param("model", "claude-sonnet")

        for case in dataset:
            test_id = case["test_id"]
            print(f"\n{'='*50}")
            print(f"Running: {test_id} - {case['description']}")
            print(f"{'='*50}")

            # --- Vision test ---
            try:
                print(f"  [Vision] Calling /vision API...")
                img_b64 = load_image_from_s3(case["image_s3_key"])
                actual_chart = call_vision_api(img_b64)
                vision_score, field_scores = evaluate_vision(
                    case["expected_chart_data"], actual_chart
                )
                all_vision_scores.append(vision_score)
                print(f"  [Vision] Score: {vision_score:.2f}")
                for field, score in field_scores.items():
                    print(f"    {field}: {score:.2f}")
                    mlflow.log_metric(f"vision_{field}_{test_id}", score)
            except Exception as e:
                print(f"  [Vision] ERROR: {e}")
                all_vision_scores.append(0.0)

            # --- RAG test ---
            try:
                print(f"  [RAG] Calling /reading API...")
                reading = call_reading_api(
                    case["expected_chart_data"],  # use golden chart_data for RAG test
                    case["test_query"]
                )
                print(f"  [RAG] Got response ({len(reading)} chars)")

                print(f"  [Judge] Scoring response...")
                scores = judge_rag_response(
                    case["test_query"],
                    case["expected_chart_data"],
                    reading,
                    case["expected_key_facts"]
                )
                all_faithfulness.append(scores["faithfulness"])
                all_relevance.append(scores["relevance"])
                all_completeness.append(scores["completeness"])
                print(f"  [Judge] Faithfulness={scores['faithfulness']}, "
                      f"Relevance={scores['relevance']}, "
                      f"Completeness={scores['completeness']}")
                mlflow.log_metric(f"faith_{test_id}", scores["faithfulness"])
                mlflow.log_metric(f"relev_{test_id}", scores["relevance"])
                mlflow.log_metric(f"compl_{test_id}", scores["completeness"])
            except Exception as e:
                print(f"  [RAG] ERROR: {e}")
                all_faithfulness.append(0)
                all_relevance.append(0)
                all_completeness.append(0)

        # --- Aggregate metrics ---
        avg_vision = sum(all_vision_scores) / len(all_vision_scores) if all_vision_scores else 0
        avg_faith = sum(all_faithfulness) / len(all_faithfulness) if all_faithfulness else 0
        avg_relev = sum(all_relevance) / len(all_relevance) if all_relevance else 0
        avg_compl = sum(all_completeness) / len(all_completeness) if all_completeness else 0

        mlflow.log_metric("avg_vision_accuracy", avg_vision)
        mlflow.log_metric("avg_faithfulness", avg_faith)
        mlflow.log_metric("avg_relevance", avg_relev)
        mlflow.log_metric("avg_completeness", avg_compl)

        print(f"\n{'='*50}")
        print(f"RESULTS SUMMARY")
        print(f"{'='*50}")
        print(f"  Vision Accuracy:  {avg_vision:.2f}  (threshold: {VISION_ACCURACY_THRESHOLD})")
        print(f"  Faithfulness:     {avg_faith:.2f}  (threshold: {FAITHFULNESS_THRESHOLD})")
        print(f"  Relevance:        {avg_relev:.2f}  (threshold: {RELEVANCE_THRESHOLD})")
        print(f"  Completeness:     {avg_compl:.2f}  (threshold: {COMPLETENESS_THRESHOLD})")

        # --- Pass/Fail ---
        failed = False
        if avg_vision < VISION_ACCURACY_THRESHOLD:
            print(f"\n  FAIL: Vision accuracy {avg_vision:.2f} < {VISION_ACCURACY_THRESHOLD}")
            failed = True
        if avg_faith < FAITHFULNESS_THRESHOLD:
            print(f"\n  FAIL: Faithfulness {avg_faith:.2f} < {FAITHFULNESS_THRESHOLD}")
            failed = True
        if avg_relev < RELEVANCE_THRESHOLD:
            print(f"\n  FAIL: Relevance {avg_relev:.2f} < {RELEVANCE_THRESHOLD}")
            failed = True
        if avg_compl < COMPLETENESS_THRESHOLD:
            print(f"\n  FAIL: Completeness {avg_compl:.2f} < {COMPLETENESS_THRESHOLD}")
            failed = True

        if failed:
            print("\n  PIPELINE STATUS: FAILED — deployment blocked")
            sys.exit(1)
        else:
            print("\n  PIPELINE STATUS: PASSED — safe to deploy")
            sys.exit(0)


if __name__ == "__main__":
    main()
