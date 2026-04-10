"""
Phase 5: Golden Dataset Evaluation Script
Runs vision accuracy + RAG quality tests, logs results to MLflow.
Called by GitHub Actions on every push.
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
# any of them lower than the threshold, the pipeline fails and deployment is blocked, get a red X in GitHub Actions 
VISION_ACCURACY_THRESHOLD = 0.80
FAITHFULNESS_THRESHOLD = 3.0
RELEVANCE_THRESHOLD = 3.0
COMPLETENESS_THRESHOLD = 3.0


# ============================================================
# Part 1: Vision Accuracy (normalized fuzzy match)
# ============================================================

# --- Normalize maps: map common variants to a canonical form ---
AUTHORITY_MAP = {
    "emotional": "emotional/solar plexus",
    "solar plexus": "emotional/solar plexus",
    "emotional - solar plexus": "emotional/solar plexus",
    "emotional/solar plexus": "emotional/solar plexus",
    "emotional solar plexus": "emotional/solar plexus",
    "sacral": "sacral",
    "splenic": "splenic",
    "spleen": "splenic",
    "ego": "ego",
    "heart": "ego",
    "self-projected": "self-projected",
    "self projected": "self-projected",
    "lunar": "lunar cycle",
    "lunar cycle": "lunar cycle",
    "moon": "lunar cycle",
    "none": "lunar cycle",
    "no authority": "lunar cycle",
    "outer authority": "lunar cycle",
    "environment": "environment",
}

STRATEGY_MAP = {
    "to respond": "to respond",
    "wait to respond": "to respond",
    "responding": "to respond",
    "wait for the invitation": "wait for the invitation",
    "wait for invitation": "wait for the invitation",
    "being invited": "wait for the invitation",
    "wait for recognition and the invitation": "wait for the invitation",
    "to inform": "to inform",
    "inform": "to inform",
    "informing": "to inform",
    "wait a lunar cycle": "wait a lunar cycle",
    "wait 28 days": "wait a lunar cycle",
    "lunar cycle": "wait a lunar cycle",
}

TYPE_MAP = {
    "generator": "generator",
    "pure generator": "generator",
    "manifesting generator": "manifesting generator",
    "mani-gen": "manifesting generator",
    "manifestor": "manifestor",
    "projector": "projector",
    "reflector": "reflector",
}

DEFINITION_MAP = {
    "single": "single",
    "single definition": "single",
    "split": "split",
    "split definition": "split",
    "triple split": "triple split",
    "triple split definition": "triple split",
    "quadruple split": "quadruple split",
    "none": "none",
    "no definition": "none",
}

CENTER_MAP = {
    "head": "head",
    "crown": "head",
    "ajna": "ajna",
    "mind": "ajna",
    "throat": "throat",
    "g": "g/self",
    "g center": "g/self",
    "g/self": "g/self",
    "self": "g/self",
    "identity": "g/self",
    "heart": "heart/will/ego",
    "will": "heart/will/ego",
    "ego": "heart/will/ego",
    "heart/will/ego": "heart/will/ego",
    "heart/ego": "heart/will/ego",
    "will/ego": "heart/will/ego",
    "sacral": "sacral",
    "solar plexus": "solar plexus/emotional",
    "emotional": "solar plexus/emotional",
    "solar plexus/emotional": "solar plexus/emotional",
    "emotional/solar plexus": "solar plexus/emotional",
    "spleen": "spleen",
    "splenic": "spleen",
    "root": "root",
}


def normalize_value(val, field):
    """Normalize a value based on field type."""
    if val is None:
        return val
    s = str(val).lower().strip()

    if field == "type":
        return TYPE_MAP.get(s, s)
    elif field == "authority":
        return AUTHORITY_MAP.get(s, s)
    elif field == "strategy":
        return STRATEGY_MAP.get(s, s)
    elif field == "definition":
        return DEFINITION_MAP.get(s, s)
    return s


def normalize_center(c):
    """Normalize a single center name."""
    return CENTER_MAP.get(c.lower().strip(), c.lower().strip())


def normalize_channel(ch):
    """Normalize channel: sort gate numbers so '59-6' == '6-59'."""
    parts = str(ch).strip().split("-")
    if len(parts) == 2:
        try:
            return "-".join(sorted(parts, key=int))
        except ValueError:
            pass
    return ch.lower().strip()


def evaluate_vision_field(expected, actual, field):
    """Compare a single field between expected and actual chart_data."""
    exp_val = expected.get(field)
    act_val = actual.get(field)

    if exp_val is None:
        return None  # skip if not in golden data

    # For list fields (centers, channels): compare as normalized sets
    if isinstance(exp_val, list):
        if field in ("defined_centers", "undefined_centers"):
            exp_set = set(normalize_center(c) for c in exp_val)
            act_set = set(normalize_center(c) for c in act_val) if isinstance(act_val, list) else set()
        elif field == "active_channels":
            exp_set = set(normalize_channel(c) for c in exp_val)
            act_set = set(normalize_channel(c) for c in act_val) if isinstance(act_val, list) else set()
        else:
            exp_set = set(exp_val)
            act_set = set(act_val) if isinstance(act_val, list) else set()

        # Handle empty expected (e.g. Reflector with no channels)
        if len(exp_set) == 0 and len(act_set) == 0:
            return 1.0
        if len(exp_set) == 0 and len(act_set) > 0:
            return 0.0

        precision = len(exp_set & act_set) / len(act_set) if act_set else 0
        recall = len(exp_set & act_set) / len(exp_set) if exp_set else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return f1

    # For string fields: normalized match
    exp_norm = normalize_value(exp_val, field)
    act_norm = normalize_value(act_val, field)
    return 1.0 if exp_norm == act_norm else 0.0


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
    """Call Anthropic API using urllib (no SDK dependency)."""
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
    """Use LLM-as-Judge to score RAG response quality."""
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
    """Call /vision endpoint via API Gateway."""
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
    """Call RAG Lambda directly via boto3 (bypasses API Gateway timeout)."""
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
                print(f"  [Vision] Expected: {json.dumps(case['expected_chart_data'], indent=2)}")
                print(f"  [Vision] Actual:   {json.dumps(actual_chart, indent=2)}")
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