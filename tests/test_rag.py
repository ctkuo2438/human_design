import json
import os
import sys

import boto3

from utils import keyword_coverage, load_json, parse_lambda_body, write_metrics_report

DATASET_PATH = "datasets/rag_golden.json"
REPORT_PATH = "reports/latest_metrics.json"
SUCCESS_THRESHOLD = 1.0
KEYWORD_THRESHOLD = 0.60


def extract_reading(parsed: dict) -> str:
    candidates = [
        parsed.get("reading"),
        parsed.get("answer"),
        parsed.get("response"),
        parsed.get("result"),
    ]

    data = parsed.get("data")
    if isinstance(data, dict):
        candidates.extend(
            [
                data.get("reading"),
                data.get("answer"),
                data.get("response"),
                data.get("result"),
            ]
        )

    for value in candidates:
        if isinstance(value, str) and value.strip():
            return value
    return ""


def response_indicates_success(invoke_response: dict, parsed_payload: dict, reading: str) -> bool:
    status_code = invoke_response.get("StatusCode", 0)
    function_error = invoke_response.get("FunctionError")
    app_success = parsed_payload.get("success")

    if function_error:
        return False
    if status_code not in (200, 202):
        return False
    if isinstance(app_success, bool) and not app_success:
        return False
    return bool(reading.strip())


def run() -> int:
    region = os.getenv("AWS_REGION", "us-east-1").strip() or "us-east-1"
    lambda_name = os.getenv("RAG_LAMBDA_NAME", "").strip()

    if not lambda_name:
        print("ERROR: RAG_LAMBDA_NAME is required")
        return 1

    client = boto3.client("lambda", region_name=region)
    samples = load_json(DATASET_PATH)

    success_total = 0
    coverage_total = 0.0

    for sample in samples:
        sample_id = sample["id"]
        payload = {"query": sample["query"]}
        if "chart_data" in sample:
            payload["chart_data"] = sample["chart_data"]

        try:
            invoke_response = client.invoke(
                FunctionName=lambda_name,
                InvocationType="RequestResponse",
                Payload=json.dumps(payload).encode("utf-8"),
            )
        except Exception as exc:
            print(f"ERROR [{sample_id}]: Lambda invoke failed: {exc}")
            return 1

        raw_payload = invoke_response["Payload"].read()
        try:
            payload_json = json.loads(raw_payload.decode("utf-8"))
        except Exception:
            payload_json = {}

        parsed_payload = parse_lambda_body(payload_json)
        reading = extract_reading(parsed_payload)

        coverage, matched_keywords = keyword_coverage(reading, sample.get("expected_keywords", []))
        success = 1 if response_indicates_success(invoke_response, parsed_payload, reading) else 0

        success_total += success
        coverage_total += coverage

        preview = reading[:160].replace("\n", " ")
        print(f"\nSample: {sample_id}")
        print(f"Query: {sample['query']}")
        print(f"chart_data_included: {'chart_data' in sample}")
        print(f"success: {success}")
        print(f"keyword_coverage: {coverage:.3f}")
        print(f"matched_keywords: {matched_keywords}")
        print(f"reading_preview: {preview}")

    sample_count = len(samples)
    success_rate = success_total / sample_count if sample_count else 0.0
    avg_coverage = coverage_total / sample_count if sample_count else 0.0

    print("\n=== RAG Metrics ===")
    print(f"rag_sample_count: {sample_count}")
    print(f"rag_success_rate: {success_rate:.3f}")
    print(f"rag_average_keyword_coverage: {avg_coverage:.3f}")

    report_data = {}
    if os.path.exists(REPORT_PATH):
        try:
            report_data = load_json(REPORT_PATH)
        except Exception:
            report_data = {}

    report_data.update(
        {
            "rag_sample_count": sample_count,
            "rag_success_rate": round(success_rate, 4),
            "rag_average_keyword_coverage": round(avg_coverage, 4),
        }
    )
    write_metrics_report(REPORT_PATH, report_data)

    if success_rate < SUCCESS_THRESHOLD:
        print(f"FAIL: rag_success_rate {success_rate:.3f} is below threshold {SUCCESS_THRESHOLD:.1f}")
        return 1

    if avg_coverage < KEYWORD_THRESHOLD:
        print(f"FAIL: rag_average_keyword_coverage {avg_coverage:.3f} is below threshold {KEYWORD_THRESHOLD:.2f}")
        return 1

    print("PASS: RAG metrics meet thresholds")
    return 0


if __name__ == "__main__":
    sys.exit(run())
