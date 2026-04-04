import base64
import os
import sys
from mimetypes import guess_type

import requests

from utils import (
    load_json,
    parse_lambda_body,
    vision_field_score,
    vision_list_score,
    write_metrics_report,
)

DATASET_PATH = "datasets/vision_golden.json"
REPORT_PATH = "reports/latest_metrics.json"
VISION_THRESHOLD = 0.80
CORE_FIELDS = ["type", "authority", "profile", "defined_centers", "active_channels"]


def encode_image(path: str) -> tuple[str, str]:
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")

    media_type, _ = guess_type(path)
    if media_type is None:
        media_type = "application/octet-stream"
    return encoded, media_type


def extract_chart_data(parsed: dict) -> dict:
    candidates = [
        parsed.get("chart_data"),
        parsed.get("data", {}).get("chart_data") if isinstance(parsed.get("data"), dict) else None,
    ]
    for candidate in candidates:
        if isinstance(candidate, dict):
            return candidate
    return {}


def run() -> int:
    vision_url = os.getenv("VISION_API_URL", "").strip()
    if not vision_url:
        print("ERROR: VISION_API_URL is required")
        return 1

    samples = load_json(DATASET_PATH)
    total_score = 0.0

    for sample in samples:
        sample_id = sample["id"]
        image_path = sample["image_path"]
        expected = sample["expected_chart_data"]

        if not os.path.exists(image_path):
            print(f"ERROR [{sample_id}]: image not found at {image_path}")
            return 1

        image_b64, media_type = encode_image(image_path)
        payload = {"image_base64": image_b64, "media_type": media_type}

        try:
            response = requests.post(vision_url, json=payload, timeout=60)
        except requests.RequestException as exc:
            print(f"ERROR [{sample_id}]: request failed: {exc}")
            return 1

        if response.status_code >= 400:
            print(f"ERROR [{sample_id}]: status_code={response.status_code} body={response.text[:300]}")
            return 1

        try:
            raw_json = response.json()
        except ValueError:
            print(f"ERROR [{sample_id}]: non-JSON response: {response.text[:300]}")
            return 1

        parsed = parse_lambda_body(raw_json)
        predicted = extract_chart_data(parsed)

        type_score = vision_field_score(predicted.get("type"), expected.get("type"))
        authority_score = vision_field_score(predicted.get("authority"), expected.get("authority"))
        profile_score = vision_field_score(predicted.get("profile"), expected.get("profile"))
        centers_score = vision_list_score(predicted.get("defined_centers", []), expected.get("defined_centers", []))
        channels_score = vision_list_score(predicted.get("active_channels", []), expected.get("active_channels", []))

        sample_score = (type_score + authority_score + profile_score + centers_score + channels_score) / 5.0
        total_score += sample_score

        pred_core = {k: predicted.get(k) for k in CORE_FIELDS}
        expected_core = {k: expected.get(k) for k in CORE_FIELDS}
        print(f"\nSample: {sample_id}")
        print(f"Predicted: {pred_core}")
        print(f"Expected : {expected_core}")
        print(f"Score    : {sample_score:.3f}")

    sample_count = len(samples)
    average_score = total_score / sample_count if sample_count else 0.0
    print("\n=== Vision Metrics ===")
    print(f"vision_sample_count: {sample_count}")
    print(f"vision_average_score: {average_score:.3f}")

    report_data = {}
    if os.path.exists(REPORT_PATH):
        try:
            report_data = load_json(REPORT_PATH)
        except Exception:
            report_data = {}

    report_data.update(
        {
            "vision_sample_count": sample_count,
            "vision_average_score": round(average_score, 4),
        }
    )
    write_metrics_report(REPORT_PATH, report_data)

    if average_score < VISION_THRESHOLD:
        print(f"FAIL: vision_average_score {average_score:.3f} is below threshold {VISION_THRESHOLD:.2f}")
        return 1

    print("PASS: vision metrics meet threshold")
    return 0


if __name__ == "__main__":
    sys.exit(run())
