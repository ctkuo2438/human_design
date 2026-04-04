import json
import os
import re
from typing import Any


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_text(text: str) -> str:
    text = "" if text is None else str(text)
    text = text.lower().strip()
    return re.sub(r"\s+", " ", text)


def normalize_list(values: list[Any]) -> list[str]:
    if values is None:
        return []
    normalized = [normalize_text(v) for v in values]
    return sorted(normalized)


def parse_lambda_body(payload: Any) -> dict[str, Any]:
    if payload is None:
        return {}

    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            return {}

    if not isinstance(payload, dict):
        return {}

    body = payload.get("body")
    if isinstance(body, str):
        try:
            body = json.loads(body)
        except json.JSONDecodeError:
            body = {}
    elif body is None:
        body = payload

    if isinstance(body, dict):
        return body
    return {}


def vision_field_score(pred: Any, expected: Any) -> float:
    return 1.0 if normalize_text(pred) == normalize_text(expected) else 0.0


def vision_list_score(pred_list: list[Any], expected_list: list[Any]) -> float:
    pred_set = set(normalize_list(pred_list))
    expected_set = set(normalize_list(expected_list))
    return 1.0 if pred_set == expected_set else 0.0


def keyword_coverage(reading: str, expected_keywords: list[str]) -> tuple[float, list[str]]:
    normalized_reading = normalize_text(reading)
    expected = normalize_list(expected_keywords)

    if not expected:
        return 0.0, []

    matched = [kw for kw in expected if kw in normalized_reading]
    return len(matched) / len(expected), matched


def write_metrics_report(path: str, data: dict[str, Any]) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
