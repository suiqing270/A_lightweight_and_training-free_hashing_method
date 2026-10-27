from __future__ import annotations

import json
from pathlib import Path


def load_tlh_encoded(path: str | Path) -> list[str]:
    input_path = Path(path)
    if input_path.suffix.lower() == ".json":
        with input_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        if "encoded" not in data:
            raise KeyError(f"TLH JSON does not contain an 'encoded' field: {input_path}")
        return [str(item) for item in data["encoded"]]

    with input_path.open("r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]


def normalize_tamper_log(data) -> list[dict]:
    if isinstance(data, dict) and "tampered_blocks" in data:
        return list(data["tampered_blocks"])
    if isinstance(data, list):
        return data
    raise ValueError("Tamper log must be either a list or an object containing 'tampered_blocks'.")


def load_tamper_log(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as file:
        return normalize_tamper_log(json.load(file))


def detect_different_tlh_blocks(original_encoded: list[str], modified_encoded: list[str]) -> list[dict]:
    difference_log = []
    for block_index, (original, modified) in enumerate(zip(original_encoded, modified_encoded)):
        if original != modified:
            difference_log.append(
                {
                    "block_index": block_index,
                    "start_param_index": block_index,
                    "end_param_index": block_index,
                    "block_size": 1,
                }
            )
    return difference_log


def calculate_rt(detected_log: list[dict], tamper_log: list[dict]) -> dict:
    true_block_indexes = {item["block_index"] for item in tamper_log}
    true_count = len(tamper_log)
    hit_count = sum(1 for item in detected_log if item["block_index"] in true_block_indexes)
    detected_count = len(detected_log)
    return {
        "rt": hit_count / true_count if true_count > 0 else 0,
        "true_tampered_blocks": true_count,
        "detected_changed_blocks": detected_count,
        "matched_tampered_blocks": hit_count,
        "false_positive_blocks": detected_count - hit_count,
    }


def compare_tlh_and_calculate_rt(
    original_tlh_path: str | Path,
    modified_tlh_path: str | Path,
    tamper_log_path: str | Path,
) -> tuple[list[dict], dict]:
    original_encoded = load_tlh_encoded(original_tlh_path)
    modified_encoded = load_tlh_encoded(modified_tlh_path)
    total_count = min(len(original_encoded), len(modified_encoded))
    detected_log = detect_different_tlh_blocks(original_encoded, modified_encoded)
    tamper_log = load_tamper_log(tamper_log_path)
    summary = calculate_rt(detected_log, tamper_log)
    summary["total_compared_blocks"] = total_count
    summary["original_blocks"] = len(original_encoded)
    summary["modified_blocks"] = len(modified_encoded)
    return detected_log, summary
