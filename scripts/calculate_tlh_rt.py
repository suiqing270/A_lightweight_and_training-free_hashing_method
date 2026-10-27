from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import add_project_root_to_path

PROJECT_ROOT = add_project_root_to_path()

from src.evaluation.tamper_accuracy import compare_tlh_and_calculate_rt
from src.utils.io import write_json


def parse_args():
    parser = argparse.ArgumentParser(description="Compare TLH outputs and calculate localization accuracy Rt.")
    parser.add_argument("--original-tlh", required=True, help="Original TLH JSON or legacy txt path.")
    parser.add_argument("--tampered-tlh", required=True, help="Tampered TLH JSON or legacy txt path.")
    parser.add_argument("--tamper-log", required=True, help="Ground-truth tamper log JSON path.")
    parser.add_argument(
        "--diff-output",
        required=True,
        help="Output JSON path for detected changed TLH blocks.",
    )
    parser.add_argument(
        "--summary-output",
        required=True,
        help="Output JSON path for Rt summary.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    detected_log, summary = compare_tlh_and_calculate_rt(
        args.original_tlh,
        args.tampered_tlh,
        args.tamper_log,
    )

    write_json(args.diff_output, detected_log)
    write_json(args.summary_output, summary)

    print(f"Detected TLH diff log saved to {Path(args.diff_output)}")
    print(f"Rt summary saved to {Path(args.summary_output)}")
    print(f"true_tampered_blocks: {summary['true_tampered_blocks']}")
    print(f"detected_changed_blocks: {summary['detected_changed_blocks']}")
    print(f"matched_tampered_blocks: {summary['matched_tampered_blocks']}")
    print(f"Rt: {summary['rt']:.4f}")


if __name__ == "__main__":
    main()
