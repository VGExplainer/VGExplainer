#!/usr/bin/env python3
"""Evaluate TP LLM results with VIR/MSP/MSR/MIoU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from eval_common import extract_prediction, json_files, load_gt_lines, load_json, read_target_list


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-dir",
        action="append",
        required=True,
        help="Directory containing per-sample LLM JSON results. Can be passed multiple times.",
    )
    parser.add_argument("--gt-dir", required=True, help="Directory containing ground-truth JSON files.")
    parser.add_argument(
        "--target-list",
        help="Optional TXT file specifying the target sample set used as the VIR denominator.",
    )
    parser.add_argument(
        "--json-out",
        help="Optional output JSON file for machine-readable summaries.",
    )
    return parser.parse_args()


def evaluate_directory(result_dir: Path, gt_dir: Path, targets: set[str] | None) -> Dict[str, object]:
    files = json_files(result_dir)
    if targets is not None:
        files = [p for p in files if p.stem in targets or p.name in targets]

    denominator = len(targets) if targets is not None else len(files)
    predicted_yes = predicted_no = parse_error = 0
    total_precision = total_recall = total_iou = 0.0
    metric_samples = 0
    missing_gt = empty_gt = 0

    for path in files:
        data = load_json(path)
        if data is None:
            parse_error += 1
            continue

        label, pred_lines = extract_prediction(data)
        if label is None:
            parse_error += 1
            continue
        if label == "YES":
            predicted_yes += 1
        else:
            predicted_no += 1

        gt_lines = load_gt_lines(gt_dir, path.stem)
        if gt_lines is None:
            missing_gt += 1
            continue
        if not gt_lines:
            empty_gt += 1
            continue
        if label != "YES":
            continue

        intersection = pred_lines & gt_lines
        union = pred_lines | gt_lines
        precision = len(intersection) / len(pred_lines) if pred_lines else 0.0
        recall = len(intersection) / len(gt_lines)
        iou = len(intersection) / len(union) if union else 0.0
        total_precision += precision
        total_recall += recall
        total_iou += iou
        metric_samples += 1

    summary = {
        "result_dir": str(result_dir),
        "evaluated_files": len(files),
        "denominator": denominator,
        "predicted_yes": predicted_yes,
        "predicted_no": predicted_no,
        "parse_error": parse_error,
        "missing_gt": missing_gt,
        "empty_gt": empty_gt,
        "vir": round(predicted_yes / denominator * 100, 2) if denominator else 0.0,
        "metric_samples": metric_samples,
        "msp": round(total_precision / metric_samples * 100, 2) if metric_samples else 0.0,
        "msr": round(total_recall / metric_samples * 100, 2) if metric_samples else 0.0,
        "miou": round(total_iou / metric_samples * 100, 2) if metric_samples else 0.0,
    }
    return summary


def main() -> None:
    args = parse_args()
    gt_dir = Path(args.gt_dir)
    targets = read_target_list(Path(args.target_list)) if args.target_list else None

    summaries: List[Dict[str, object]] = []
    for result_dir_str in args.result_dir:
        summary = evaluate_directory(Path(result_dir_str), gt_dir, targets)
        summaries.append(summary)
        print(
            f"[TP] {summary['result_dir']}\n"
            f"  evaluated={summary['evaluated_files']} denominator={summary['denominator']} "
            f"VIR={summary['vir']:.2f}%\n"
            f"  metric_samples={summary['metric_samples']} MSP={summary['msp']:.2f}% "
            f"MSR={summary['msr']:.2f}% MIoU={summary['miou']:.2f}%\n"
            f"  predicted_yes={summary['predicted_yes']} predicted_no={summary['predicted_no']} "
            f"parse_error={summary['parse_error']} missing_gt={summary['missing_gt']} empty_gt={summary['empty_gt']}"
        )

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
        print(f"Saved JSON summary to {out_path}")


if __name__ == "__main__":
    main()
