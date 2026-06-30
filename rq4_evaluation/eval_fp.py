#!/usr/bin/env python3
"""Evaluate FP LLM results with FPRR against a code-only baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from eval_common import extract_prediction, json_files, load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", required=True, help="Directory containing code-only per-sample JSON results.")
    parser.add_argument(
        "--compare-dir",
        action="append",
        required=True,
        help="Directory containing hint-based per-sample JSON results. Can be passed multiple times.",
    )
    parser.add_argument(
        "--pool-dir",
        help="Optional directory whose JSON filenames define the fixed sample pool. If omitted, each compare-dir uses its own pool.",
    )
    parser.add_argument(
        "--json-out",
        help="Optional output JSON file for machine-readable summaries.",
    )
    return parser.parse_args()


def evaluate_compare(base_dir: Path, compare_dir: Path, pool_dir: Path | None) -> Dict[str, object]:
    pool_source = pool_dir or compare_dir
    pool_files = json_files(pool_source)

    base_yes = base_no = base_invalid = 0
    base_map: Dict[str, str | None] = {}
    for path in pool_files:
        data = load_json(base_dir / path.name)
        label, _ = extract_prediction(data or {})
        base_map[path.name] = label
        if label == "YES":
            base_yes += 1
        elif label == "NO":
            base_no += 1
        else:
            base_invalid += 1

    yes_to_no = no_to_yes = hint_invalid = matched_pairs = 0
    final_no_fixed = base_invalid
    final_no_valid = 0

    for path in pool_files:
        hint_data = load_json(compare_dir / path.name)
        hint_label, _ = extract_prediction(hint_data or {})
        base_label = base_map[path.name]

        if hint_label is None:
            hint_invalid += 1
        elif hint_label == "NO":
            final_no_fixed += 1

        if base_label in {"YES", "NO"} and hint_label in {"YES", "NO"}:
            matched_pairs += 1
            if hint_label == "NO":
                final_no_valid += 1
            if base_label == "YES" and hint_label == "NO":
                yes_to_no += 1
            elif base_label == "NO" and hint_label == "YES":
                no_to_yes += 1

    pool_size = len(pool_files)
    summary = {
        "pool_source": str(pool_source),
        "compare_dir": str(compare_dir),
        "pool_size": pool_size,
        "base_yes": base_yes,
        "base_no": base_no,
        "base_invalid": base_invalid,
        "base_fprr_fixed": round((base_no + base_invalid) / pool_size * 100, 2) if pool_size else 0.0,
        "base_fprr_valid": round(base_no / (base_yes + base_no) * 100, 2) if (base_yes + base_no) else 0.0,
        "yes_to_no": yes_to_no,
        "no_to_yes": no_to_yes,
        "net_gain_pct": round((yes_to_no - no_to_yes) / pool_size * 100, 2) if pool_size else 0.0,
        "final_fprr_fixed": round(final_no_fixed / pool_size * 100, 2) if pool_size else 0.0,
        "final_fprr_valid": round(final_no_valid / matched_pairs * 100, 2) if matched_pairs else 0.0,
        "hint_invalid": hint_invalid,
        "matched_pairs": matched_pairs,
    }
    return summary


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir)
    pool_dir = Path(args.pool_dir) if args.pool_dir else None

    summaries: List[Dict[str, object]] = []
    for compare_dir_str in args.compare_dir:
        summary = evaluate_compare(base_dir, Path(compare_dir_str), pool_dir)
        summaries.append(summary)
        print(
            f"[FP] {summary['compare_dir']}\n"
            f"  pool={summary['pool_size']} base_FPRR_fixed={summary['base_fprr_fixed']:.2f}% "
            f"base_FPRR_valid={summary['base_fprr_valid']:.2f}%\n"
            f"  yes_to_no={summary['yes_to_no']} no_to_yes={summary['no_to_yes']} "
            f"net_gain={summary['net_gain_pct']:+.2f}%\n"
            f"  final_FPRR_fixed={summary['final_fprr_fixed']:.2f}% "
            f"final_FPRR_valid={summary['final_fprr_valid']:.2f}% "
            f"hint_invalid={summary['hint_invalid']} matched_pairs={summary['matched_pairs']}"
        )

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
        print(f"Saved JSON summary to {out_path}")


if __name__ == "__main__":
    main()
