#!/usr/bin/env python3
"""Shared helpers for RQ4 evaluation scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set, Tuple


LABELS = {"YES", "NO"}


def load_json(path: Path) -> Optional[dict]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def json_files(directory: Path) -> List[Path]:
    return sorted(p for p in directory.iterdir() if p.suffix == ".json")


def read_target_list(list_path: Optional[Path]) -> Optional[Set[str]]:
    if list_path is None:
        return None
    targets: Set[str] = set()
    for line in list_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.endswith(".json"):
            line = line[:-5]
        if line.endswith(".cpp") or line.endswith(".c"):
            line = Path(line).stem
        targets.add(line)
    return targets


def normalize_label(value: object) -> Optional[str]:
    if value is None:
        return None
    label = str(value).strip().upper()
    return label if label in LABELS else None


def _iter_findings(data: dict) -> List[dict]:
    findings = data.get("analysis_findings")
    if isinstance(findings, list):
        return [x for x in findings if isinstance(x, dict)]

    # Compatibility with older scripts that stored a single parsed result.
    candidates = [
        data.get("results", {}).get("code_only", {}).get("parsed"),
        data.get("results", {}).get("with_hint", {}).get("parsed"),
        data.get("results", {}).get("hint_only", {}).get("parsed"),
        data.get("code_only"),
        data.get("with_hint"),
        data.get("hint_result"),
        data.get("parsed"),
        data,
    ]
    collected: List[dict] = []
    for candidate in candidates:
        if isinstance(candidate, dict) and normalize_label(candidate.get("is_vulnerable")):
            collected.append(candidate)
            break
    return collected


def extract_prediction(data: dict) -> Tuple[Optional[str], Set[int]]:
    findings = _iter_findings(data)
    has_yes = False
    line_set: Set[int] = set()
    saw_label = False

    for finding in findings:
        label = normalize_label(finding.get("is_vulnerable"))
        if label is None:
            continue
        saw_label = True
        if label == "YES":
            has_yes = True
            lines = finding.get("related_lines", [])
            if isinstance(lines, list):
                for line in lines:
                    if isinstance(line, int):
                        line_set.add(line)
                    elif isinstance(line, str) and line.isdigit():
                        line_set.add(int(line))

    if not saw_label:
        return None, set()
    return ("YES" if has_yes else "NO"), line_set


def candidate_gt_names(sample_stem: str) -> List[str]:
    names = [f"{sample_stem}.json"]
    if len(sample_stem) > 2 and sample_stem[1] == "_" and sample_stem[0].isdigit():
        names.append(f"{sample_stem[2:]}.json")
    return names


def resolve_gt_path(gt_dir: Path, sample_stem: str) -> Optional[Path]:
    for name in candidate_gt_names(sample_stem):
        candidate = gt_dir / name
        if candidate.exists():
            return candidate
    return None


def load_gt_lines(gt_dir: Path, sample_stem: str) -> Optional[Set[int]]:
    gt_path = resolve_gt_path(gt_dir, sample_stem)
    if gt_path is None:
        return None
    data = load_json(gt_path)
    if not data:
        return None
    raw = data.get("ground_truth", [])
    if not isinstance(raw, list):
        return None
    lines: Set[int] = set()
    for item in raw:
        if isinstance(item, int):
            lines.add(item)
        elif isinstance(item, str) and item.isdigit():
            lines.add(int(item))
    return lines
