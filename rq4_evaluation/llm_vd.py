#!/usr/bin/env python3
"""Code-only LLM vulnerability analysis for RQ4."""

from __future__ import annotations

import argparse
from pathlib import Path

from llm_vd_common import (
    build_chain,
    dump_result,
    read_target_list,
    resolve_source_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-list", required=True, help="TXT file containing sample ids.")
    parser.add_argument("--code-dir", required=True, help="Directory containing line-numbered source files.")
    parser.add_argument("--output-dir", required=True, help="Directory to save per-sample JSON results.")
    parser.add_argument("--model", default="gpt-oss:20b", help="Ollama model name.")
    parser.add_argument("--base-url", default="http://localhost:11434", help="Ollama base URL.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--num-ctx", type=int, default=32768, help="Ollama context window.")
    parser.add_argument("--keep-alive", default="8h", help="Ollama keep_alive value.")
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".cpp", ".c"],
        help="Source-file extensions to try in order.",
    )
    parser.add_argument("--resume", action="store_true", help="Skip samples whose output already exists.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chain = build_chain(
        model_name=args.model,
        base_url=args.base_url,
        temperature=args.temperature,
        num_ctx=args.num_ctx,
        keep_alive=args.keep_alive,
        with_hint=False,
    )

    target_list = Path(args.target_list)
    code_dir = Path(args.code_dir)
    output_dir = Path(args.output_dir)
    samples = read_target_list(target_list)

    print(f"Found {len(samples)} samples.")
    for idx, sample_name in enumerate(samples, start=1):
        source_path = resolve_source_path(code_dir, sample_name, args.extensions)
        if source_path is None:
            print(f"[{idx}/{len(samples)}] missing source: {sample_name}")
            continue

        output_path = output_dir / f"{sample_name}.json"
        if args.resume and output_path.exists():
            print(f"[{idx}/{len(samples)}] skip existing: {sample_name}")
            continue

        source_code = source_path.read_text(encoding="utf-8", errors="ignore")
        if not source_code.strip():
            print(f"[{idx}/{len(samples)}] empty source: {sample_name}")
            continue

        print(f"[{idx}/{len(samples)}] analyzing: {sample_name}")
        result = chain.invoke({"source_code": source_code})
        payload = {
            "sample": sample_name,
            "mode": "code_only",
            "model": args.model,
            "source_file": source_path.name,
            "analysis_findings": [result.model_dump()],
        }
        dump_result(output_path, payload)

    print("Done.")


if __name__ == "__main__":
    main()
