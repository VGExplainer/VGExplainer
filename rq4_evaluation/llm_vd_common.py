#!/usr/bin/env python3
"""Shared utilities for RQ4 LLM vulnerability analysis scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Literal, Optional

from langchain.output_parsers import OutputFixingParser, PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain_community.llms.ollama import Ollama
from pydantic import BaseModel, Field


CODE_ONLY_PROMPT_TEMPLATE = """
You are a security expert that is good at static program analysis.
Please analyze the following C/C++ code for potential security vulnerabilities.

CRITICAL INSTRUCTION: The provided source code has been explicitly annotated with line numbers at the beginning of each line (e.g., 'Line 12:').

Task:
- Analyze the ENTIRE current function and determine whether the CURRENT code contains a concrete security vulnerability.
- Your judgment must be based on evidence visible in the current code itself.

[Expert Analysis Hints]:
No expert analysis hint is provided for this sample.

IMPORTANT:
- You must conduct a thorough, system-wide static analysis of the ENTIRE code and draw your own independent conclusion.
- Do NOT rely only on hypothetical conditions, unknown callee behavior, missing external assumptions, or unsupported speculation.
- Similarity to a historical vulnerability pattern is not sufficient by itself to conclude that the current code is vulnerable.

Let's think step-by-step. Conduct a thorough static analysis of the code.

Recommended reasoning process:
1. First perform an independent preliminary analysis of the full code and identify any concrete unsafe operation, missing or insufficient validation, or unsafe data/control-flow.
2. Then verify whether the suspected issue is fully supported by the visible current-code evidence, rather than by pattern similarity, patch-history resemblance, or hypothetical execution conditions.
3. Finally provide a single verdict for the current code as a whole.

Before deciding, check the following:
1. Does the current code show a concrete unsafe operation or a missing/insufficient safety check?
2. Is the claimed vulnerability supported by the visible code rather than by a hypothetical scenario?
3. Could the suspicious pattern be benign, already guarded, unreachable, incomplete, or part of a repaired code path?
4. Can you point to exact line-numbered evidence that justifies your final judgment?

Please indicate your analysis result with one of the options:
(1) YES: A security vulnerability detected.
(2) NO: No security vulnerability.

Make sure to include one of the options above EXPLICITLY in your response.

You must provide your final answer strictly in the following JSON format.
Here are the definitions for the required JSON fields:
- is_vulnerable: Must be exactly "YES" or "NO", corresponding to your conclusion.
- vulnerability_type: The specific type of the vulnerability. Use "None" if NO vulnerability.
- vulnerability_reason: Put your step-by-step reasoning here. Explain why the vulnerability exists or why the code should be judged safe based on the visible evidence.
- related_lines: An array of integers representing the specific line numbers related to the conclusion. STRICT REQUIREMENT: You MUST NOT count the lines yourself. You must strictly output the numbers exactly as they appear in the 'Line X:' annotations. If NO vulnerability, output [] unless a small number of lines are necessary to justify the rejection.
- confidence: Integer confidence score from 0 to 100.

{format_instructions}

---
[Source Code]:
{source_code}
""".strip()


HINT_PROMPT_TEMPLATE = """
You are a security expert that is good at static program analysis.
Please analyze the following C/C++ code for potential security vulnerabilities.

CRITICAL INSTRUCTION: The provided source code has been explicitly annotated with line numbers at the beginning of each line (e.g., 'Line 12:').

Task:
- Analyze the ENTIRE current function and determine whether the CURRENT code contains a concrete security vulnerability.
- Your judgment must be based on evidence visible in the current code itself.

[Expert Analysis Hints]:
A previous analysis tool has flagged the following line sequence as a potential clue:
{expert_hint_block}

IMPORTANT:
- This hint is strictly for your REFERENCE ONLY.
- You MUST NOT assume the vulnerability exclusively resides in or is limited to these lines, and you must not blindly trust the hint.
- You are required to conduct a thorough, system-wide static analysis of the ENTIRE code and draw your own independent conclusion.
- Do NOT rely only on hypothetical conditions, unknown callee behavior, missing external assumptions, or unsupported speculation.
- Similarity to a historical vulnerability pattern is not sufficient by itself to conclude that the current code is vulnerable.

Let's think step-by-step. Conduct a thorough static analysis of the code.

Recommended reasoning process:
1. First perform an independent preliminary analysis of the full code without relying on the highlighted clue.
2. Then review the clue and decide whether it supports, refines, or conflicts with your preliminary judgment.
3. Finally provide a single verdict for the current code as a whole.

Before deciding, check the following:
1. Does the current code show a concrete unsafe operation or a missing/insufficient safety check?
2. Is the claimed vulnerability supported by the visible code rather than by a hypothetical scenario?
3. Could the highlighted pattern be benign, already guarded, unreachable, incomplete, or only loosely related to the actual vulnerability logic?
4. Can you point to exact line-numbered evidence that justifies your final judgment after reviewing the clue?

Please indicate your analysis result with one of the options:
(1) YES: A security vulnerability detected.
(2) NO: No security vulnerability.

Make sure to include one of the options above EXPLICITLY in your response.

You must provide your final answer strictly in the following JSON format.
Here are the definitions for the required JSON fields:
- is_vulnerable: Must be exactly "YES" or "NO", corresponding to your conclusion.
- vulnerability_type: The specific type of the vulnerability. Use "None" if NO vulnerability.
- vulnerability_reason: Put your step-by-step reasoning here. Explain why the vulnerability exists or why the code should be judged safe based on the visible evidence. Briefly note whether the clue confirmed or changed your preliminary judgment.
- related_lines: An array of integers representing the specific line numbers related to the conclusion. STRICT REQUIREMENT: You MUST NOT count the lines yourself. You must strictly output the numbers exactly as they appear in the 'Line X:' annotations.
- confidence: Integer confidence score from 0 to 100.

{format_instructions}

---
[Source Code]:
{source_code}
""".strip()


class VulnerabilityFinding(BaseModel):
    is_vulnerable: Literal["YES", "NO"] = Field(
        description="Output YES if the function truly contains a security vulnerability, otherwise NO."
    )
    vulnerability_type: str = Field(
        description="Specific vulnerability type if YES, or None if NO."
    )
    vulnerability_reason: str = Field(
        description="Concrete reasoning for why the code should be judged vulnerable or safe."
    )
    related_lines: List[int] = Field(
        description="Exact line numbers taken from the provided Line X labels."
    )
    confidence: int = Field(
        description="Integer confidence score from 0 to 100."
    )


def build_chain(
    model_name: str,
    base_url: str,
    temperature: float,
    num_ctx: int,
    keep_alive: str,
    with_hint: bool,
):
    llm = Ollama(
        model=model_name,
        temperature=temperature,
        num_ctx=num_ctx,
        keep_alive=keep_alive,
        base_url=base_url,
    )
    parser = PydanticOutputParser(pydantic_object=VulnerabilityFinding)
    output_fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)
    template = HINT_PROMPT_TEMPLATE if with_hint else CODE_ONLY_PROMPT_TEMPLATE
    input_variables = ["source_code", "expert_hint_block"] if with_hint else ["source_code"]
    prompt = PromptTemplate(
        template=template,
        input_variables=input_variables,
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    return prompt | llm | output_fixing_parser


def read_target_list(list_path: Path) -> List[str]:
    samples: List[str] = []
    for line in list_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.endswith(".json"):
            line = line[:-5]
        if line.endswith(".cpp") or line.endswith(".c"):
            line = Path(line).stem
        samples.append(line)
    return samples


def resolve_source_path(code_dir: Path, sample_name: str, extensions: Iterable[str]) -> Optional[Path]:
    for ext in extensions:
        candidate = code_dir / f"{sample_name}{ext}"
        if candidate.exists():
            return candidate
    return None


def parse_hint_lines(data: dict) -> List[int]:
    candidates = [
        data.get("important_lines"),
        data.get("results", {}).get("important_lines"),
        data.get("hint_lines"),
        data.get("lines"),
    ]
    for value in candidates:
        if isinstance(value, list):
            normalized = sorted({int(x) for x in value if isinstance(x, int) or str(x).isdigit()})
            if normalized:
                return normalized
    return []


def build_hint_block(lines: List[int]) -> str:
    if not lines:
        return "No expert analysis hint is available for this sample."
    return f"Lines {lines}"


def load_hint_block(hint_dir: Path, sample_name: str) -> Optional[str]:
    hint_path = hint_dir / f"{sample_name}.json"
    if not hint_path.exists():
        return None
    try:
        with hint_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    lines = parse_hint_lines(data if isinstance(data, dict) else {})
    if not lines:
        return None
    return build_hint_block(lines)


def dump_result(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
