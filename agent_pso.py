#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

try:
    import pyarrow.parquet as pq
except ModuleNotFoundError:
    pq = None


SCRIPT_PATH = Path(__file__).resolve()
IMPLEMENTATION_DIR = SCRIPT_PATH.parent
WORKSPACE_ROOT = IMPLEMENTATION_DIR.parent.parent
PARETO_ROOT = WORKSPACE_ROOT / "the_pareto_one"

if str(PARETO_ROOT) not in sys.path:
    sys.path.insert(0, str(PARETO_ROOT))

try:
    from math_grader import extract_answer, grade  # type: ignore  # noqa: E402
except ModuleNotFoundError:
    def extract_answer(text: str) -> str:
        value = str(text)
        boxed = re.search(r"\\boxed\{([^{}]*)\}", value)
        if boxed:
            return boxed.group(1).strip()
        return value.strip()

    def _normalize_math_answer(text: Any) -> str:
        value = extract_answer(str(text))
        value = re.sub(r"^(final answer|answer)\s*[:=]\s*", "", value.strip(), flags=re.IGNORECASE)
        value = value.strip().strip("$").rstrip(".")
        return re.sub(r"\s+", "", value)

    def grade(selected_answer: str, ground_truth: Any) -> bool:
        return _normalize_math_answer(selected_answer) == _normalize_math_answer(ground_truth)


DEFAULT_TRAIN_DATASET = WORKSPACE_ROOT / "data/MATH-benchmark/train-00000-of-00001.parquet"
DEFAULT_TEST_DATASET = WORKSPACE_ROOT / "data/MATH-benchmark/test-00000-of-00001.parquet"
DEFAULT_DEEPMATH_DATASET = WORKSPACE_ROOT / "data/DeepMath-103K/data/train-00000-of-00010.parquet"
DEFAULT_AIME24_TEST_DATASET = WORKSPACE_ROOT / "data/AIME24/test-00000-of-00001.parquet"
DEFAULT_AIME25_TEST_DATASET = WORKSPACE_ROOT / "data/AIME25/test.jsonl"
DEFAULT_MINERVA_TEST_DATASET = WORKSPACE_ROOT / "data/minervamath/test.jsonl"
DEFAULT_BBH_DATASET = WORKSPACE_ROOT / "data/BigBenchHard/bigbenchhard_all.jsonl"
DEFAULT_OUTPUT_ROOT = IMPLEMENTATION_DIR
DEFAULT_MODEL = "gpt-5.4-mini"
DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_API_BASE = "https://api.openai.com/v1"

INITIAL_SKILLS = {
    1: {
        "name": "cot_basic",
        "identity": "CoT Basic",
        "text": "Solve the problem step by step.",
    },
    2: {
        "name": "step_back_prompting",
        "identity": "Step-Back Prompting",
        "text": "Before solving, step back and identify the general principle or problem type.",
    },
    3: {
        "name": "self_refine",
        "identity": "Self-Refine",
        "text": "First solve the problem, then review and improve the solution.",
    },
    4: {
        "name": "reflection",
        "identity": "Reflection",
        "text": "Solve while reflecting on assumptions and possible failure points.",
    },
}


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def strip_code_fences(text: str) -> str:
    cleaned = str(text).strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return cleaned


def extract_json_object(text: str) -> dict[str, Any] | None:
    cleaned = strip_code_fences(text)
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    for match in re.finditer(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.IGNORECASE | re.DOTALL):
        try:
            parsed = json.loads(match.group(1).strip())
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", cleaned):
        try:
            parsed, _ = decoder.raw_decode(cleaned[match.start() :])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(cleaned[start : end + 1])
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def clean_answer(text: str) -> str:
    cleaned = strip_code_fences(str(text)).strip().strip("`")
    if "\\boxed" in cleaned:
        boxed = extract_answer(cleaned)
        if boxed:
            return boxed.strip()
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if not lines:
        return ""
    final = lines[-1].strip()
    final = re.sub(r"^(final answer|answer)\s*[:=]\s*", "", final, flags=re.IGNORECASE)
    return final.rstrip(".").strip()


def clean_option_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def sample_items(items: list[dict[str, Any]], sample_size: int, seed: int, label: str) -> list[dict[str, Any]]:
    if len(items) < sample_size:
        raise RuntimeError(f"{label} needs {sample_size} rows, found {len(items)}")
    return random.Random(seed).sample(items, sample_size)


def exclude_items(items: list[dict[str, Any]], excluded_ids: set[str]) -> list[dict[str, Any]]:
    return [item for item in items if item["unique_id"] not in excluded_ids]


def sample_disjoint_benchmark_pools(
    items: list[dict[str, Any]],
    train_size: int,
    validation_size: int,
    test_size: int,
    seed: int,
    label: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    total_needed = train_size + validation_size + test_size
    if len(items) < total_needed:
        raise RuntimeError(f"{label} split needs {total_needed} rows, found {len(items)}")
    train_pool = sample_items(items, train_size, seed + 101, f"{label} train pool")
    train_ids = {item["unique_id"] for item in train_pool}
    validation_candidates = exclude_items(items, train_ids)
    validation_pool = sample_items(validation_candidates, validation_size, seed + 102, f"{label} validation pool")
    validation_ids = {item["unique_id"] for item in validation_pool}
    test_candidates = exclude_items(validation_candidates, validation_ids)
    test_pool = sample_items(test_candidates, test_size, seed + 201, f"{label} final test pool")
    metadata = {
        "enabled": True,
        "source_candidate_count": len(items),
        "train_candidate_count": len(items),
        "validation_candidate_count": len(validation_candidates),
        "test_candidate_count": len(test_candidates),
        "train_sample_size": len(train_pool),
        "validation_sample_size": len(validation_pool),
        "test_sample_size": len(test_pool),
        "train_subject_counts": subject_counts(train_pool),
        "validation_subject_counts": subject_counts(validation_pool),
        "test_subject_counts": subject_counts(test_pool),
        "train_sample_seed": seed + 101,
        "validation_sample_seed": seed + 102,
        "test_sample_seed": seed + 201,
        "sampling": "train sampled first, validation sampled from remaining, test sampled from remaining",
    }
    return train_pool, validation_pool, test_pool, metadata


def sample_deepmath_remaining_test_pool(
    all_items: list[dict[str, Any]],
    train_pool: list[dict[str, Any]],
    validation_pool: list[dict[str, Any]],
    test_size: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    used_ids = {item["unique_id"] for item in train_pool + validation_pool}
    allowed_subjects = {str(item.get("subject", "")) for item in train_pool + validation_pool}
    candidates = [
        item
        for item in all_items
        if item.get("benchmark") == "DeepMath-103K"
        and item["unique_id"] not in used_ids
        and str(item.get("subject", "")) in allowed_subjects
    ]
    test_pool = sample_items(candidates, test_size, seed + 201, "DeepMath remaining subject-matched final test pool")
    metadata = {
        "enabled": True,
        "source_candidate_count": len(all_items),
        "excluded_train_validation_count": len(used_ids),
        "allowed_subject_count": len(allowed_subjects),
        "allowed_subjects": sorted(allowed_subjects),
        "test_candidate_count": len(candidates),
        "test_sample_size": len(test_pool),
        "test_subject_counts": subject_counts(test_pool),
        "test_sample_seed": seed + 201,
        "sampling": "remaining DeepMath rows after train/validation, filtered to subjects seen in train or validation",
    }
    return test_pool, metadata


def subject_counts(items: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(item.get("subject", "")) for item in items).most_common())


def write_items_jsonl(path: Path, items: list[dict[str, Any]]) -> None:
    if path.exists():
        path.unlink()
    append_jsonl(path, items)


def read_items_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def display_path(path: Path | str) -> str:
    resolved = Path(path).expanduser().resolve()
    return os.path.relpath(resolved, IMPLEMENTATION_DIR)


def normalize_bbh_row(raw: dict[str, Any], index: int, source_name: str) -> dict[str, Any]:
    if "input" not in raw or "target" not in raw:
        raise ValueError("BBH row is missing input/target")
    task = clean_option_text(raw.get("task", raw.get("subject", source_name))) or "bigbenchhard"
    unique_id = clean_option_text(raw.get("unique_id") or raw.get("id") or f"{task}:{index}")
    return {
        "id": f"BBH-{index:05d}",
        "unique_id": unique_id,
        "benchmark": "BigBenchHard",
        "answer_type": "free_form",
        "subject": task,
        "level": 0,
        "question": str(raw["input"]),
        "solution": str(raw["target"]),
        "ground_truth": str(raw["target"]),
        "task": task,
    }


def normalize_for_text_match(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def item_result_metadata(item: dict[str, Any]) -> dict[str, Any]:
    metadata = {
        "benchmark_id": item["id"],
        "unique_id": item["unique_id"],
        "benchmark": item.get("benchmark", "MATH"),
        "answer_type": item.get("answer_type", "free_form"),
        "subject": item["subject"],
        "level": item["level"],
        "ground_truth": item["ground_truth"],
    }
    return metadata


def benchmark_name_from_path(path: Path) -> str:
    for part in reversed(path.parts):
        upper = part.upper()
        if "BIGBENCHHARD" in upper or upper == "BBH":
            return "BigBenchHard"
        if upper.startswith("AIME"):
            return upper
        if "MINERVA" in upper:
            return "Minerva"
        if "DEEPMATH" in upper:
            return "DeepMath-103K"
        if upper == "MATH-BENCHMARK" or upper == "MATH":
            return "MATH"
    return "MATH"


def read_benchmark_jsonl(path: Path) -> list[dict[str, Any]]:
    benchmark_name = benchmark_name_from_path(path)
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for index, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if {"input", "target"}.issubset(raw):
                rows.append(normalize_bbh_row(raw, index, path.name))
                continue
            if {"question", "ground_truth", "unique_id", "benchmark"}.issubset(raw):
                rows.append(
                    {
                        "id": str(raw.get("id", f"{benchmark_name}-{index:05d}")),
                        "unique_id": str(raw["unique_id"]),
                        "benchmark": str(raw["benchmark"]),
                        "answer_type": str(raw.get("answer_type", "free_form")),
                        "subject": str(raw.get("subject", raw["benchmark"])),
                        "level": raw.get("level", 0),
                        "question": str(raw["question"]),
                        "solution": str(raw.get("solution", raw["ground_truth"])),
                        "ground_truth": str(raw["ground_truth"]),
                    }
                )
                continue
            if {"problem", "answer", "id"}.issubset(raw):
                rows.append(
                    {
                        "id": f"{benchmark_name}-{index:05d}",
                        "unique_id": str(raw["id"]),
                        "benchmark": benchmark_name,
                        "answer_type": "free_form",
                        "subject": benchmark_name,
                        "level": 0,
                        "question": str(raw["problem"]),
                        "solution": str(raw.get("solution", raw["answer"])),
                        "ground_truth": clean_answer(str(raw["answer"])),
                    }
                )
                continue
            if benchmark_name == "Minerva" and {"question", "answer"}.issubset(raw):
                rows.append(
                    {
                        "id": f"{benchmark_name}-{index:05d}",
                        "unique_id": str(raw.get("id", f"{path.name}:{index}")),
                        "benchmark": benchmark_name,
                        "answer_type": "free_form",
                        "subject": benchmark_name,
                        "level": 0,
                        "question": str(raw["question"]),
                        "solution": str(raw.get("solution", raw["answer"])),
                        "ground_truth": clean_answer(str(raw["answer"])),
                    }
                )
                continue
            known = ", ".join(sorted(str(key) for key in raw))
            raise RuntimeError(f"Unsupported benchmark jsonl schema for {path}: {known}")
    return rows


def read_benchmark_csv(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for index, raw in enumerate(reader, start=1):
            if {"input", "target"}.issubset(raw):
                rows.append(normalize_bbh_row(raw, index, path.name))
                continue
            known = ", ".join(sorted(str(key) for key in raw))
            raise RuntimeError(f"Unsupported benchmark csv schema for {path}: {known}")
    return rows


def read_benchmark_parquet(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return read_benchmark_jsonl(path)
    if path.suffix.lower() == ".csv":
        return read_benchmark_csv(path)
    if pq is None:
        raise RuntimeError("pyarrow is required to read parquet datasets")
    table = pq.read_table(path)
    benchmark_name = benchmark_name_from_path(path)
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(table.to_pylist(), start=1):
        if {"input", "target"}.issubset(raw):
            rows.append(normalize_bbh_row(raw, index, path.name))
            continue
        if {"unique_id", "subject", "level", "problem", "solution", "answer"}.issubset(raw):
            rows.append(
                {
                    "id": f"MATH-{index:05d}",
                    "unique_id": str(raw["unique_id"]),
                    "benchmark": "MATH",
                    "answer_type": "free_form",
                    "subject": str(raw["subject"]),
                    "level": int(raw["level"]),
                    "question": str(raw["problem"]),
                    "solution": str(raw["solution"]),
                    "ground_truth": str(raw["answer"]),
                }
            )
            continue
        if {"id", "problem", "solution", "url"}.issubset(raw):
            solution = str(raw["solution"])
            rows.append(
                {
                    "id": f"{benchmark_name}-{index:05d}",
                    "unique_id": str(raw["id"]),
                    "benchmark": benchmark_name,
                    "answer_type": "free_form",
                    "subject": benchmark_name,
                    "level": 0,
                    "question": str(raw["problem"]),
                    "solution": solution,
                    "ground_truth": clean_answer(solution),
                    "url": str(raw["url"]),
                }
            )
            continue
        if {"question", "final_answer", "difficulty", "topic"}.issubset(raw):
            rows.append(
                {
                    "id": f"DeepMath-103K-{index:06d}",
                    "unique_id": f"{path.name}:{index}",
                    "benchmark": "DeepMath-103K",
                    "answer_type": "free_form",
                    "subject": str(raw["topic"]),
                    "level": raw["difficulty"],
                    "question": str(raw["question"]),
                    "solution": str(raw.get("r1_solution_1", "")),
                    "ground_truth": str(raw["final_answer"]),
                }
            )
            continue
        known = ", ".join(sorted(str(key) for key in raw))
        raise RuntimeError(f"Unsupported benchmark parquet schema for {path}: {known}")
    return rows


def read_math_parquet(path: Path) -> list[dict[str, Any]]:
    return read_benchmark_parquet(path)


def read_benchmark_dataset(path: Path) -> list[dict[str, Any]]:
    return read_benchmark_parquet(path)


def grade_item_answer(selected_answer: str, item: dict[str, Any]) -> bool:
    if item.get("benchmark") == "BigBenchHard":
        selected = clean_answer(selected_answer)
        target = clean_answer(str(item["ground_truth"]))
        if selected.strip() == target.strip():
            return True
        selected_norm = normalize_for_text_match(selected)
        target_norm = normalize_for_text_match(target)
        return bool(selected_norm) and selected_norm == target_norm
    return bool(selected_answer) and bool(grade(selected_answer, item["ground_truth"]))


def load_openai_credentials() -> tuple[str | None, str | None]:
    api_key = os.environ.get("OPENAI_API_KEY")
    org_key = os.environ.get("OPENAI_ORG_ID") or os.environ.get("OPENAI_ORG_KEY")
    return api_key, org_key


def load_anthropic_credentials() -> str | None:
    return os.environ.get("ANTHROPIC_API_KEY")


def extract_response_text(payload: dict[str, Any]) -> str:
    if payload.get("output_text"):
        return str(payload["output_text"]).strip()
    fragments: list[str] = []
    for item in payload.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                fragments.append(str(content.get("text", "")))
    return "\n".join(fragment for fragment in fragments if fragment).strip()


class LLMClient:
    def __init__(self, backend: str, model: str, api_base: str) -> None:
        self.backend = backend
        self.model = model
        self.api_base = api_base

    def text(self, prompt: str, max_output_tokens: int, seed: int | None = None) -> str:
        if self.backend == "mock":
            return self._mock_text(prompt, seed)
        if self.backend == "local-openai":
            return self._local_openai_text(prompt, max_output_tokens, seed)
        if self.backend == "openai":
            return self._openai_text(prompt, max_output_tokens, seed)
        if self.backend == "anthropic":
            return self._anthropic_text(prompt, max_output_tokens)
        raise RuntimeError(f"Unsupported backend: {self.backend}")

    def json(self, prompt: str, max_output_tokens: int, seed: int | None = None) -> dict[str, Any]:
        text = self.text(prompt, max_output_tokens=max_output_tokens, seed=seed)
        parsed = extract_json_object(text)
        if parsed is None:
            raise RuntimeError("Model response did not contain a JSON object")
        return parsed

    def _local_openai_text(self, prompt: str, max_output_tokens: int, seed: int | None) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_output_tokens,
            "temperature": 0,
        }
        if seed is not None:
            payload["seed"] = seed
        request = urllib.request.Request(
            f"{self.api_base.rstrip('/')}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": "Bearer local"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=600) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
        choices = response_payload.get("choices", [])
        if not choices:
            return ""
        return str(choices[0].get("message", {}).get("content", "")).strip()

    def _openai_text(self, prompt: str, max_output_tokens: int, seed: int | None) -> str:
        api_key, org_key = load_openai_credentials()
        if not api_key:
            raise RuntimeError("OpenAI API key not found in OPENAI_API_KEY")
        payload: dict[str, Any] = {
            "model": self.model,
            "input": prompt,
            "max_output_tokens": max_output_tokens,
            "reasoning": {"effort": "low"},
            "text": {"verbosity": "low"},
        }
        if seed is not None:
            payload["seed"] = seed
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        if org_key:
            headers["OpenAI-Organization"] = org_key

        last_error: Exception | None = None
        for attempt in range(4):
            try:
                request = urllib.request.Request(
                    f"{self.api_base.rstrip('/')}/responses",
                    data=json.dumps(payload).encode("utf-8"),
                    headers=headers,
                    method="POST",
                )
                with urllib.request.urlopen(request, timeout=240) as response:
                    return extract_response_text(json.loads(response.read().decode("utf-8")))
            except urllib.error.HTTPError as exc:
                last_error = exc
                if "seed" in payload and exc.code in {400, 404}:
                    payload.pop("seed", None)
                    continue
                if attempt == 3:
                    break
                time.sleep(2 * (attempt + 1))
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt == 3:
                    break
                time.sleep(2 * (attempt + 1))
        raise last_error if last_error is not None else RuntimeError("OpenAI request failed")

    def _anthropic_text(self, prompt: str, max_output_tokens: int) -> str:
        api_key = load_anthropic_credentials()
        if not api_key:
            raise RuntimeError("Anthropic API key not found in ANTHROPIC_API_KEY")
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_output_tokens,
            "temperature": 0,
            "messages": [{"role": "user", "content": prompt}],
        }
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

        last_error: Exception | None = None
        for attempt in range(4):
            try:
                request = urllib.request.Request(
                    "https://api.anthropic.com/v1/messages",
                    data=json.dumps(payload).encode("utf-8"),
                    headers=headers,
                    method="POST",
                )
                with urllib.request.urlopen(request, timeout=240) as response:
                    response_payload = json.loads(response.read().decode("utf-8"))
                fragments = [
                    str(block.get("text", ""))
                    for block in response_payload.get("content", [])
                    if block.get("type") == "text"
                ]
                return "\n".join(fragment for fragment in fragments if fragment).strip()
            except urllib.error.HTTPError as exc:
                last_error = exc
                if attempt == 3:
                    break
                time.sleep(2 * (attempt + 1))
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt == 3:
                    break
                time.sleep(2 * (attempt + 1))
        raise last_error if last_error is not None else RuntimeError("Anthropic request failed")

    def _mock_text(self, prompt: str, seed: int | None) -> str:
        prompt_lower = prompt.lower()
        digest = hashlib.sha256(f"{seed}:{prompt}".encode("utf-8")).hexdigest()
        if "return exactly one json object" in prompt_lower:
            answer_match = re.search(r"Reference answer for mock backend:\s*(.+)", prompt)
            answer = answer_match.group(1).strip() if answer_match else str(int(digest[:2], 16) % 10)
            return json.dumps(
                {
                    "reasoning": "mock_reasoning: identify the required result, apply the current skill, and check the answer.",
                    "answer": answer,
                }
            )
        if "return only the update direction" in prompt_lower:
            return "Improve by checking intermediate assumptions, comparing peer strategies, and finalizing only after a brief verification."
        if "return a concise natural-language velocity" in prompt_lower:
            return "Move toward more explicit principle identification, targeted verification, and concise final review while preserving the agent's original style."
        if "return only the updated skill" in prompt_lower:
            return "\n".join(
                [
                    "- Preserve the agent's original reasoning role.",
                    "- Identify the problem type and useful general principle before detailed work.",
                    "- Solve step by step with concise intermediate checks.",
                    "- Verify assumptions, computations, and the final answer.",
                    "- Revise only when review reveals a concrete issue.",
                ]
            )
        return "mock_response"


def base_prompt(skill_text: str, benchmark: str = "MATH") -> str:
    benchmark_label = str(benchmark or "MATH")
    if benchmark_label == "BigBenchHard":
        return f"""
You are a BIG-Bench Hard reasoning agent.

Base rules:
- Solve the provided BIG-Bench Hard problem.
- The answer is usually a short word, phrase, option, sequence, or logical result.
- Do not invent missing problem details.
- Follow the current skill file below.

Current skill file:
{skill_text}
        """.strip()
    return f"""
You are a mathematics competition agent.

Base rules:
- Solve the provided {benchmark_label} problem.
- The answer may be a number, expression, tuple, interval, or simplified LaTeX expression.
- Do not invent missing problem details.
- Follow the current skill file below.

Current skill file:
{skill_text}
    """.strip()


def build_solve_prompt(item: dict[str, Any], agent_id: int, skill_text: str, include_mock_answer: bool) -> str:
    mock_line = f"\nReference answer for mock backend: {item['ground_truth']}" if include_mock_answer else ""
    return f"""
{base_prompt(skill_text, item.get('benchmark', 'MATH'))}

Solve this problem using only your current skill.

Problem:
{item['question']}
{mock_line}

Return exactly one JSON object:
{{
  "agent_id": {agent_id},
  "reasoning": "...",
  "answer": "..."
}}
    """.strip()


def prompt_a_direction(current_skill: str, own_outputs: list[dict[str, Any]], peer_outputs: list[dict[str, Any]]) -> str:
    return f"""
Prompt A: Generate self-reflective direction

Current agent skill:
{current_skill}

Agent's own reasoning traces and answers:
{json.dumps(own_outputs, ensure_ascii=False, indent=2)}

Other agents' reasoning traces and answers, including correctness:
{json.dumps(peer_outputs, ensure_ascii=False, indent=2)}

Instruction:
Analyze the agent's performance compared with peers.
Identify general reasoning improvements.
Do not overfit to a single problem.
Do not rewrite the skill yet.
Return only the update direction.
    """.strip()


def prompt_b_velocity(
    previous_velocity: str,
    direction: str,
    current_skill: str,
    personal_best_skill: str,
    global_best_skill: str,
    agent_identity: str,
    max_velocity_words: int,
) -> str:
    return f"""
Prompt B: Generate PSO-guided velocity

Agent identity to preserve:
{agent_identity}

Previous velocity v_i:
{previous_velocity}

Self-reflective direction d_i:
{direction}

Current skill s_i:
{current_skill}

Personal best skill p_i:
{personal_best_skill}

Global best skill g:
{global_best_skill}

Instruction:
Combine the previous velocity, self-reflective direction, lessons from the personal best, and lessons from the global best.
Focus on generalizable improvements.
Do not copy the personal best or global best directly.
Preserve the agent's identity.
Return a concise natural-language velocity of at most {max_velocity_words} words.
    """.strip()


def safe_problem_for_test_feedback(item: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "benchmark_id": item.get("id", ""),
        "unique_id": item.get("unique_id", ""),
        "benchmark": item.get("benchmark", ""),
        "answer_type": item.get("answer_type", ""),
        "subject": item.get("subject", ""),
        "level": item.get("level", ""),
        "question": item.get("question", ""),
    }
    if item.get("answer_type") == "multiple_choice":
        payload["options"] = item.get("options", {})
        payload["choice_type"] = item.get("choice_type", "")
        payload["topic"] = item.get("topic", "")
    return payload


def safe_response_for_test_feedback(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "agent_id": int(row["agent_id"]),
        "selected_answer": str(row.get("selected_answer", "")),
        "reasoning": str(row.get("reasoning", "")),
    }


def prompt_test_feedback_direction(
    current_skill: str,
    own_response: dict[str, Any],
    peer_responses: list[dict[str, Any]],
    problem: dict[str, Any],
) -> str:
    return f"""
Prompt A-test: Generate problem-specific peer feedback direction

This is test-time adaptation. Do not use or infer the ground-truth answer.

Current agent skill:
{current_skill}

Problem:
{json.dumps(problem, ensure_ascii=False, indent=2)}

Agent's own initial solution:
{json.dumps(own_response, ensure_ascii=False, indent=2)}

Other agents' initial solutions:
{json.dumps(peer_responses, ensure_ascii=False, indent=2)}

Instruction:
Analyze the agent's initial solution compared with the peer solutions for this specific problem.
Identify what the agent should reconsider, revise, verify, or improve before answering again.
Focus on disagreement, useful evidence, missing discriminators, and possible near-miss errors.
Do not decide by majority alone.
Do not rewrite the skill.
Return only the problem-specific feedback direction d_i.
    """.strip()


def prompt_test_time_velocity(
    direction: str,
    current_skill: str,
    global_best_skill: str,
    agent_identity: str,
    max_velocity_words: int,
    is_global_best: bool,
) -> str:
    if is_global_best:
        return f"""
Prompt B-test: Generate temporary test-time velocity

Agent identity to preserve:
{agent_identity}

Problem-specific peer feedback d_i:
{direction}

Instruction:
Generate a temporary test-time velocity for this one problem.
Use only the problem-specific feedback above.
Do not compare this agent against a global-best skill, because this agent is already the global-best agent.
Focus on how the agent should temporarily adjust its reasoning before re-answering this problem.
Do not update any saved skill.
Return a concise natural-language velocity of at most {max_velocity_words} words.
        """.strip()

    return f"""
Prompt B-test: Generate temporary test-time velocity

Agent identity to preserve:
{agent_identity}

Problem-specific peer feedback d_i:
{direction}

Current personal-best skill s_i:
{current_skill}

Global-best skill g:
{global_best_skill}

Instruction:
Generate a temporary test-time velocity for this one problem.
Infer how the current agent should temporarily adjust its reasoning by combining the problem-specific feedback with useful lessons from the global-best skill.
Do not copy the global-best skill directly.
Do not update any saved skill.
Return a concise natural-language velocity of at most {max_velocity_words} words.
    """.strip()


def skill_with_temporary_velocity(skill_text: str, velocity: str) -> str:
    velocity_text = strip_code_fences(velocity).strip()
    if not velocity_text:
        return skill_text
    return f"""
{skill_text}

Temporary test-time guidance for this problem only:
{velocity_text}
    """.strip()


def prompt_c_apply_velocity(current_skill: str, velocity: str, agent_identity: str, max_skill_words: int) -> str:
    return f"""
Prompt C: Apply velocity to skill

Agent identity to preserve:
{agent_identity}

Current skill s_i:
{current_skill}

Velocity v_i:
{velocity}

Instruction:
Rewrite the skill according to the velocity.
Keep the skill concise and general.
Preserve the agent's original role.
Remove redundant, overly specific, or contradictory instructions.
Use at most 10 bullet points and at most {max_skill_words} words.
Do not copy another skill verbatim.
Return only the updated skill.
    """.strip()


def solve_item(
    client: LLMClient,
    item: dict[str, Any],
    agent_id: int,
    skill_text: str,
    iteration: int,
    phase: str,
    seed: int,
) -> dict[str, Any]:
    prompt = build_solve_prompt(item, agent_id, skill_text, include_mock_answer=client.backend == "mock")
    try:
        text = client.text(prompt, max_output_tokens=1800, seed=seed)
        payload = extract_json_object(text)
        if payload is None:
            reasoning = text
            selected_answer = clean_answer(text)
        else:
            reasoning = str(payload.get("reasoning", ""))
            selected_answer = clean_answer(str(payload.get("answer", "")))
    except Exception as exc:
        reasoning = f"api_error: {type(exc).__name__}: {exc}"
        selected_answer = ""
    correct = grade_item_answer(selected_answer, item)
    return {
        **item_result_metadata(item),
        "iteration": iteration,
        "phase": phase,
        "agent_id": agent_id,
        "selected_answer": selected_answer,
        "correct": correct,
        "reasoning": reasoning,
    }


def compute_test_feedback_direction(
    client: LLMClient,
    item: dict[str, Any],
    agent_id: int,
    current_skill: str,
    own_response: dict[str, Any],
    peer_responses: list[dict[str, Any]],
    seed: int,
    max_output_tokens: int,
) -> str:
    prompt = prompt_test_feedback_direction(
        current_skill=current_skill,
        own_response=safe_response_for_test_feedback(own_response),
        peer_responses=[safe_response_for_test_feedback(row) for row in peer_responses],
        problem=safe_problem_for_test_feedback(item),
    )
    try:
        return client.text(prompt, max_output_tokens=max_output_tokens, seed=seed).strip()
    except Exception as exc:
        return f"api_error: {type(exc).__name__}: {exc}"


def generate_test_time_velocity(
    client: LLMClient,
    agent_id: int,
    direction: str,
    current_skill: str,
    global_best_skill: str,
    is_global_best: bool,
    seed: int,
    max_velocity_words: int,
    max_output_tokens: int,
) -> str:
    prompt = prompt_test_time_velocity(
        direction=direction,
        current_skill=current_skill,
        global_best_skill=global_best_skill,
        agent_identity=INITIAL_SKILLS[agent_id]["identity"],
        max_velocity_words=max_velocity_words,
        is_global_best=is_global_best,
    )
    try:
        return client.text(prompt, max_output_tokens=max_output_tokens, seed=seed).strip()
    except Exception as exc:
        return f"api_error: {type(exc).__name__}: {exc}"


def evaluate_skills(
    client: LLMClient,
    validation_items: list[dict[str, Any]],
    skills: dict[int, str],
    iteration: int,
    seed: int,
    output_path: Path,
    validation_subset_idx: int | None = None,
    validation_subset_range: list[int] | None = None,
) -> dict[int, dict[str, Any]]:
    if output_path.exists():
        output_path.unlink()
    summaries: dict[int, dict[str, Any]] = {}
    for agent_id, skill_text in skills.items():
        correct_count = 0
        for index, item in enumerate(validation_items):
            row = solve_item(
                client,
                item,
                agent_id,
                skill_text,
                iteration=iteration,
                phase="validation",
                seed=seed + iteration * 100000 + agent_id * 1000 + index,
            )
            if validation_subset_idx is not None:
                row["validation_subset_idx"] = validation_subset_idx
            if validation_subset_range is not None:
                row["validation_subset_range"] = validation_subset_range
            correct_count += int(row["correct"])
            append_jsonl(output_path, [row])
        completed = len(validation_items)
        summaries[agent_id] = {
            "agent_id": agent_id,
            "iteration": iteration,
            "score": correct_count / completed if completed else 0.0,
            "correct_count": correct_count,
            "question_count": completed,
        }
        if validation_subset_idx is not None:
            summaries[agent_id]["validation_subset_idx"] = validation_subset_idx
        if validation_subset_range is not None:
            summaries[agent_id]["validation_subset_range"] = validation_subset_range
        print(
            f"validation_checkpoint iteration={iteration:03d} agent={agent_id} "
            f"correct={correct_count}/{completed} accuracy={summaries[agent_id]['score']:.4f}",
            flush=True,
        )
    return summaries


def evaluate_single_skill(
    client: LLMClient,
    validation_items: list[dict[str, Any]],
    agent_id: int,
    skill_text: str,
    iteration: int,
    phase: str,
    seed: int,
    output_path: Path,
    validation_subset_idx: int | None = None,
    validation_subset_range: list[int] | None = None,
) -> dict[str, Any]:
    if output_path.exists():
        output_path.unlink()
    correct_count = 0
    for index, item in enumerate(validation_items):
        row = solve_item(
            client,
            item,
            agent_id,
            skill_text,
            iteration=iteration,
            phase=phase,
            seed=seed + iteration * 100000 + agent_id * 1000 + index,
        )
        if validation_subset_idx is not None:
            row["validation_subset_idx"] = validation_subset_idx
        if validation_subset_range is not None:
            row["validation_subset_range"] = validation_subset_range
        correct_count += int(row["correct"])
        append_jsonl(output_path, [row])
    completed = len(validation_items)
    summary = {
        "agent_id": agent_id,
        "iteration": iteration,
        "phase": phase,
        "score": correct_count / completed if completed else 0.0,
        "correct_count": correct_count,
        "question_count": completed,
    }
    if validation_subset_idx is not None:
        summary["validation_subset_idx"] = validation_subset_idx
    if validation_subset_range is not None:
        summary["validation_subset_range"] = validation_subset_range
    print(
        f"{phase}_checkpoint iteration={iteration:03d} agent={agent_id} "
        f"correct={correct_count}/{completed} accuracy={summary['score']:.4f}",
        flush=True,
    )
    return summary


def validation_subset_count(validation_pool_size: int, validation_subset_size: int) -> int:
    if validation_subset_size <= 0:
        raise ValueError("validation subset size must be positive")
    count = validation_pool_size // validation_subset_size
    if count < 1:
        raise ValueError(
            f"Validation pool has {validation_pool_size} examples, fewer than subset size {validation_subset_size}"
        )
    return count


def validation_state_path(run_dir: Path) -> Path:
    return run_dir / "state" / "validation_state.json"


def load_validation_state(run_dir: Path, validation_pool_size: int, validation_subset_size: int) -> dict[str, Any]:
    subset_count = validation_subset_count(validation_pool_size, validation_subset_size)
    path = validation_state_path(run_dir)
    subset_idx = 0
    if path.exists():
        payload = json.loads(path.read_text())
        subset_idx = int(payload.get("validation_subset_idx", 0)) % subset_count
    return {
        "validation_subset_idx": subset_idx,
        "validation_subset_size": validation_subset_size,
        "validation_subset_count": subset_count,
        "validation_pool_size": validation_pool_size,
    }


def save_validation_state(run_dir: Path, payload: dict[str, Any]) -> None:
    save_json(validation_state_path(run_dir), payload)


def select_validation_subset(validation_pool: list[dict[str, Any]], subset_idx: int, subset_size: int) -> tuple[list[dict[str, Any]], list[int]]:
    start = subset_idx * subset_size
    end = start + subset_size
    if end > len(validation_pool):
        raise ValueError(
            f"Validation subset {subset_idx} range {start}-{end - 1} exceeds pool size {len(validation_pool)}"
        )
    return validation_pool[start:end], [start, end - 1]


def begin_validation_subset(run_dir: Path, validation_pool: list[dict[str, Any]], subset_size: int, iteration: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    state = load_validation_state(run_dir, len(validation_pool), subset_size)
    subset_idx = int(state["validation_subset_idx"])
    items, subset_range = select_validation_subset(validation_pool, subset_idx, subset_size)
    state = {
        **state,
        "iteration": iteration,
        "active_validation_subset_idx": subset_idx,
        "active_validation_subset_range": subset_range,
        "status": "in_progress",
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    }
    save_validation_state(run_dir, state)
    print(
        f"validation_subset_start iteration={iteration:03d} subset={subset_idx} "
        f"range={subset_range[0]}-{subset_range[1]}",
        flush=True,
    )
    return items, state


def finish_validation_subset(run_dir: Path, state: dict[str, Any], summaries: dict[int, dict[str, Any]]) -> dict[str, Any]:
    subset_idx = int(state["active_validation_subset_idx"])
    subset_count = int(state["validation_subset_count"])
    subset_size = int(state["validation_subset_size"])
    subset_range = list(state["active_validation_subset_range"])
    perfect_agents = [
        int(agent_id)
        for agent_id, summary in summaries.items()
        if int(summary.get("correct_count", 0)) == subset_size
    ]
    is_perfect = bool(perfect_agents)
    next_subset_idx = (subset_idx + 1) % subset_count if is_perfect else subset_idx
    score_text = ", ".join(
        f"{agent_id}:{summaries[agent_id]['correct_count']}/{summaries[agent_id]['question_count']}"
        for agent_id in sorted(summaries)
    )
    next_state = {
        "validation_subset_idx": next_subset_idx,
        "validation_subset_size": subset_size,
        "validation_subset_count": subset_count,
        "validation_pool_size": int(state["validation_pool_size"]),
        "last_validation_iteration": int(state["iteration"]),
        "last_validation_subset_idx": subset_idx,
        "last_validation_subset_range": subset_range,
        "last_validation_scores": {
            str(agent_id): {
                "correct_count": int(summary["correct_count"]),
                "question_count": int(summary["question_count"]),
                "score": float(summary["score"]),
            }
            for agent_id, summary in summaries.items()
        },
        "last_validation_perfect": is_perfect,
        "last_validation_perfect_agents": perfect_agents,
        "status": "ready",
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    }
    save_validation_state(run_dir, next_state)
    print(
        f"validation_subset_result iteration={state['iteration']:03d} subset={subset_idx} "
        f"range={subset_range[0]}-{subset_range[1]} scores={{{score_text}}} "
        f"perfect={is_perfect} perfect_agents={perfect_agents} next_subset={next_subset_idx}",
        flush=True,
    )
    return next_state


def write_skill(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(strip_code_fences(text).strip() + "\n")


def read_skill(path: Path) -> str:
    return path.read_text().strip()


def add_boolean_optional_argument(
    parser: argparse.ArgumentParser,
    name: str,
    default: bool,
    help: str | None = None,
) -> None:
    if hasattr(argparse, "BooleanOptionalAction"):
        parser.add_argument(name, action=argparse.BooleanOptionalAction, default=default, help=help)
        return
    dest = name.lstrip("-").replace("-", "_")
    parser.add_argument(name, dest=dest, action="store_true", default=default, help=help)
    parser.add_argument(f"--no-{name.lstrip('-')}", dest=dest, action="store_false")


def initialize_run(args: argparse.Namespace, run_dir: Path, client: LLMClient) -> dict[str, Any]:
    train_all = read_benchmark_dataset(Path(args.train_dataset).resolve())
    deepmath_sampling_metadata: dict[str, Any] = {}
    bbh_sampling_metadata: dict[str, Any] = {}
    if args.dataset_preset == "bbh":
        test_sample_size = args.test_limit if args.test_limit is not None else 200
        train_pool, validation_pool, bbh_test_pool, bbh_sampling_metadata = sample_disjoint_benchmark_pools(
            train_all,
            args.train_pool_size,
            args.validation_pool_size,
            test_sample_size,
            args.seed,
            "BigBenchHard",
        )
        validation_indices = "random_disjoint_sample_from_bbh_source"
        bbh_train_pool_path = run_dir / "data" / "bbh_train_pool.jsonl"
        bbh_validation_pool_path = run_dir / "data" / "bbh_validation_pool.jsonl"
        bbh_test_pool_path = run_dir / "data" / "bbh_test_pool.jsonl"
        write_items_jsonl(bbh_train_pool_path, train_pool)
        write_items_jsonl(bbh_validation_pool_path, validation_pool)
        write_items_jsonl(bbh_test_pool_path, bbh_test_pool)
        bbh_sampling_metadata["train_pool_path"] = display_path(bbh_train_pool_path)
        bbh_sampling_metadata["validation_pool_path"] = display_path(bbh_validation_pool_path)
        bbh_sampling_metadata["test_pool_path"] = display_path(bbh_test_pool_path)
        args.test_limit = test_sample_size
    elif args.dataset_preset == "deepmath":
        train_pool = train_all[: args.train_pool_size]
        if args.validation_dataset:
            validation_all = read_benchmark_dataset(Path(args.validation_dataset).resolve())
            validation_pool = validation_all[: args.validation_pool_size]
            validation_indices = [0, args.validation_pool_size]
        else:
            validation_pool = train_all[args.train_pool_size : args.train_pool_size + args.validation_pool_size]
            validation_indices = [args.train_pool_size, args.train_pool_size + args.validation_pool_size]
        train_path = Path(args.train_dataset).resolve()
        test_path = Path(args.test_dataset).resolve()
        if test_path == train_path:
            test_sample_size = args.test_limit if args.test_limit is not None else 200
            deepmath_test_pool, deepmath_sampling_metadata = sample_deepmath_remaining_test_pool(
                train_all,
                train_pool,
                validation_pool,
                test_sample_size,
                args.seed,
            )
            deepmath_test_pool_path = run_dir / "data" / "deepmath_test_pool.jsonl"
            write_items_jsonl(deepmath_test_pool_path, deepmath_test_pool)
            deepmath_sampling_metadata["test_pool_path"] = display_path(deepmath_test_pool_path)
            deepmath_sampling_metadata["train_subject_counts"] = subject_counts(train_pool)
            deepmath_sampling_metadata["validation_subject_counts"] = subject_counts(validation_pool)
            args.test_limit = test_sample_size
    else:
        train_pool = train_all[: args.train_pool_size]
        if args.validation_dataset:
            validation_all = read_benchmark_dataset(Path(args.validation_dataset).resolve())
            validation_pool = validation_all[: args.validation_pool_size]
            validation_indices = [0, args.validation_pool_size]
        else:
            validation_pool = train_all[args.train_pool_size : args.train_pool_size + args.validation_pool_size]
            validation_indices = [args.train_pool_size, args.train_pool_size + args.validation_pool_size]
    if len(train_pool) < args.train_pool_size:
        raise RuntimeError(f"Training pool needs {args.train_pool_size} rows, found {len(train_pool)}")
    if len(validation_pool) < args.validation_pool_size:
        raise RuntimeError(f"Validation pool needs {args.validation_pool_size} rows, found {len(validation_pool)}")

    config = {
        "mode": "agent_pso_natural_language_skill_evolution",
        "dataset_preset": args.dataset_preset,
        "num_agents": args.num_agents,
        "initial_skills": [INITIAL_SKILLS[i]["name"] for i in range(1, args.num_agents + 1)],
        "num_iterations": args.num_iterations,
        "train_batch_size": args.train_batch_size,
        "validation_batch_size": args.validation_batch_size,
        "train_pool_size": args.train_pool_size,
        "validation_pool_size": args.validation_pool_size,
        "validation_subset_size": args.validation_batch_size,
        "validation_subset_count": validation_subset_count(len(validation_pool), args.validation_batch_size),
        "validation_subset_rotation": "advance_to_next_subset_when_any_agent_scores_full_marks",
        "fitness": "accuracy",
        "fitness_accuracy_weight": args.fitness_accuracy_weight,
        "fitness_reasoning_weight": args.fitness_reasoning_weight,
        "epsilon": args.epsilon,
        "preserve_agent_identity": args.preserve_agent_identity,
        "max_skill_words": args.max_skill_words,
        "max_velocity_words": args.max_velocity_words,
        "use_reasoning_quality_score": args.use_reasoning_quality_score,
        "backend": args.backend,
        "model": args.model,
        "api_base": args.api_base if args.backend == "local-openai" else "",
        "train_dataset": display_path(args.train_dataset),
        "validation_dataset": display_path(args.validation_dataset) if args.validation_dataset else "",
        "test_dataset": display_path(args.test_dataset),
        "test_limit": args.test_limit,
        "test_mode": args.test_mode,
        "final_test_enabled": not args.skip_final_test,
        "train_indices": (
            "random_disjoint_sample_from_bbh_source"
            if args.dataset_preset == "bbh"
            else [0, args.train_pool_size]
        ),
        "validation_indices": validation_indices,
        "deepmath_test_sampling": deepmath_sampling_metadata,
        "bbh_sampling": bbh_sampling_metadata,
        "seed": args.seed,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    }
    save_json(run_dir / "config.json", config)

    current_skills: dict[int, str] = {}
    for agent_id in range(1, args.num_agents + 1):
        skill = INITIAL_SKILLS[agent_id]["text"]
        current_skills[agent_id] = skill
        write_skill(run_dir / "skills" / "current" / f"agent_{agent_id}.md", skill)
        write_skill(run_dir / "skills" / "initial" / f"agent_{agent_id}.md", skill)
        write_skill(run_dir / "skills" / "personal_best" / f"agent_{agent_id}.md", skill)
        write_skill(run_dir / "velocities" / "iter_000" / f"agent_{agent_id}_velocity.md", "")

    val_items, validation_state = begin_validation_subset(run_dir, validation_pool, args.validation_batch_size, iteration=0)
    initial_scores = evaluate_skills(
        client,
        val_items,
        current_skills,
        iteration=0,
        seed=args.seed,
        output_path=run_dir / "results" / "validation" / "iter_000.jsonl",
        validation_subset_idx=int(validation_state["active_validation_subset_idx"]),
        validation_subset_range=list(validation_state["active_validation_subset_range"]),
    )
    next_validation_state = finish_validation_subset(run_dir, validation_state, initial_scores)
    personal_best_scores = {
        str(agent_id): {
            "score": summary["score"],
            "iteration": 0,
            "skill_path": display_path(run_dir / "skills" / "personal_best" / f"agent_{agent_id}.md"),
        }
        for agent_id, summary in initial_scores.items()
    }
    best_agent_id = max(initial_scores, key=lambda aid: (initial_scores[aid]["score"], -aid))
    global_best_score = initial_scores[best_agent_id]["score"]
    write_skill(run_dir / "global_best.md", current_skills[best_agent_id])
    save_json(run_dir / "scores" / "personal_best_scores.json", personal_best_scores)
    save_json(
        run_dir / "scores" / "global_best_score.json",
        {
            "score": global_best_score,
            "agent_id": best_agent_id,
            "iteration": 0,
            "skill_path": display_path(run_dir / "global_best.md"),
            "validation_subset_idx": int(validation_state["active_validation_subset_idx"]),
            "validation_subset_range": list(validation_state["active_validation_subset_range"]),
            "score_rebased_at_iteration": 0,
        },
    )
    append_jsonl(
        run_dir / "scores" / "iteration_scores.jsonl",
        [
            {
                "iteration": 0,
                "phase": "initial_validation",
                "scores": {str(agent_id): summary for agent_id, summary in initial_scores.items()},
                "global_best_agent_id": best_agent_id,
                "global_best_score": global_best_score,
                "validation_subset": {
                    "used_subset_idx": int(validation_state["active_validation_subset_idx"]),
                    "used_subset_range": list(validation_state["active_validation_subset_range"]),
                    "perfect": bool(next_validation_state["last_validation_perfect"]),
                    "perfect_agents": list(next_validation_state["last_validation_perfect_agents"]),
                    "next_subset_idx": int(next_validation_state["validation_subset_idx"]),
                },
            }
        ],
    )
    return {
        "train_pool": train_pool,
        "validation_pool": validation_pool,
        "current_skills": current_skills,
        "personal_best_scores": personal_best_scores,
        "global_best_score": global_best_score,
        "global_best_agent_id": best_agent_id,
    }


def load_scores(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    return (
        json.loads((run_dir / "scores" / "personal_best_scores.json").read_text()),
        json.loads((run_dir / "scores" / "global_best_score.json").read_text()),
    )


def global_best_needs_subset_rebase(global_best_score_payload: dict[str, Any], validation_state: dict[str, Any]) -> bool:
    active_subset_idx = int(validation_state["active_validation_subset_idx"])
    active_subset_range = list(validation_state["active_validation_subset_range"])
    if "validation_subset_idx" not in global_best_score_payload:
        return True
    if int(global_best_score_payload.get("validation_subset_idx", -1)) != active_subset_idx:
        return True
    return list(global_best_score_payload.get("validation_subset_range", [])) != active_subset_range


def rebase_global_best_for_validation_subset(
    client: LLMClient,
    run_dir: Path,
    validation_items: list[dict[str, Any]],
    validation_state: dict[str, Any],
    global_best_score_payload: dict[str, Any],
    global_best_skill: str,
    iteration: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if not global_best_needs_subset_rebase(global_best_score_payload, validation_state):
        return global_best_score_payload, None

    active_subset_idx = int(validation_state["active_validation_subset_idx"])
    active_subset_range = list(validation_state["active_validation_subset_range"])
    global_best_agent_id = int(global_best_score_payload.get("agent_id", 1))
    previous_payload = dict(global_best_score_payload)
    summary = evaluate_single_skill(
        client,
        validation_items,
        global_best_agent_id,
        global_best_skill,
        iteration=iteration,
        phase="global_best_rebase",
        seed=seed + 600000,
        output_path=run_dir / "results" / "validation_global_best_rebase" / f"iter_{iteration:03d}.jsonl",
        validation_subset_idx=active_subset_idx,
        validation_subset_range=active_subset_range,
    )
    updated_payload = {
        **global_best_score_payload,
        "score": float(summary["score"]),
        "validation_subset_idx": active_subset_idx,
        "validation_subset_range": active_subset_range,
        "score_rebased_at_iteration": iteration,
        "score_rebase_phase": "global_best_rebase",
        "score_rebase_result_path": display_path(
            run_dir / "results" / "validation_global_best_rebase" / f"iter_{iteration:03d}.jsonl"
        ),
    }
    event = {
        "type": "global_best_subset_rebase",
        "iteration": iteration,
        "agent_id": global_best_agent_id,
        "previous_score": previous_payload.get("score"),
        "new_score": float(summary["score"]),
        "previous_validation_subset_idx": previous_payload.get("validation_subset_idx"),
        "previous_validation_subset_range": previous_payload.get("validation_subset_range"),
        "new_validation_subset_idx": active_subset_idx,
        "new_validation_subset_range": active_subset_range,
        "correct_count": int(summary["correct_count"]),
        "question_count": int(summary["question_count"]),
    }
    save_json(run_dir / "scores" / "global_best_score.json", updated_payload)
    print(
        f"global_best_rebase iteration={iteration:03d} agent={global_best_agent_id} "
        f"subset={active_subset_idx} range={active_subset_range[0]}-{active_subset_range[1]} "
        f"score={float(summary['score']):.4f}",
        flush=True,
    )
    return updated_payload, event


def deterministic_batch(items: list[dict[str, Any]], size: int, iteration: int) -> list[dict[str, Any]]:
    if size >= len(items):
        return list(items)
    start = ((iteration - 1) * size) % len(items)
    doubled = items + items
    return doubled[start : start + size]


def build_observations(train_rows: list[dict[str, Any]], agent_id: int) -> dict[str, Any]:
    by_problem: dict[str, list[dict[str, Any]]] = {}
    for row in train_rows:
        by_problem.setdefault(str(row["benchmark_id"]), []).append(row)
    problems = []
    for problem_id, rows in by_problem.items():
        own = [row for row in rows if int(row["agent_id"]) == agent_id]
        peers = [row for row in rows if int(row["agent_id"]) != agent_id]
        problems.append({"problem_id": problem_id, "own": own, "peers": peers, "all_agents": rows})
    return {"agent_id": agent_id, "problems": problems}


def run_iteration(
    args: argparse.Namespace,
    run_dir: Path,
    client: LLMClient,
    train_pool: list[dict[str, Any]],
    validation_pool: list[dict[str, Any]],
    iteration: int,
) -> None:
    current_skills = {
        agent_id: read_skill(run_dir / "skills" / "current" / f"agent_{agent_id}.md")
        for agent_id in range(1, args.num_agents + 1)
    }
    personal_best_scores, global_best_score_payload = load_scores(run_dir)
    personal_best_skills = {
        agent_id: read_skill(run_dir / "skills" / "personal_best" / f"agent_{agent_id}.md")
        for agent_id in range(1, args.num_agents + 1)
    }
    global_best_skill = read_skill(run_dir / "global_best.md")

    train_items = deterministic_batch(train_pool, args.train_batch_size, iteration)
    train_result_path = run_dir / "results" / "train" / f"iter_{iteration:03d}.jsonl"
    if train_result_path.exists():
        train_result_path.unlink()
    train_rows: list[dict[str, Any]] = []
    for item_index, item in enumerate(train_items):
        for agent_id, skill_text in current_skills.items():
            row = solve_item(
                client,
                item,
                agent_id,
                skill_text,
                iteration=iteration,
                phase="train",
                seed=args.seed + iteration * 100000 + agent_id * 1000 + item_index,
            )
            train_rows.append(row)
            append_jsonl(train_result_path, [row])

    directions: dict[int, str] = {}
    velocities: dict[int, str] = {}
    updated_skills: dict[int, str] = {}
    for agent_id in range(1, args.num_agents + 1):
        observation = build_observations(train_rows, agent_id)
        observation_path = run_dir / "observations" / f"iter_{iteration:03d}" / f"agent_{agent_id}_observation.json"
        save_json(observation_path, observation)
        own_outputs = [row for row in train_rows if int(row["agent_id"]) == agent_id]
        peer_outputs = [row for row in train_rows if int(row["agent_id"]) != agent_id]

        direction_prompt = prompt_a_direction(current_skills[agent_id], own_outputs, peer_outputs)
        save_json(
            run_dir / "prompts" / f"iter_{iteration:03d}" / f"agent_{agent_id}_direction_prompt.json",
            {"prompt": direction_prompt},
        )
        direction = client.text(
            direction_prompt,
            max_output_tokens=args.max_velocity_tokens,
            seed=args.seed + iteration * 10000 + agent_id * 100 + 1,
        ).strip()
        directions[agent_id] = direction
        write_skill(run_dir / "directions" / f"iter_{iteration:03d}" / f"agent_{agent_id}_direction.md", direction)

        previous_velocity_path = run_dir / "velocities" / f"iter_{iteration - 1:03d}" / f"agent_{agent_id}_velocity.md"
        previous_velocity = previous_velocity_path.read_text().strip() if previous_velocity_path.exists() else ""
        velocity_prompt = prompt_b_velocity(
            previous_velocity,
            direction,
            current_skills[agent_id],
            personal_best_skills[agent_id],
            global_best_skill,
            INITIAL_SKILLS[agent_id]["identity"],
            args.max_velocity_words,
        )
        save_json(
            run_dir / "prompts" / f"iter_{iteration:03d}" / f"agent_{agent_id}_velocity_prompt.json",
            {"prompt": velocity_prompt},
        )
        velocity = client.text(
            velocity_prompt,
            max_output_tokens=args.max_velocity_tokens,
            seed=args.seed + iteration * 10000 + agent_id * 100 + 2,
        ).strip()
        velocities[agent_id] = velocity
        write_skill(run_dir / "velocities" / f"iter_{iteration:03d}" / f"agent_{agent_id}_velocity.md", velocity)

        apply_prompt = prompt_c_apply_velocity(
            current_skills[agent_id],
            velocity,
            INITIAL_SKILLS[agent_id]["identity"],
            args.max_skill_words,
        )
        save_json(
            run_dir / "prompts" / f"iter_{iteration:03d}" / f"agent_{agent_id}_apply_velocity_prompt.json",
            {"prompt": apply_prompt},
        )
        updated = client.text(
            apply_prompt,
            max_output_tokens=args.max_skill_tokens,
            seed=args.seed + iteration * 10000 + agent_id * 100 + 3,
        ).strip()
        updated_skills[agent_id] = updated
        write_skill(run_dir / "skills" / "updated" / f"iter_{iteration:03d}" / f"agent_{agent_id}.md", updated)

    validation_items, validation_state = begin_validation_subset(run_dir, validation_pool, args.validation_batch_size, iteration)
    global_best_score_payload, global_best_rebase_event = rebase_global_best_for_validation_subset(
        client,
        run_dir,
        validation_items,
        validation_state,
        global_best_score_payload,
        global_best_skill,
        iteration,
        args.seed,
    )
    validation_scores = evaluate_skills(
        client,
        validation_items,
        updated_skills,
        iteration=iteration,
        seed=args.seed + 700000,
        output_path=run_dir / "results" / "validation" / f"iter_{iteration:03d}.jsonl",
        validation_subset_idx=int(validation_state["active_validation_subset_idx"]),
        validation_subset_range=list(validation_state["active_validation_subset_range"]),
    )
    next_validation_state = finish_validation_subset(run_dir, validation_state, validation_scores)

    personal_best_events = []
    for agent_id, summary in validation_scores.items():
        current_score = float(summary["score"])
        previous_best = float(personal_best_scores[str(agent_id)]["score"])
        if current_score > previous_best + args.epsilon:
            write_skill(run_dir / "skills" / "personal_best" / f"agent_{agent_id}.md", updated_skills[agent_id])
            personal_best_scores[str(agent_id)] = {
                "score": current_score,
                "iteration": iteration,
                "skill_path": display_path(run_dir / "skills" / "personal_best" / f"agent_{agent_id}.md"),
            }
            personal_best_events.append(
                {
                    "agent_id": agent_id,
                    "previous_score": previous_best,
                    "new_score": current_score,
                    "iteration": iteration,
                }
            )

    global_best_events = []
    global_best_score = float(global_best_score_payload["score"])
    best_agent_id = max(validation_scores, key=lambda aid: (validation_scores[aid]["score"], -aid))
    best_score = float(validation_scores[best_agent_id]["score"])
    if best_score > global_best_score + args.epsilon:
        write_skill(run_dir / "global_best.md", updated_skills[best_agent_id])
        global_best_score_payload = {
            "score": best_score,
            "agent_id": best_agent_id,
            "iteration": iteration,
            "skill_path": display_path(run_dir / "global_best.md"),
            "validation_subset_idx": int(validation_state["active_validation_subset_idx"]),
            "validation_subset_range": list(validation_state["active_validation_subset_range"]),
            "score_rebased_at_iteration": iteration,
        }
        global_best_events.append(
            {
                "agent_id": best_agent_id,
                "previous_score": global_best_score,
                "new_score": best_score,
                "iteration": iteration,
            }
        )

    for agent_id, skill_text in updated_skills.items():
        write_skill(run_dir / "skills" / "current" / f"agent_{agent_id}.md", skill_text)

    save_json(run_dir / "scores" / "personal_best_scores.json", personal_best_scores)
    save_json(run_dir / "scores" / "global_best_score.json", global_best_score_payload)
    append_jsonl(
        run_dir / "scores" / "iteration_scores.jsonl",
        [
            {
                "iteration": iteration,
                "phase": "post_update_validation",
                "scores": {str(agent_id): summary for agent_id, summary in validation_scores.items()},
                "personal_best_update_events": personal_best_events,
                "global_best_rebase_event": global_best_rebase_event,
                "global_best_update_events": global_best_events,
                "global_best_score": global_best_score_payload,
                "validation_subset": {
                    "used_subset_idx": int(validation_state["active_validation_subset_idx"]),
                    "used_subset_range": list(validation_state["active_validation_subset_range"]),
                    "perfect": bool(next_validation_state["last_validation_perfect"]),
                    "perfect_agents": list(next_validation_state["last_validation_perfect_agents"]),
                    "next_subset_idx": int(next_validation_state["validation_subset_idx"]),
                },
            }
        ],
    )
    save_json(
        run_dir / "metadata" / f"iter_{iteration:03d}.json",
        {
            "iteration": iteration,
            "train_problem_ids": [item["id"] for item in train_items],
            "validation_problem_ids": [item["id"] for item in validation_items],
            "validation_subset": {
                "used_subset_idx": int(validation_state["active_validation_subset_idx"]),
                "used_subset_range": list(validation_state["active_validation_subset_range"]),
                "perfect": bool(next_validation_state["last_validation_perfect"]),
                "perfect_agents": list(next_validation_state["last_validation_perfect_agents"]),
                "next_subset_idx": int(next_validation_state["validation_subset_idx"]),
            },
            "global_best_rebase_event": global_best_rebase_event,
            "direction_paths": {
                str(agent_id): display_path(run_dir / "directions" / f"iter_{iteration:03d}" / f"agent_{agent_id}_direction.md")
                for agent_id in directions
            },
            "velocity_paths": {
                str(agent_id): display_path(run_dir / "velocities" / f"iter_{iteration:03d}" / f"agent_{agent_id}_velocity.md")
                for agent_id in velocities
            },
            "updated_skill_paths": {
                str(agent_id): display_path(run_dir / "skills" / "updated" / f"iter_{iteration:03d}" / f"agent_{agent_id}.md")
                for agent_id in updated_skills
            },
        },
    )
    score_text = ", ".join(f"{aid}: {validation_scores[aid]['score']:.4f}" for aid in sorted(validation_scores))
    print(
        f"iteration={iteration:03d} "
        f"validation_scores={{{score_text}}} "
        f"global_best={global_best_score_payload['score']:.4f}",
        flush=True,
    )


def canonical_answer_for_vote(answer: str, item: dict[str, Any]) -> str:
    if item.get("answer_type") == "multiple_choice":
        return normalize_multiple_choice_answer(answer, item)
    return str(answer or "")


def build_final_test_aggregate(
    agent_rows: dict[int, list[dict[str, Any]]],
    mode: str = "final_test_personal_best_agents_independent",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    item_count = min((len(rows) for rows in agent_rows.values()), default=0)
    agent_ids = sorted(agent_rows)
    agent_count = len(agent_ids)
    aggregate_rows: list[dict[str, Any]] = []
    majority_correct_count = 0
    pass_correct_count = 0
    total_correct_count = 0

    for index in range(item_count):
        rows = [agent_rows[agent_id][index] for agent_id in agent_ids]
        reference = rows[0]
        answers = {str(row["agent_id"]): str(row.get("selected_answer", "")) for row in rows}
        corrects = {str(row["agent_id"]): bool(row.get("correct")) for row in rows}
        total_correct_count += sum(int(flag) for flag in corrects.values())
        pass_at_k_correct = any(corrects.values())
        pass_correct_count += int(pass_at_k_correct)

        nonempty_answers = [
            canonical_answer_for_vote(answers[str(agent_id)], reference)
            for agent_id in agent_ids
            if canonical_answer_for_vote(answers[str(agent_id)], reference)
        ]
        majority_answer = ""
        majority_vote_count = 0
        if nonempty_answers:
            counts = Counter(nonempty_answers)
            majority_vote_count = max(counts.values())
            tied = [answer for answer, count in counts.items() if count == majority_vote_count]
            majority_answer = min(tied, key=lambda answer: nonempty_answers.index(answer))
        majority_correct = grade_item_answer(majority_answer, reference)
        majority_correct_count += int(majority_correct)

        aggregate_rows.append(
            {
                "benchmark_id": reference["benchmark_id"],
                "unique_id": reference["unique_id"],
                "ground_truth": reference["ground_truth"],
                "answers_by_agent": answers,
                "correct_by_agent": corrects,
                "majority_answer": majority_answer,
                "majority_vote_count": majority_vote_count,
                "majority_correct": majority_correct,
                "pass_at_k_correct": pass_at_k_correct,
            }
        )

    summary = {
        "mode": mode,
        "question_count": item_count,
        "agent_count": agent_count,
        "k": agent_count,
        "majority_correct_count": majority_correct_count,
        "majority_accuracy": majority_correct_count / item_count if item_count else 0.0,
        "pass_at_k_correct_count": pass_correct_count,
        "pass_at_k": pass_correct_count / item_count if item_count else 0.0,
        "avg_at_k_correct_count": total_correct_count,
        "avg_at_k_total": item_count * agent_count,
        "avg_at_k": total_correct_count / (item_count * agent_count) if item_count and agent_count else 0.0,
    }
    return aggregate_rows, summary


def run_adaptive_agentpso_final_test(
    args: argparse.Namespace,
    run_dir: Path,
    client: LLMClient,
    test_items: list[dict[str, Any]],
    test_sampling_metadata: dict[str, Any],
    personal_best_scores: dict[str, Any],
    global_best_score_payload: dict[str, Any],
) -> dict[str, Any]:
    test_dir = run_dir / "results" / "test" / "adaptive_agentpso"
    test_dir.mkdir(parents=True, exist_ok=True)
    detail_path = test_dir / "adaptive_agentpso_problem_details.jsonl"
    if detail_path.exists():
        detail_path.unlink()

    global_best_agent_id = int(global_best_score_payload.get("agent_id", 0) or 0)
    global_best_path = run_dir / "global_best.md"
    global_best_skill = read_skill(global_best_path)
    skill_paths = {
        agent_id: run_dir / "skills" / "personal_best" / f"agent_{agent_id}.md"
        for agent_id in range(1, args.num_agents + 1)
    }
    personal_best_skills = {agent_id: read_skill(path) for agent_id, path in skill_paths.items()}

    agent_rows: dict[int, list[dict[str, Any]]] = {agent_id: [] for agent_id in range(1, args.num_agents + 1)}
    agent_correct_counts: dict[int, int] = {agent_id: 0 for agent_id in range(1, args.num_agents + 1)}
    agent_result_paths = {
        agent_id: test_dir / f"agent_{agent_id}_adaptive_agentpso.jsonl"
        for agent_id in range(1, args.num_agents + 1)
    }
    for path in agent_result_paths.values():
        if path.exists():
            path.unlink()

    for index, item in enumerate(test_items):
        initial_rows: dict[int, dict[str, Any]] = {}
        for agent_id in range(1, args.num_agents + 1):
            initial_row = solve_item(
                client,
                item,
                agent_id,
                personal_best_skills[agent_id],
                iteration=args.num_iterations,
                phase="final_test_adaptive_agentpso_initial",
                seed=args.seed + 910000 + agent_id * 100000 + index,
            )
            initial_row["skill_path"] = display_path(skill_paths[agent_id])
            initial_row["is_global_best_agent"] = agent_id == global_best_agent_id
            initial_rows[agent_id] = initial_row

        directions: dict[int, str] = {}
        velocities: dict[int, str] = {}
        adapted_rows: dict[int, dict[str, Any]] = {}
        for agent_id in range(1, args.num_agents + 1):
            peer_rows = [initial_rows[peer_id] for peer_id in range(1, args.num_agents + 1) if peer_id != agent_id]
            direction = compute_test_feedback_direction(
                client,
                item,
                agent_id,
                personal_best_skills[agent_id],
                initial_rows[agent_id],
                peer_rows,
                seed=args.seed + 920000 + agent_id * 100000 + index,
                max_output_tokens=args.max_velocity_tokens,
            )
            directions[agent_id] = direction
            is_global_best = agent_id == global_best_agent_id
            velocity = generate_test_time_velocity(
                client,
                agent_id,
                direction,
                personal_best_skills[agent_id],
                global_best_skill,
                is_global_best,
                seed=args.seed + 930000 + agent_id * 100000 + index,
                max_velocity_words=args.max_velocity_words,
                max_output_tokens=args.max_velocity_tokens,
            )
            velocities[agent_id] = velocity
            adapted_skill = skill_with_temporary_velocity(personal_best_skills[agent_id], velocity)
            adapted_row = solve_item(
                client,
                item,
                agent_id,
                adapted_skill,
                iteration=args.num_iterations,
                phase="final_test_adaptive_agentpso_adapted",
                seed=args.seed + 940000 + agent_id * 100000 + index,
            )
            adapted_row.update(
                {
                    "test_mode": "adaptive_agentpso",
                    "is_global_best_agent": is_global_best,
                    "skill_path": display_path(skill_paths[agent_id]),
                    "global_best_skill_path": display_path(global_best_path) if is_global_best else "",
                    "initial_reasoning": initial_rows[agent_id].get("reasoning", ""),
                    "initial_answer": initial_rows[agent_id].get("selected_answer", ""),
                    "initial_correct": initial_rows[agent_id].get("correct", False),
                    "problem_specific_peer_feedback": direction,
                    "test_time_velocity": velocity,
                    "adapted_reasoning": adapted_row.get("reasoning", ""),
                    "adapted_answer": adapted_row.get("selected_answer", ""),
                    "adapted_correct": adapted_row.get("correct", False),
                }
            )
            adapted_rows[agent_id] = adapted_row

        rows_for_vote = [adapted_rows[agent_id] for agent_id in range(1, args.num_agents + 1)]
        reference = rows_for_vote[0]
        adapted_answers = {str(row["agent_id"]): str(row.get("selected_answer", "")) for row in rows_for_vote}
        adapted_corrects = {str(row["agent_id"]): bool(row.get("correct")) for row in rows_for_vote}
        nonempty_answers = [
            canonical_answer_for_vote(adapted_answers[str(agent_id)], reference)
            for agent_id in range(1, args.num_agents + 1)
            if canonical_answer_for_vote(adapted_answers[str(agent_id)], reference)
        ]
        majority_answer = ""
        majority_vote_count = 0
        if nonempty_answers:
            counts = Counter(nonempty_answers)
            majority_vote_count = max(counts.values())
            tied = [answer for answer, count in counts.items() if count == majority_vote_count]
            majority_answer = min(tied, key=lambda answer: nonempty_answers.index(answer))
        majority_correct = grade_item_answer(majority_answer, reference)

        for agent_id in range(1, args.num_agents + 1):
            row = adapted_rows[agent_id]
            row["final_majority_answer"] = majority_answer
            row["final_majority_vote_count"] = majority_vote_count
            row["final_majority_correct"] = majority_correct
            agent_rows[agent_id].append(row)
            agent_correct_counts[agent_id] += int(row["correct"])
            append_jsonl(agent_result_paths[agent_id], [row])

        detail_row = {
            **item_result_metadata(item),
            "test_mode": "adaptive_agentpso",
            "global_best_agent_id": global_best_agent_id,
            "agents": {
                str(agent_id): {
                    "agent_id": agent_id,
                    "is_global_best_agent": agent_id == global_best_agent_id,
                    "skill_path": display_path(skill_paths[agent_id]),
                    "initial_reasoning": initial_rows[agent_id].get("reasoning", ""),
                    "initial_final_answer": initial_rows[agent_id].get("selected_answer", ""),
                    "problem_specific_peer_feedback": directions[agent_id],
                    "generated_test_time_velocity": velocities[agent_id],
                    "adapted_reasoning": adapted_rows[agent_id].get("reasoning", ""),
                    "adapted_final_answer": adapted_rows[agent_id].get("selected_answer", ""),
                    "adapted_correct": adapted_rows[agent_id].get("correct", False),
                }
                for agent_id in range(1, args.num_agents + 1)
            },
            "final_majority_answer": majority_answer,
            "final_majority_vote_count": majority_vote_count,
            "final_majority_correct": majority_correct,
        }
        append_jsonl(detail_path, [detail_row])

        completed = index + 1
        if completed % 10 == 0 or completed == len(test_items):
            score_text = ", ".join(
                f"{agent_id}:{agent_correct_counts[agent_id]}/{completed}"
                for agent_id in range(1, args.num_agents + 1)
            )
            print(
                f"adaptive_agentpso_checkpoint completed={completed}/{len(test_items)} "
                f"agent_corrects={{{score_text}}}",
                flush=True,
            )

    agent_summaries: dict[str, dict[str, Any]] = {}
    for agent_id in range(1, args.num_agents + 1):
        question_count = len(agent_rows[agent_id])
        summary = {
            "mode": "final_test_adaptive_agentpso_agent",
            "agent_id": agent_id,
            "is_global_best_agent": agent_id == global_best_agent_id,
            "score": agent_correct_counts[agent_id] / question_count if question_count else 0.0,
            "accuracy": agent_correct_counts[agent_id] / question_count if question_count else 0.0,
            "correct_count": agent_correct_counts[agent_id],
            "question_count": question_count,
            "skill_path": display_path(skill_paths[agent_id]),
            "personal_best_validation": personal_best_scores.get(str(agent_id), {}),
            "result_path": display_path(agent_result_paths[agent_id]),
        }
        agent_summaries[str(agent_id)] = summary
        save_json(test_dir / f"agent_{agent_id}_summary.json", summary)

    aggregate_rows, aggregate_summary = build_final_test_aggregate(
        agent_rows,
        mode="final_test_adaptive_agentpso_agents_second_pass",
    )
    aggregate_path = test_dir / "aggregate_majority_pass_avg.jsonl"
    if aggregate_path.exists():
        aggregate_path.unlink()
    append_jsonl(aggregate_path, aggregate_rows)

    final_summary = {
        "mode": "final_test_adaptive_agentpso",
        "test_mode": "adaptive_agentpso",
        "test_dataset": display_path(args.test_dataset),
        "test_limit": args.test_limit,
        "test_sampling": test_sampling_metadata,
        "global_best_agent_id": global_best_agent_id,
        "global_best_validation": global_best_score_payload,
        "global_best_skill_path": display_path(global_best_path),
        "agent_summaries": agent_summaries,
        "aggregate": aggregate_summary,
        "aggregate_result_path": display_path(aggregate_path),
        "detail_result_path": display_path(detail_path),
    }
    save_json(test_dir / "aggregate_summary.json", final_summary)
    save_json(run_dir / "scores" / "final_test_summary_adaptive_agentpso.json", final_summary)
    append_jsonl(
        run_dir / "scores" / "iteration_scores.jsonl",
        [{"iteration": args.num_iterations, "phase": "final_test_adaptive_agentpso", "summary": final_summary}],
    )
    print(
        f"adaptive_agentpso_aggregate majority={aggregate_summary['majority_accuracy']:.4f} "
        f"pass@{aggregate_summary['k']}={aggregate_summary['pass_at_k']:.4f} "
        f"avg@{aggregate_summary['k']}={aggregate_summary['avg_at_k']:.4f}",
        flush=True,
    )
    return final_summary


def run_final_test(args: argparse.Namespace, run_dir: Path, client: LLMClient) -> dict[str, Any]:
    test_items = read_benchmark_dataset(Path(args.test_dataset).resolve())
    test_sampling_metadata: dict[str, Any] = {}
    if args.dataset_preset == "deepmath":
        sampled_test_path = run_dir / "data" / "deepmath_test_pool.jsonl"
        if sampled_test_path.exists():
            test_items = read_items_jsonl(sampled_test_path)
            test_sampling_metadata = {
                "enabled": True,
                "source": "saved_remaining_subject_matched_test_pool",
                "test_pool_path": display_path(sampled_test_path),
                "test_sample_size": len(test_items),
                "test_subject_counts": subject_counts(test_items),
            }
        elif args.test_limit is not None:
            test_items = test_items[: args.test_limit]
    elif args.dataset_preset == "bbh":
        sampled_test_path = run_dir / "data" / "bbh_test_pool.jsonl"
        if sampled_test_path.exists():
            test_items = read_items_jsonl(sampled_test_path)
            test_sampling_metadata = {
                "enabled": True,
                "source": "saved_disjoint_test_pool",
                "test_pool_path": display_path(sampled_test_path),
                "test_sample_size": len(test_items),
                "test_subject_counts": subject_counts(test_items),
            }
        else:
            sample_size = args.test_limit if args.test_limit is not None else 200
            candidate_count = len(test_items)
            test_items = sample_items(test_items, sample_size, args.seed + 201, "BigBenchHard final test")
            test_sampling_metadata = {
                "enabled": True,
                "source": "direct_random_sample_without_saved_train_validation_exclusion",
                "test_candidate_count": candidate_count,
                "test_sample_size": sample_size,
                "test_subject_counts": subject_counts(test_items),
                "test_sample_seed": args.seed + 201,
            }
    elif args.test_limit is not None:
        test_items = test_items[: args.test_limit]

    personal_best_scores, global_best_score_payload = load_scores(run_dir)
    if args.test_mode == "adaptive_agentpso":
        return run_adaptive_agentpso_final_test(
            args,
            run_dir,
            client,
            test_items,
            test_sampling_metadata,
            personal_best_scores,
            global_best_score_payload,
        )

    test_dir = run_dir / "results" / "test"
    agent_rows: dict[int, list[dict[str, Any]]] = {}
    agent_summaries: dict[str, dict[str, Any]] = {}

    for agent_id in range(1, args.num_agents + 1):
        skill_path = run_dir / "skills" / "personal_best" / f"agent_{agent_id}.md"
        skill_text = read_skill(skill_path)
        result_path = test_dir / f"agent_{agent_id}_personal_best.jsonl"
        if result_path.exists():
            result_path.unlink()
        rows: list[dict[str, Any]] = []
        correct_count = 0
        for index, item in enumerate(test_items):
            row = solve_item(
                client,
                item,
                agent_id,
                skill_text,
                iteration=args.num_iterations,
                phase="final_test_personal_best",
                seed=args.seed + 900000 + agent_id * 100000 + index,
            )
            row["skill_path"] = display_path(skill_path)
            rows.append(row)
            correct_count += int(row["correct"])
            append_jsonl(result_path, [row])
            completed = index + 1
            if completed % 50 == 0 or completed == len(test_items):
                print(
                    f"final_test_checkpoint agent={agent_id} completed={completed}/{len(test_items)} "
                    f"correct={correct_count} accuracy={correct_count / completed if completed else 0.0:.4f}",
                    flush=True,
                )
        agent_rows[agent_id] = rows
        question_count = len(rows)
        summary = {
            "mode": "final_test_personal_best_agent",
            "agent_id": agent_id,
            "score": correct_count / question_count if question_count else 0.0,
            "accuracy": correct_count / question_count if question_count else 0.0,
            "correct_count": correct_count,
            "question_count": question_count,
            "skill_path": display_path(skill_path),
            "personal_best_validation": personal_best_scores.get(str(agent_id), {}),
            "result_path": display_path(result_path),
        }
        agent_summaries[str(agent_id)] = summary
        save_json(test_dir / f"agent_{agent_id}_summary.json", summary)
        print(
            f"final_test_agent={agent_id} correct={correct_count}/{question_count} "
            f"accuracy={summary['accuracy']:.4f}",
            flush=True,
        )

    aggregate_rows, aggregate_summary = build_final_test_aggregate(agent_rows)
    aggregate_path = test_dir / "aggregate_majority_pass_avg.jsonl"
    if aggregate_path.exists():
        aggregate_path.unlink()
    append_jsonl(aggregate_path, aggregate_rows)

    final_summary = {
        "mode": "final_test_all_personal_best_agents_independent",
        "test_mode": "personal_best",
        "test_dataset": display_path(args.test_dataset),
        "test_limit": args.test_limit,
        "test_sampling": test_sampling_metadata,
        "agent_summaries": agent_summaries,
        "aggregate": aggregate_summary,
        "global_best_validation": global_best_score_payload,
        "aggregate_result_path": display_path(aggregate_path),
    }
    save_json(test_dir / "aggregate_summary.json", final_summary)
    save_json(run_dir / "scores" / "final_test_summary.json", final_summary)
    append_jsonl(
        run_dir / "scores" / "iteration_scores.jsonl",
        [{"iteration": args.num_iterations, "phase": "final_test", "summary": final_summary}],
    )
    print(
        f"final_test_aggregate majority={aggregate_summary['majority_accuracy']:.4f} "
        f"pass@{aggregate_summary['k']}={aggregate_summary['pass_at_k']:.4f} "
        f"avg@{aggregate_summary['k']}={aggregate_summary['avg_at_k']:.4f}",
        flush=True,
    )
    return final_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PSO-inspired natural-language skill evolution for independent LLM agents.")
    parser.add_argument("--run-name", default="", help="Run directory name under --output-root.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--backend", choices=["openai", "local-openai", "anthropic", "mock"], default="openai")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--train-dataset", default=str(DEFAULT_TRAIN_DATASET))
    parser.add_argument(
        "--validation-dataset",
        default="",
        help="Optional separate validation dataset. If omitted, validation examples are sliced from --train-dataset after the train pool.",
    )
    parser.add_argument("--test-dataset", default=str(DEFAULT_TEST_DATASET))
    parser.add_argument(
        "--dataset-preset",
        choices=[
            "math",
            "deepmath",
            "aime24",
            "aime25",
            "minerva",
            "bbh",
        ],
        default="math",
        help="Convenience preset for default train/test paths and config metadata.",
    )
    parser.add_argument("--test-limit", type=int, default=None, help="Optional number of test rows to evaluate.")
    parser.add_argument(
        "--test_mode",
        "--test-mode",
        choices=["personal_best", "adaptive_agentpso"],
        default="personal_best",
        help="Final test inference mode. personal_best keeps the original independent personal-best majority vote; adaptive_agentpso adds one temporary test-time AgentPSO-style refinement step.",
    )
    parser.add_argument("--skip-final-test", action="store_true", help="Skip final test evaluation after training.")
    parser.add_argument("--num-agents", type=int, default=4)
    parser.add_argument("--num-iterations", type=int, default=10)
    parser.add_argument("--train-pool-size", type=int, default=100)
    parser.add_argument("--validation-pool-size", type=int, default=100)
    parser.add_argument("--train-batch-size", type=int, default=10)
    parser.add_argument("--validation-batch-size", type=int, default=20)
    parser.add_argument("--fitness-accuracy-weight", type=float, default=0.8)
    parser.add_argument("--fitness-reasoning-weight", type=float, default=0.2)
    parser.add_argument("--epsilon", type=float, default=0.01)
    add_boolean_optional_argument(parser, "--preserve-agent-identity", default=True)
    parser.add_argument("--max-skill-words", type=int, default=1200)
    parser.add_argument("--max-velocity-words", type=int, default=200)
    parser.add_argument("--max-skill-tokens", type=int, default=800)
    parser.add_argument("--max-velocity-tokens", type=int, default=300)
    parser.add_argument("--use-reasoning-quality-score", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.dataset_preset == "deepmath":
        if args.train_dataset == str(DEFAULT_TRAIN_DATASET):
            args.train_dataset = str(DEFAULT_DEEPMATH_DATASET)
        if args.test_dataset == str(DEFAULT_TEST_DATASET):
            args.test_dataset = str(DEFAULT_DEEPMATH_DATASET)
    elif args.dataset_preset == "aime24":
        if args.test_dataset == str(DEFAULT_TEST_DATASET):
            args.test_dataset = str(DEFAULT_AIME24_TEST_DATASET)
    elif args.dataset_preset == "aime25":
        if args.test_dataset == str(DEFAULT_TEST_DATASET):
            args.test_dataset = str(DEFAULT_AIME25_TEST_DATASET)
    elif args.dataset_preset == "minerva":
        if args.test_dataset == str(DEFAULT_TEST_DATASET):
            args.test_dataset = str(DEFAULT_MINERVA_TEST_DATASET)
    elif args.dataset_preset == "bbh":
        if args.train_dataset == str(DEFAULT_TRAIN_DATASET):
            args.train_dataset = str(DEFAULT_BBH_DATASET)
        if args.test_dataset == str(DEFAULT_TEST_DATASET):
            args.test_dataset = str(DEFAULT_BBH_DATASET)
        if args.test_limit is None:
            args.test_limit = 200
    if args.num_agents != 4:
        raise ValueError("This implementation currently expects exactly 4 agents.")
    return args


def main() -> None:
    args = parse_args()
    run_name = args.run_name or "agent_pso_run"
    run_dir = Path(args.output_root).expanduser().resolve() / run_name
    if run_dir.exists():
        if not args.overwrite:
            raise RuntimeError(f"Run directory already exists: {run_dir}. Use --overwrite to replace it.")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    client = LLMClient(backend=args.backend, model=args.model, api_base=args.api_base)
    state = initialize_run(args, run_dir, client)
    for iteration in range(1, args.num_iterations + 1):
        run_iteration(args, run_dir, client, state["train_pool"], state["validation_pool"], iteration)
    if not args.skip_final_test:
        run_final_test(args, run_dir, client)
    print(f"run_dir={display_path(run_dir)}", flush=True)


if __name__ == "__main__":
    main()
