"""Generation and executable-SQL evaluation shared by all baseline notebooks."""

from __future__ import annotations

import json
import random
import re
import sqlite3
import time
from collections import Counter
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

import sqlglot
from sqlglot import exp

from .data import build_messages

_FENCE_PATTERN = re.compile(r"```(?:sql|sqlite)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_QUERY_START_PATTERN = re.compile(r"\b(?:SELECT|WITH)\b", re.IGNORECASE)
_STOP_MARKERS = ("<|im_end|>", "<|eot_id|>", "<|endoftext|>")
_FORBIDDEN_EXPRESSION_NAMES = {
    "Alter",
    "Attach",
    "Command",
    "Create",
    "Delete",
    "Drop",
    "Insert",
    "Merge",
    "Pragma",
    "Set",
    "Transaction",
    "Update",
}


def _response_headers(exc: Exception) -> Any:
    response = getattr(exc, "response", None)
    return getattr(response, "headers", None) or getattr(exc, "headers", None) or {}


def _header(headers: Any, name: str) -> str | None:
    try:
        value = headers.get(name)
        if value is None:
            value = headers.get(name.title())
    except AttributeError:
        return None
    return str(value).strip() if value is not None else None


def _parse_delay_seconds(value: str, *, may_be_epoch: bool = False) -> float | None:
    text = value.strip().lower()
    unit_match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(ms|s|m|h)", text)
    if unit_match:
        amount = float(unit_match.group(1))
        multiplier = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}[unit_match.group(2)]
        return amount * multiplier
    try:
        amount = float(text)
        if may_be_epoch and amount > 1_000_000_000:
            return max(0.0, amount - time.time())
        return max(0.0, amount)
    except ValueError:
        pass
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return max(0.0, parsed.timestamp() - time.time())


def _retry_after_seconds(exc: Exception) -> float | None:
    headers = _response_headers(exc)
    retry_after = _header(headers, "retry-after")
    if retry_after:
        return _parse_delay_seconds(retry_after)
    for name in ("x-ratelimit-reset-requests", "x-ratelimit-reset"):
        reset = _header(headers, name)
        if reset:
            return _parse_delay_seconds(reset, may_be_epoch=True)
    return None


def extract_sql(text: str) -> str | None:
    """Extract one read-only SQLite query from a model response."""
    candidate = text.strip()
    if "</think>" in candidate:
        candidate = candidate.rsplit("</think>", maxsplit=1)[-1].strip()
    fenced = _FENCE_PATTERN.search(candidate)
    if fenced:
        candidate = fenced.group(1).strip()
    for marker in _STOP_MARKERS:
        candidate = candidate.split(marker, maxsplit=1)[0].strip()
    match = _QUERY_START_PATTERN.search(candidate)
    if not match:
        return None
    candidate = candidate[match.start() :].strip()
    try:
        parsed = sqlglot.parse(candidate, read="sqlite")
    except sqlglot.errors.ParseError:
        return None
    if len(parsed) != 1 or parsed[0] is None:
        return None
    expression = parsed[0]
    if not expression.find(exp.Select):
        return None
    if any(type(node).__name__ in _FORBIDDEN_EXPRESSION_NAMES for node in expression.walk()):
        return None
    return expression.sql(dialect="sqlite", pretty=False)


def normalize_sql(text: str) -> str | None:
    query = extract_sql(text)
    if query is None:
        return None
    return sqlglot.parse_one(query, read="sqlite").sql(dialect="sqlite", pretty=False, normalize=True)


def _canonical_value(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 8)
    if isinstance(value, bytes):
        return value.hex()
    return value


def execute_read_only_sql(
    database_path: str | Path,
    sql: str,
    *,
    timeout_seconds: float = 30.0,
    max_rows: int = 100_000,
) -> set[tuple[Any, ...]]:
    """Execute a single parsed query using BIRD-compatible set comparison semantics."""
    query = extract_sql(sql)
    if query is None:
        raise ValueError("Only one parseable SELECT/WITH query is allowed.")
    path = Path(database_path).resolve()
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    started = time.monotonic()

    def progress_handler() -> int:
        return int(time.monotonic() - started > timeout_seconds)

    connection.set_progress_handler(progress_handler, 1_000)
    try:
        cursor = connection.execute(query)
        values = cursor.fetchmany(max_rows + 1)
        if len(values) > max_rows:
            raise RuntimeError(f"Query exceeded the {max_rows}-row evaluation limit.")
        return {tuple(_canonical_value(value) for value in row) for row in values}
    finally:
        connection.close()


def score_predictions(
    rows: Iterable[dict[str, Any]],
    *,
    data_dir: str | Path,
    timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    """Score syntax, normalized exact match, and official-style execution accuracy."""
    root = Path(data_dir).resolve()
    scored: list[dict[str, Any]] = []
    for row in rows:
        generated = str(row.get("generated", ""))
        predicted_sql = extract_sql(generated)
        expected_sql = str(row["expected_sql"])
        expected_normalized = normalize_sql(expected_sql)
        predicted_normalized = normalize_sql(generated)
        exact_match = predicted_normalized is not None and predicted_normalized == expected_normalized
        executable = False
        execution_correct = False
        execution_error: str | None = None
        if predicted_sql is not None:
            try:
                database_path = (root / str(row["database_path"])).resolve()
                if database_path != root and root not in database_path.parents:
                    raise ValueError("Evaluation database path escapes the prepared data directory.")
                expected_result = execute_read_only_sql(
                    database_path, expected_sql, timeout_seconds=timeout_seconds
                )
                predicted_result = execute_read_only_sql(
                    database_path, predicted_sql, timeout_seconds=timeout_seconds
                )
                executable = True
                execution_correct = predicted_result == expected_result
            except Exception as exc:
                execution_error = f"{type(exc).__name__}: {exc}"[:500]
        scored.append(
            {
                **row,
                "predicted_sql": predicted_sql,
                "sql_valid": predicted_sql is not None,
                "sql_executable": executable,
                "normalized_exact_match": exact_match,
                "execution_correct": execution_correct,
                "execution_error": execution_error,
            }
        )

    total = len(scored)
    difficulty_counts = Counter(str(row["difficulty"]) for row in scored)
    difficulty_correct = Counter(str(row["difficulty"]) for row in scored if row["execution_correct"])
    per_difficulty = {
        name: difficulty_correct[name] / count for name, count in sorted(difficulty_counts.items())
    }
    return {
        "n": total,
        "execution_accuracy": (
            sum(bool(row["execution_correct"]) for row in scored) / total if total else 0.0
        ),
        "sql_valid_rate": sum(bool(row["sql_valid"]) for row in scored) / total if total else 0.0,
        "sql_executable_rate": (sum(bool(row["sql_executable"]) for row in scored) / total if total else 0.0),
        "normalized_exact_match": (
            sum(bool(row["normalized_exact_match"]) for row in scored) / total if total else 0.0
        ),
        "per_difficulty_execution_accuracy": per_difficulty,
        "rows": scored,
    }


def generate_nvidia_api_predictions(
    client: Any,
    rows: list[dict[str, Any]],
    *,
    model: str,
    message_builder: Callable[[dict[str, Any]], list[dict[str, str]]] = build_messages,
    resume_path: str | Path | None = None,
    max_tokens: int = 512,
    max_attempts: int = 8,
    requests_per_minute: float | None = 30.0,
    rate_limit_retry_seconds: float = 65.0,
) -> list[dict[str, Any]]:
    """Generate resumable Text2SQL predictions through an OpenAI-compatible endpoint."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive.")
    if requests_per_minute is not None and requests_per_minute <= 0:
        raise ValueError("requests_per_minute must be positive or None.")
    if rate_limit_retry_seconds < 0:
        raise ValueError("rate_limit_retry_seconds cannot be negative.")
    target = Path(resume_path) if resume_path is not None else None
    cached: dict[str, dict[str, Any]] = {}
    if target is not None and target.exists():
        for line in target.read_text(encoding="utf-8").splitlines():
            if line.strip():
                cached_row = json.loads(line)
                cached[str(cached_row["example_id"])] = cached_row

    requested_ids = [str(row["example_id"]) for row in rows]
    if len(requested_ids) != len(set(requested_ids)):
        raise ValueError("API evaluation rows must have unique example IDs.")
    unexpected_ids = set(cached).difference(requested_ids)
    if unexpected_ids:
        raise ValueError(f"Resume file contains IDs outside this evaluation: {sorted(unexpected_ids)[:3]}")
    completed = sum(example_id in cached for example_id in requested_ids)
    if completed:
        print(f"API evaluation: resuming with {completed}/{len(rows)} cached responses", flush=True)

    minimum_interval = 60.0 / requests_per_minute if requests_per_minute else 0.0
    last_request_started_at: float | None = None

    def persist() -> None:
        if target is None:
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            for example_id in requested_ids:
                if example_id in cached:
                    handle.write(json.dumps(cached[example_id], ensure_ascii=False) + "\n")

    for row in rows:
        example_id = str(row["example_id"])
        if example_id in cached:
            continue
        last_error: Exception | None = None
        for attempt in range(max_attempts):
            try:
                now = time.monotonic()
                if last_request_started_at is not None:
                    delay = minimum_interval - (now - last_request_started_at)
                    if delay > 0:
                        time.sleep(delay)
                last_request_started_at = time.monotonic()
                response = client.chat.completions.create(
                    model=model,
                    messages=message_builder(row),
                    temperature=0.0,
                    max_tokens=max_tokens,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                content = response.choices[0].message.content or ""
                cached[example_id] = {**row, "generated": content.strip()}
                persist()
                completed += 1
                break
            except Exception as exc:  # Provider SDK exception classes vary by release.
                last_error = exc
                status_code = getattr(exc, "status_code", None)
                is_rate_limit = status_code == 429 or type(exc).__name__ == "RateLimitError"
                is_retryable = (
                    is_rate_limit
                    or status_code is None
                    or status_code in {408, 409, 425}
                    or (isinstance(status_code, int) and status_code >= 500)
                )
                if not is_retryable:
                    raise RuntimeError(
                        f"API returned non-retryable HTTP {status_code} for {example_id}."
                    ) from exc
                if attempt + 1 < max_attempts:
                    server_delay = _retry_after_seconds(exc) if is_rate_limit else None
                    delay = (
                        server_delay + 1.0
                        if server_delay is not None
                        else rate_limit_retry_seconds
                        if is_rate_limit
                        else min(2**attempt, 30)
                    )
                    print(
                        f"API retry for {example_id} in {delay:.1f}s ({attempt + 2}/{max_attempts}).",
                        flush=True,
                    )
                    time.sleep(delay)
        else:
            raise RuntimeError(f"API failed for {example_id} after {max_attempts} attempts.") from last_error
        if completed == 1 or completed % 10 == 0 or completed == len(rows):
            print(f"API evaluation: {completed}/{len(rows)} requests complete", flush=True)
    return [cached[example_id] for example_id in requested_ids]


def save_report(path: str | Path, report: dict[str, Any], *, model: str, run_type: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "run_type": run_type,
        "saved_at_utc": datetime.now(UTC).isoformat(),
        **report,
    }
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def paired_execution_comparison(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    bootstrap_samples: int = 5_000,
    seed: int = 1234,
) -> dict[str, Any]:
    """Bootstrap tuned-minus-baseline execution accuracy on identical IDs."""
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive.")
    baseline_by_id = {str(row["example_id"]): bool(row["execution_correct"]) for row in baseline["rows"]}
    candidate_by_id = {str(row["example_id"]): bool(row["execution_correct"]) for row in candidate["rows"]}
    if baseline_by_id.keys() != candidate_by_id.keys():
        raise ValueError("Baseline and candidate must contain identical evaluation example IDs.")
    ids = sorted(baseline_by_id)
    if not ids:
        raise ValueError("Cannot compare empty reports.")
    deltas = [int(candidate_by_id[item]) - int(baseline_by_id[item]) for item in ids]
    observed = sum(deltas) / len(deltas)
    rng = random.Random(seed)
    bootstrapped = sorted(
        sum(deltas[rng.randrange(len(deltas))] for _ in deltas) / len(deltas)
        for _ in range(bootstrap_samples)
    )
    low_index = int(0.025 * (bootstrap_samples - 1))
    high_index = int(0.975 * (bootstrap_samples - 1))
    return {
        "n": len(ids),
        "absolute_execution_accuracy_gain": observed,
        "paired_bootstrap_95ci": [bootstrapped[low_index], bootstrapped[high_index]],
        "improved_examples": sum(delta == 1 for delta in deltas),
        "regressed_examples": sum(delta == -1 for delta in deltas),
        "unchanged_examples": sum(delta == 0 for delta in deltas),
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
    }
