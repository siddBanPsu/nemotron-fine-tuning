"""Exact-match generation evaluation shared by all three notebooks."""

from __future__ import annotations

import json
import random
import re
import time
from collections import Counter
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Callable, Iterable

from .constants import ROUTE_PREFIX
from .data import build_messages

ROUTE_PATTERN = re.compile(rf"\b{re.escape(ROUTE_PREFIX)}(?:[0-6]\d|7[0-6])\b", re.IGNORECASE)


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


def extract_route_code(text: str) -> str | None:
    match = ROUTE_PATTERN.search(text.strip())
    return match.group(0).upper() if match else None


def score_predictions(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    scored: list[dict[str, Any]] = []
    for row in rows:
        expected = str(row["expected"]).upper()
        predicted = extract_route_code(str(row.get("generated", "")))
        scored.append({**row, "predicted": predicted, "correct": predicted == expected})

    total = len(scored)
    correct = sum(bool(row["correct"]) for row in scored)
    valid = sum(row["predicted"] is not None for row in scored)
    per_label_total = Counter(str(row["expected"]) for row in scored)
    per_label_correct = Counter(str(row["expected"]) for row in scored if row["correct"])
    per_label_accuracy = {
        label: per_label_correct[label] / count for label, count in sorted(per_label_total.items())
    }
    macro_accuracy = (
        sum(per_label_accuracy.values()) / len(per_label_accuracy) if per_label_accuracy else 0.0
    )
    return {
        "n": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "macro_accuracy": macro_accuracy,
        "valid_code_rate": valid / total if total else 0.0,
        "per_label_accuracy": per_label_accuracy,
        "rows": scored,
    }


def generate_predictions(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    *,
    batch_size: int = 8,
    max_new_tokens: int = 8,
) -> list[dict[str, Any]]:
    """Greedily generate short route codes in bounded batches."""
    import torch

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.eval()

    output_rows: list[dict[str, Any]] = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        encoded = tokenizer(
            [str(row["prompt"]) for row in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        encoded = {name: tensor.to(model.device) for name, tensor in encoded.items()}
        prompt_width = encoded["input_ids"].shape[1]
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        texts = tokenizer.batch_decode(generated[:, prompt_width:], skip_special_tokens=True)
        output_rows.extend({**row, "generated": text.strip()} for row, text in zip(batch, texts))
    return output_rows


def generate_nvidia_api_predictions(
    client: Any,
    rows: list[dict[str, Any]],
    *,
    model: str,
    message_builder: Callable[[str], list[dict[str, str]]] = build_messages,
    resume_path: str | Path | None = None,
    max_tokens: int = 8,
    max_attempts: int = 8,
    requests_per_minute: float | None = 30.0,
    rate_limit_retry_seconds: float = 65.0,
) -> list[dict[str, Any]]:
    """Generate route codes through NVIDIA's OpenAI-compatible API.

    Results are checkpointed after every successful response so a trial-endpoint
    throttle or notebook interruption does not discard completed requests. Calls
    are paced below the public quota, and 429 responses honor provider retry
    headers or fall back to waiting for the rolling minute window to reset.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive.")
    if requests_per_minute is not None and requests_per_minute <= 0:
        raise ValueError("requests_per_minute must be positive or None.")
    if rate_limit_retry_seconds < 0:
        raise ValueError("rate_limit_retry_seconds cannot be negative.")

    target = Path(resume_path) if resume_path is not None else None
    cached: dict[str, dict[str, Any]] = {}
    if target is not None and target.exists():
        with target.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    cached[str(row["example_id"])] = row

    requested_ids = [str(row["example_id"]) for row in rows]
    if len(requested_ids) != len(set(requested_ids)):
        raise ValueError("API evaluation rows must have unique example IDs.")
    unexpected_ids = set(cached).difference(requested_ids)
    if unexpected_ids:
        raise ValueError(f"Resume file contains IDs outside this evaluation: {sorted(unexpected_ids)[:3]}")

    completed = sum(example_id in cached for example_id in requested_ids)
    if completed:
        print(
            f"NVIDIA API baseline: resuming with {completed}/{len(rows)} cached responses",
            flush=True,
        )

    minimum_interval = 60.0 / requests_per_minute if requests_per_minute is not None else 0.0
    last_request_started_at: float | None = None

    def wait_for_request_slot() -> None:
        nonlocal last_request_started_at
        now = time.monotonic()
        if last_request_started_at is not None:
            remaining = minimum_interval - (now - last_request_started_at)
            if remaining > 0:
                time.sleep(remaining)
        last_request_started_at = time.monotonic()

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
                wait_for_request_slot()
                response = client.chat.completions.create(
                    model=model,
                    messages=message_builder(str(row["utterance"])),
                    temperature=0.0,
                    max_tokens=max_tokens,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                content = response.choices[0].message.content or ""
                cached[example_id] = {**row, "generated": content.strip()}
                persist()
                completed += 1
                break
            except Exception as exc:  # Provider SDK exception types vary by release.
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
                        f"NVIDIA API returned non-retryable HTTP {status_code} for {example_id}."
                    ) from exc
                if attempt + 1 < max_attempts:
                    if is_rate_limit:
                        server_delay = _retry_after_seconds(exc)
                        delay = (
                            server_delay + 1.0
                            if server_delay is not None
                            else rate_limit_retry_seconds
                        )
                        reason = "rate limited"
                    else:
                        delay = min(2**attempt, 30)
                        reason = f"transient error {status_code or type(exc).__name__}"
                    print(
                        f"NVIDIA API {reason} on {example_id}; waiting {delay:.1f}s "
                        f"before retry {attempt + 2}/{max_attempts}.",
                        flush=True,
                    )
                    time.sleep(delay)
        else:
            raise RuntimeError(
                f"NVIDIA API failed for {example_id} after {max_attempts} attempts."
            ) from last_error
        if completed == 1 or completed % 25 == 0 or completed == len(rows):
            print(f"NVIDIA API baseline: {completed}/{len(rows)} requests complete", flush=True)

    return [cached[example_id] for example_id in requested_ids]


def save_report(path: str | Path, report: dict[str, Any], *, model: str, run_type: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "run_type": run_type,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        **report,
    }
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_summary(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {key: payload[key] for key in ("model", "run_type", "n", "accuracy", "macro_accuracy", "valid_code_rate")}


def paired_accuracy_comparison(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    bootstrap_samples: int = 5_000,
    seed: int = 1234,
) -> dict[str, Any]:
    """Compare correctness on identical IDs and bootstrap the paired accuracy delta."""
    baseline_by_id = {str(row["example_id"]): bool(row["correct"]) for row in baseline["rows"]}
    candidate_by_id = {str(row["example_id"]): bool(row["correct"]) for row in candidate["rows"]}
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
        "absolute_accuracy_gain": observed,
        "paired_bootstrap_95ci": [bootstrapped[low_index], bootstrapped[high_index]],
        "improved_examples": sum(delta == 1 for delta in deltas),
        "regressed_examples": sum(delta == -1 for delta in deltas),
        "unchanged_examples": sum(delta == 0 for delta in deltas),
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
    }
