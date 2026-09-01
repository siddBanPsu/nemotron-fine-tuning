"""Deterministic BANKING77 preparation for prompt-completion SFT.

The target is an intentionally opaque enterprise route code. That makes the
exercise measure whether tuning learned a private taxonomy, rather than whether
the base model already knows natural-language intent names.
"""

from __future__ import annotations

import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from .constants import ROUTE_PERMUTATION_SEED, ROUTE_PREFIX


_RETRIEVAL_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


_ROUTE_INDICES = list(range(77))
random.Random(ROUTE_PERMUTATION_SEED).shuffle(_ROUTE_INDICES)
ROUTE_INDEX_BY_LABEL = tuple(_ROUTE_INDICES)


@dataclass(frozen=True)
class PreparedExample:
    example_id: str
    utterance: str
    label_id: int
    label_name: str
    route_code: str


def route_code(label_id: int) -> str:
    """Return the stable two-digit internal code for a BANKING77 label."""
    if not 0 <= label_id <= 76:
        raise ValueError(f"BANKING77 label must be in [0, 76], received {label_id}.")
    return f"{ROUTE_PREFIX}{ROUTE_INDEX_BY_LABEL[label_id]:02d}"


def system_instruction() -> str:
    return (
        "You route online-banking support messages into an internal intent code. "
        "Return exactly one code from B77_00 through B77_76. Return only the code: "
        "no words, punctuation, JSON, or explanation."
    )


def build_messages(utterance: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_instruction()},
        {"role": "user", "content": utterance.strip()},
    ]


def build_taxonomy_messages(
    utterance: str,
    label_map: dict[str, dict[str, Any]],
) -> list[dict[str, str]]:
    """Build an inference prompt that exposes the private taxonomy explicitly."""
    taxonomy = "\n".join(
        f"{code} = {str(details['label_name']).replace('_', ' ')}"
        for code, details in sorted(label_map.items())
    )
    return [
        {
            "role": "system",
            "content": f"{system_instruction()}\n\nInternal taxonomy:\n{taxonomy}",
        },
        {"role": "user", "content": utterance.strip()},
    ]


def build_few_shot_messages(
    utterance: str,
    demonstrations: Sequence[dict[str, Any]],
) -> list[dict[str, str]]:
    """Build a prompt using only retrieved training examples as demonstrations."""
    messages = [{"role": "system", "content": system_instruction()}]
    for row in demonstrations:
        messages.extend(
            [
                {"role": "user", "content": str(row["utterance"]).strip()},
                {"role": "assistant", "content": str(row["route_code"]).strip()},
            ]
        )
    messages.append({"role": "user", "content": utterance.strip()})
    return messages


class LexicalDemonstrationRetriever:
    """Deterministic TF-IDF cosine retrieval over frozen training examples."""

    def __init__(self, rows: Iterable[dict[str, Any]]):
        self.rows = [dict(row) for row in rows]
        if not self.rows:
            raise ValueError("At least one training row is required for retrieval.")

        self.term_counts = [Counter(self._tokens(str(row["utterance"]))) for row in self.rows]
        document_frequency = Counter(
            token for counts in self.term_counts for token in counts
        )
        total = len(self.rows)
        self.idf = {
            token: math.log((total + 1) / (frequency + 1)) + 1.0
            for token, frequency in document_frequency.items()
        }
        self.document_norms = [self._norm(counts) for counts in self.term_counts]

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return _RETRIEVAL_TOKEN_PATTERN.findall(text.lower())

    def _norm(self, counts: Counter[str]) -> float:
        return math.sqrt(
            sum((frequency * self.idf.get(token, 0.0)) ** 2 for token, frequency in counts.items())
        )

    def retrieve(self, query: str, *, k: int = 5) -> list[dict[str, Any]]:
        if k <= 0:
            raise ValueError("k must be positive.")
        if k > len(self.rows):
            raise ValueError(f"Requested {k} demonstrations from only {len(self.rows)} rows.")

        query_counts = Counter(self._tokens(query))
        query_norm = self._norm(query_counts)
        scored: list[tuple[float, str, int]] = []
        for index, (row, counts, document_norm) in enumerate(
            zip(self.rows, self.term_counts, self.document_norms)
        ):
            dot = sum(
                query_frequency
                * counts.get(token, 0)
                * self.idf.get(token, 0.0) ** 2
                for token, query_frequency in query_counts.items()
            )
            denominator = query_norm * document_norm
            score = dot / denominator if denominator else 0.0
            scored.append((score, str(row.get("example_id", index)), index))

        scored.sort(key=lambda item: (-item[0], item[1]))
        return [self.rows[index] for _, _, index in scored[:k]]


def stratified_take(
    rows: Iterable[dict[str, Any]],
    *,
    per_label: int,
    seed: int,
    excluded_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Select the same number of examples per label with deterministic shuffling."""
    if per_label <= 0:
        raise ValueError("per_label must be positive.")

    excluded_ids = excluded_ids or set()
    by_label: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if str(row["example_id"]) not in excluded_ids:
            by_label[int(row["label"])].append(dict(row))

    selected: list[dict[str, Any]] = []
    for label_id in sorted(by_label):
        candidates = by_label[label_id]
        random.Random(seed + label_id).shuffle(candidates)
        if len(candidates) < per_label:
            raise ValueError(
                f"Label {label_id} has only {len(candidates)} available rows; {per_label} requested."
            )
        selected.extend(candidates[:per_label])
    return selected


def balanced_evaluation_subset(
    rows: Iterable[dict[str, Any]],
    *,
    examples_per_label: int,
) -> list[dict[str, Any]]:
    """Take the first frozen evaluation examples for every BANKING77 label.

    Prepared evaluation data is already selected deterministically. This helper
    changes only evaluation breadth and, unlike a positional slice, guarantees
    that a 77-row smoke run contains one example from each of the 77 labels.
    """
    if examples_per_label <= 0:
        raise ValueError("examples_per_label must be positive.")

    by_label: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_label[int(row["label_id"])].append(dict(row))

    expected_labels = set(range(77))
    if set(by_label) != expected_labels:
        missing = sorted(expected_labels.difference(by_label))
        unexpected = sorted(set(by_label).difference(expected_labels))
        raise ValueError(
            f"Evaluation rows must contain all 77 labels; missing={missing[:3]}, "
            f"unexpected={unexpected[:3]}."
        )

    selected: list[dict[str, Any]] = []
    for label_id in range(77):
        candidates = by_label[label_id]
        if len(candidates) < examples_per_label:
            raise ValueError(
                f"Label {label_id} has only {len(candidates)} evaluation rows; "
                f"{examples_per_label} requested."
            )
        selected.extend(candidates[:examples_per_label])
    return selected


def as_prepared(row: dict[str, Any], label_names: Sequence[str]) -> PreparedExample:
    label_id = int(row["label"])
    return PreparedExample(
        example_id=str(row["example_id"]),
        utterance=str(row["text"]).strip(),
        label_id=label_id,
        label_name=str(label_names[label_id]),
        route_code=route_code(label_id),
    )


def render_training_record(example: PreparedExample, tokenizer: Any) -> dict[str, Any]:
    """Render one prompt-completion row using the checkpoint's native chat template."""
    prompt = tokenizer.apply_chat_template(
        build_messages(example.utterance),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    marker = tokenizer.apply_chat_template(
        [{"role": "assistant", "content": "__ROUTE_CODE__"}],
        tokenize=False,
    ).split("__ROUTE_CODE__", maxsplit=1)[-1]
    completion = example.route_code + marker
    text = prompt + completion
    length = len(tokenizer(text, add_special_tokens=False).input_ids)
    return {
        "input": prompt,
        "output": completion,
        "text": text,
        "length": length,
        "example_id": example.example_id,
        "label_id": example.label_id,
        "label_name": example.label_name,
        "route_code": example.route_code,
        "utterance": example.utterance,
    }


def render_evaluation_record(example: PreparedExample, tokenizer: Any) -> dict[str, Any]:
    prompt = tokenizer.apply_chat_template(
        build_messages(example.utterance),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return {
        "example_id": example.example_id,
        "prompt": prompt,
        "utterance": example.utterance,
        "label_id": example.label_id,
        "label_name": example.label_name,
        "expected": example.route_code,
    }


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
