from __future__ import annotations

import unittest
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_ft_lab.evaluation import (
    extract_route_code,
    generate_nvidia_api_predictions,
    paired_accuracy_comparison,
    score_predictions,
)


class EvaluationTests(unittest.TestCase):
    def test_extract_route_code_is_bounded(self):
        self.assertEqual(extract_route_code("B77_04"), "B77_04")
        self.assertEqual(extract_route_code("answer: b77_76."), "B77_76")
        self.assertIsNone(extract_route_code("B77_77"))
        self.assertIsNone(extract_route_code("card_arrival"))

    def test_score_distinguishes_validity_from_accuracy(self):
        report = score_predictions(
            [
                {"expected": "B77_01", "generated": "B77_01"},
                {"expected": "B77_02", "generated": "B77_03"},
                {"expected": "B77_03", "generated": "not a code"},
            ]
        )
        self.assertAlmostEqual(report["accuracy"], 1 / 3)
        self.assertAlmostEqual(report["valid_code_rate"], 2 / 3)
        self.assertEqual(report["correct"], 1)

    def test_paired_comparison_requires_same_ids_and_counts_changes(self):
        baseline = {"rows": [{"example_id": "a", "correct": False}, {"example_id": "b", "correct": True}]}
        candidate = {"rows": [{"example_id": "a", "correct": True}, {"example_id": "b", "correct": True}]}
        result = paired_accuracy_comparison(baseline, candidate, bootstrap_samples=100, seed=1)
        self.assertEqual(result["absolute_accuracy_gain"], 0.5)
        self.assertEqual(result["improved_examples"], 1)
        self.assertEqual(result["regressed_examples"], 0)

    def test_api_generation_is_resumable_and_uses_non_thinking_messages(self):
        calls = []

        class Completions:
            def create(self, **kwargs):
                calls.append(kwargs)
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="B77_01"))]
                )

        client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
        rows = [{"example_id": "test-1", "utterance": "My card has not arrived", "expected": "B77_01"}]
        with tempfile.TemporaryDirectory() as directory:
            resume_path = Path(directory) / "responses.jsonl"
            first = generate_nvidia_api_predictions(
                client, rows, model="nvidia/test", resume_path=resume_path
            )
            second = generate_nvidia_api_predictions(
                client, rows, model="nvidia/test", resume_path=resume_path
            )
        self.assertEqual(first, second)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["messages"][0]["role"], "system")
        self.assertFalse(calls[0]["extra_body"]["chat_template_kwargs"]["enable_thinking"])

    def test_api_generation_accepts_a_prompt_condition_builder(self):
        calls = []

        class Completions:
            def create(self, **kwargs):
                calls.append(kwargs)
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="B77_01"))]
                )

        client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
        rows = [{"example_id": "test-1", "utterance": "hello", "expected": "B77_01"}]
        generate_nvidia_api_predictions(
            client,
            rows,
            model="nvidia/test",
            message_builder=lambda utterance: [
                {"role": "system", "content": "taxonomy prompt"},
                {"role": "user", "content": utterance},
            ],
        )
        self.assertEqual(calls[0]["messages"][0]["content"], "taxonomy prompt")

    def test_api_generation_paces_requests_below_the_rpm_limit(self):
        calls = []
        now = [0.0]
        sleeps = []

        class Completions:
            def create(self, **kwargs):
                calls.append(kwargs)
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="B77_01"))]
                )

        def fake_sleep(seconds):
            sleeps.append(seconds)
            now[0] += seconds

        client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
        rows = [
            {"example_id": "test-1", "utterance": "one", "expected": "B77_01"},
            {"example_id": "test-2", "utterance": "two", "expected": "B77_01"},
        ]
        with (
            patch("nemotron_ft_lab.evaluation.time.monotonic", side_effect=lambda: now[0]),
            patch("nemotron_ft_lab.evaluation.time.sleep", side_effect=fake_sleep),
        ):
            generate_nvidia_api_predictions(
                client, rows, model="nvidia/test", requests_per_minute=30
            )
        self.assertEqual(len(calls), 2)
        self.assertEqual(sleeps, [2.0])

    def test_api_generation_honors_retry_after_on_rate_limit(self):
        attempts = []

        class TrialRateLimitError(Exception):
            status_code = 429
            response = SimpleNamespace(headers={"retry-after": "3"})

        class Completions:
            def create(self, **kwargs):
                attempts.append(kwargs)
                if len(attempts) == 1:
                    raise TrialRateLimitError("quota")
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="B77_01"))]
                )

        client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
        rows = [{"example_id": "test-1", "utterance": "one", "expected": "B77_01"}]
        with patch("nemotron_ft_lab.evaluation.time.sleep") as sleep:
            generated = generate_nvidia_api_predictions(
                client,
                rows,
                model="nvidia/test",
                requests_per_minute=None,
                max_attempts=2,
            )
        self.assertEqual(generated[0]["generated"], "B77_01")
        self.assertEqual(len(attempts), 2)
        sleep.assert_called_once_with(4.0)


if __name__ == "__main__":
    unittest.main()
