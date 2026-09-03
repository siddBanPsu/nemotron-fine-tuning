from __future__ import annotations

import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_ft_lab.evaluation import (
    execute_read_only_sql,
    extract_sql,
    generate_nvidia_api_predictions,
    normalize_sql,
    paired_execution_comparison,
    score_predictions,
)


def evaluation_row(generated: str) -> dict:
    return {
        "example_id": "bird-mini-dev-0001",
        "question_id": 1,
        "db_id": "company",
        "difficulty": "simple",
        "question": "Who works in sales?",
        "evidence": "",
        "schema": "CREATE TABLE staff(name TEXT, dept TEXT);",
        "expected_sql": "SELECT name FROM staff WHERE dept = 'sales'",
        "database_path": "company.sqlite",
        "generated": generated,
    }


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        connection = sqlite3.connect(self.root / "company.sqlite")
        connection.execute("CREATE TABLE staff(name TEXT, dept TEXT)")
        connection.executemany(
            "INSERT INTO staff VALUES (?, ?)",
            [("Ada", "sales"), ("Grace", "engineering"), ("Lin", "sales")],
        )
        connection.commit()
        connection.close()

    def tearDown(self):
        self.temp.cleanup()

    def test_extract_sql_removes_thinking_fences_and_prose(self):
        text = "<think>reason</think>Here is the query:\n```sql\nSELECT name FROM staff;\n```"
        self.assertEqual(extract_sql(text), "SELECT name FROM staff")
        self.assertIsNone(extract_sql("I cannot answer"))
        self.assertIsNone(extract_sql("DELETE FROM staff"))
        self.assertIsNone(extract_sql("SELECT name FROM staff; DROP TABLE staff"))
        self.assertIsNone(extract_sql("SELECT name FROM staff WHERE note = 'unterminated"))
        self.assertEqual(
            normalize_sql("select NAME from STAFF where DEPT='sales'"),
            "SELECT name FROM staff WHERE dept = 'sales'",
        )

    def test_execution_uses_bird_set_semantics_and_read_only_database(self):
        expected = execute_read_only_sql(
            self.root / "company.sqlite", "SELECT name FROM staff WHERE dept='sales'"
        )
        reversed_order = execute_read_only_sql(
            self.root / "company.sqlite",
            "SELECT name FROM staff WHERE dept='sales' ORDER BY name DESC",
        )
        self.assertEqual(expected, reversed_order)
        with self.assertRaises(ValueError):
            execute_read_only_sql(self.root / "company.sqlite", "DROP TABLE staff")

    def test_score_separates_syntax_execution_and_string_match(self):
        equivalent = evaluation_row("SELECT name FROM staff WHERE dept='sales' ORDER BY name DESC")
        report = score_predictions([equivalent], data_dir=self.root)
        self.assertEqual(report["execution_accuracy"], 1.0)
        self.assertEqual(report["sql_valid_rate"], 1.0)
        self.assertEqual(report["sql_executable_rate"], 1.0)
        self.assertEqual(report["normalized_exact_match"], 0.0)

        invalid = score_predictions([evaluation_row("not sql")], data_dir=self.root)
        self.assertEqual(invalid["execution_accuracy"], 0.0)
        self.assertEqual(invalid["sql_valid_rate"], 0.0)

        escaped = evaluation_row("SELECT name FROM staff")
        escaped["database_path"] = "../company.sqlite"
        escaped_report = score_predictions([escaped], data_dir=self.root)
        self.assertEqual(escaped_report["sql_executable_rate"], 0.0)
        self.assertIn("escapes", escaped_report["rows"][0]["execution_error"])

    def test_paired_comparison_requires_identical_ids(self):
        baseline = {
            "rows": [
                {"example_id": "a", "execution_correct": False},
                {"example_id": "b", "execution_correct": True},
            ]
        }
        candidate = {
            "rows": [
                {"example_id": "a", "execution_correct": True},
                {"example_id": "b", "execution_correct": True},
            ]
        }
        result = paired_execution_comparison(baseline, candidate, bootstrap_samples=100, seed=1)
        self.assertEqual(result["absolute_execution_accuracy_gain"], 0.5)
        self.assertEqual(result["improved_examples"], 1)
        with self.assertRaisesRegex(ValueError, "positive"):
            paired_execution_comparison(baseline, candidate, bootstrap_samples=0)

    def test_api_generation_is_resumable_and_disables_thinking(self):
        calls = []

        class Completions:
            def create(self, **kwargs):
                calls.append(kwargs)
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="SELECT 1"))])

        client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
        row = evaluation_row("")
        with tempfile.TemporaryDirectory() as directory:
            resume = Path(directory) / "responses.jsonl"
            first = generate_nvidia_api_predictions(client, [row], model="nvidia/test", resume_path=resume)
            second = generate_nvidia_api_predictions(client, [row], model="nvidia/test", resume_path=resume)
        self.assertEqual(first, second)
        self.assertEqual(len(calls), 1)
        self.assertIn("CREATE TABLE staff", calls[0]["messages"][1]["content"])
        self.assertFalse(calls[0]["extra_body"]["chat_template_kwargs"]["enable_thinking"])

    def test_api_generation_paces_and_honors_retry_after(self):
        attempts = []

        class TrialRateLimitError(Exception):
            status_code = 429
            response = SimpleNamespace(headers={"retry-after": "3"})

        class Completions:
            def create(self, **kwargs):
                attempts.append(kwargs)
                if len(attempts) == 1:
                    raise TrialRateLimitError("quota")
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="SELECT 1"))])

        client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
        with patch("nemotron_ft_lab.evaluation.time.sleep") as sleep:
            result = generate_nvidia_api_predictions(
                client,
                [evaluation_row("")],
                model="nvidia/test",
                requests_per_minute=None,
                max_attempts=2,
            )
        self.assertEqual(result[0]["generated"], "SELECT 1")
        sleep.assert_called_once_with(4.0)


if __name__ == "__main__":
    unittest.main()
