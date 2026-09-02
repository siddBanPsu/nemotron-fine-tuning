from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
import unittest
import zipfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_ft_lab.constants import EVAL_EXCLUDED_QUESTION_IDS, EVALUATION_PROTOCOL_VERSION
from nemotron_ft_lab.data import (
    balanced_evaluation_subset,
    build_messages,
    determine_eot_marker,
    find_minidev_root,
    render_evaluation_record,
    render_training_record,
    safe_extract_zip,
    schema_from_sqlite,
    text2sql_user_content,
)


class FakeEncoding:
    def __init__(self, text: str):
        self.input_ids = text.split()


class FakeTokenizer:
    def apply_chat_template(self, messages, *, tokenize=False, add_generation_prompt=False, **kwargs):
        rendered = "".join(f"<{row['role']}>{row['content']}" for row in messages)
        if add_generation_prompt:
            rendered += "<think>\n" if kwargs.get("enable_thinking") else "<think></think>"
        elif messages[-1]["role"] == "assistant":
            rendered += "</assistant>"
        return rendered

    def __call__(self, text, **kwargs):
        return FakeEncoding(text)


class DataTests(unittest.TestCase):
    def test_evaluation_protocol_pins_known_slow_gold_query(self):
        self.assertEqual(EVALUATION_PROTOCOL_VERSION, 1)
        self.assertEqual(EVAL_EXCLUDED_QUESTION_IDS, (701,))

    def test_prompt_matches_official_schema_question_evidence_layout(self):
        content = text2sql_user_content("CREATE TABLE x(id INT);", "Count rows", "rows means records")
        self.assertEqual(content, "CREATE TABLE x(id INT);\n\nCount rows\n\nrows means records")
        messages = build_messages({"schema": "S", "question": "Q", "evidence": "E"})
        self.assertEqual(messages[0], {"role": "system", "content": ""})
        self.assertEqual(messages[1]["content"], "S\n\nQ\n\nE")

    def test_training_render_supports_direct_and_reasoning_examples(self):
        tokenizer = FakeTokenizer()
        marker = determine_eot_marker(tokenizer)
        row = {
            "db_id": "company",
            "_source_index": 7,
            "schema": "CREATE TABLE staff(id INT);",
            "question": "Count staff",
            "evidence": "",
            "SQL": "SELECT COUNT(*) FROM staff",
            "reasoning_trace": "Use COUNT.",
        }
        direct = render_training_record(row, tokenizer, include_reasoning=False, eot_marker=marker)
        reasoning = render_training_record(row, tokenizer, include_reasoning=True, eot_marker=marker)
        self.assertIn("<think></think>", direct["input"])
        self.assertEqual(direct["output"], "SELECT COUNT(*) FROM staff</assistant>")
        self.assertIn("<think>\n", reasoning["input"])
        self.assertEqual(reasoning["output"], "Use COUNT.</think>SELECT COUNT(*) FROM staff</assistant>")

    def test_balanced_subset_is_deterministic_and_keeps_difficulty_mix(self):
        rows = []
        question_id = 0
        for difficulty, count in (("simple", 150), ("moderate", 250), ("challenging", 100)):
            for index in range(count):
                rows.append(
                    {
                        "question_id": question_id,
                        "difficulty": difficulty,
                        "db_id": f"db_{index % 10}",
                    }
                )
                question_id += 1
        first = balanced_evaluation_subset(rows, size=100, seed=1234)
        second = balanced_evaluation_subset(rows, size=100, seed=1234)
        self.assertEqual(first, second)
        self.assertEqual(
            Counter(row["difficulty"] for row in first),
            Counter({"simple": 30, "moderate": 50, "challenging": 20}),
        )
        self.assertEqual(len({row["db_id"] for row in first}), 10)

    def test_schema_and_evaluation_record_point_to_readable_sqlite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db = root / "company.sqlite"
            connection = sqlite3.connect(db)
            connection.execute("CREATE TABLE staff(id INTEGER PRIMARY KEY, name TEXT)")
            connection.commit()
            connection.close()
            schema = schema_from_sqlite(db)
            row = render_evaluation_record(
                {
                    "question_id": 4,
                    "db_id": "company",
                    "difficulty": "simple",
                    "question": "Count staff",
                    "evidence": "",
                    "SQL": "SELECT COUNT(*) FROM staff",
                },
                schema=schema,
                database_relative_path="company.sqlite",
            )
        self.assertIn("CREATE TABLE staff", schema)
        self.assertEqual(row["example_id"], "bird-mini-dev-0004")
        self.assertEqual(row["expected_sql"], "SELECT COUNT(*) FROM staff")

    def test_safe_zip_rejects_traversal_and_finds_minidev_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bad = root / "bad.zip"
            with zipfile.ZipFile(bad, "w") as archive:
                archive.writestr("../escape.txt", "bad")
            with self.assertRaises(RuntimeError):
                safe_extract_zip(bad, root / "out")

            expected = root / "package/minidev/MINIDEV"
            (expected / "dev_databases").mkdir(parents=True)
            (expected / "mini_dev_sqlite.json").write_text(json.dumps([]))
            self.assertEqual(find_minidev_root(root / "package"), expected)


if __name__ == "__main__":
    unittest.main()
