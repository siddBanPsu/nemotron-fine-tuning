from __future__ import annotations

import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemotron_ft_lab.data import (
    LexicalDemonstrationRetriever,
    PreparedExample,
    balanced_evaluation_subset,
    build_few_shot_messages,
    build_taxonomy_messages,
    render_evaluation_record,
    render_training_record,
    route_code,
    stratified_take,
)


class FakeEncoding:
    def __init__(self, text: str):
        self.input_ids = text.split()


class FakeTokenizer:
    def apply_chat_template(self, messages, *, tokenize=False, add_generation_prompt=False, **kwargs):
        rendered = "".join(f"<{row['role']}>{row['content']}" for row in messages)
        if add_generation_prompt:
            rendered += "<assistant>"
        return rendered

    def __call__(self, text, **kwargs):
        return FakeEncoding(text)


class DataTests(unittest.TestCase):
    def test_route_code_boundaries(self):
        self.assertEqual(route_code(0), "B77_61")
        self.assertEqual(route_code(76), "B77_52")
        with self.assertRaises(ValueError):
            route_code(77)

    def test_stratified_take_is_balanced_and_deterministic(self):
        rows = [
            {"example_id": f"{label}-{index}", "label": label, "text": "x"}
            for label in range(3)
            for index in range(5)
        ]
        first = stratified_take(rows, per_label=2, seed=9)
        second = stratified_take(rows, per_label=2, seed=9)
        self.assertEqual(first, second)
        self.assertEqual([row["label"] for row in first].count(0), 2)
        self.assertEqual([row["label"] for row in first].count(1), 2)
        self.assertEqual([row["label"] for row in first].count(2), 2)

    def test_balanced_evaluation_subset_is_one_per_label_not_a_slice(self):
        rows = [
            {"example_id": f"{label}-{index}", "label_id": label}
            for label in range(77)
            for index in range(3)
        ]
        selected = balanced_evaluation_subset(rows, examples_per_label=1)
        self.assertEqual(len(selected), 77)
        self.assertEqual({row["label_id"] for row in selected}, set(range(77)))
        self.assertEqual([row["example_id"] for row in selected[:3]], ["0-0", "1-0", "2-0"])

    def test_balanced_evaluation_subset_rejects_incomplete_labels(self):
        with self.assertRaises(ValueError):
            balanced_evaluation_subset(
                [{"example_id": "0-0", "label_id": 0}], examples_per_label=1
            )

    def test_training_and_evaluation_render_share_prompt(self):
        example = PreparedExample("test-00001", "Where is my card?", 11, "card_arrival", "B77_72")
        tokenizer = FakeTokenizer()
        training = render_training_record(example, tokenizer)
        evaluation = render_evaluation_record(example, tokenizer)
        self.assertEqual(training["input"], evaluation["prompt"])
        self.assertTrue(training["output"].startswith("B77_72"))
        self.assertEqual(evaluation["expected"], "B77_72")

    def test_taxonomy_prompt_exposes_the_private_mapping(self):
        messages = build_taxonomy_messages(
            "Where is my card?",
            {"B77_72": {"label_id": 11, "label_name": "card_arrival"}},
        )
        self.assertIn("B77_72 = card arrival", messages[0]["content"])
        self.assertEqual(messages[-1]["content"], "Where is my card?")

    def test_lexical_retrieval_and_few_shot_messages_use_training_rows(self):
        training_rows = [
            {"example_id": "train-1", "utterance": "cash withdrawal fee", "route_code": "B77_01"},
            {"example_id": "train-2", "utterance": "card delivery is late", "route_code": "B77_02"},
            {"example_id": "train-3", "utterance": "transfer still pending", "route_code": "B77_03"},
        ]
        retriever = LexicalDemonstrationRetriever(training_rows)
        demonstrations = retriever.retrieve("Why is my card delivery late?", k=2)
        self.assertEqual(demonstrations[0]["example_id"], "train-2")
        messages = build_few_shot_messages("Where is the card?", demonstrations)
        self.assertEqual(messages[1]["content"], "card delivery is late")
        self.assertEqual(messages[2]["content"], "B77_02")
        self.assertEqual(messages[-1]["content"], "Where is the card?")


if __name__ == "__main__":
    unittest.main()
