import json
import unittest
from unittest.mock import patch

import pandas as pd

from src.evaluation import (
    answer_contains_keywords,
    load_qa_pairs,
    parse_qa_pairs_json,
    retrieval_hit,
    run_evaluation,
    run_ragas_evaluation,
)


class FakeQueryEngine:
    pass


def fake_query(_engine, question):
    if "France" in question:
        return {
            "answer": "I don't know.",
            "latency_ms": 50.0,
            "sources": [{"text": "remote work policy", "file_name": "sample_policy.txt"}],
        }
    return {
        "answer": "Employees may work remotely up to three days per week.",
        "latency_ms": 120.5,
        "sources": [
            {
                "text": "All employees may work remotely up to three days per week.",
                "file_name": "sample_policy.txt",
                "chunk_id": 1,
            }
        ],
    }


class EvaluationTests(unittest.TestCase):
    def test_load_qa_pairs(self):
        pairs = load_qa_pairs()
        self.assertGreaterEqual(len(pairs), 3)
        self.assertIn("question", pairs[0])

    def test_parse_qa_pairs_json(self):
        payload = json.dumps(
            [
                {
                    "question": "Example?",
                    "expected_keywords": ["example"],
                    "file_name": "doc.txt",
                }
            ]
        )
        pairs = parse_qa_pairs_json(payload)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["question"], "Example?")

    def test_parse_qa_pairs_json_requires_question(self):
        with self.assertRaises(ValueError):
            parse_qa_pairs_json(json.dumps([{"expected_keywords": ["x"]}]))

    def test_retrieval_hit_requires_keywords_in_sources(self):
        sources = [{"text": "Use the company VPN for remote access."}]
        self.assertTrue(retrieval_hit(["vpn"], sources))
        self.assertFalse(retrieval_hit(["paris"], sources))

    def test_answer_contains_keywords(self):
        self.assertTrue(answer_contains_keywords("I don't know.", ["don't know"]))
        self.assertFalse(answer_contains_keywords("Paris", ["don't know"]))

    def test_parse_qa_pairs_json_accepts_ground_truth(self):
        payload = json.dumps(
            [
                {
                    "question": "Example?",
                    "expected_keywords": ["example"],
                    "ground_truth": "An example answer.",
                    "file_name": "doc.txt",
                }
            ]
        )
        pairs = parse_qa_pairs_json(payload)
        self.assertEqual(pairs[0]["ground_truth"], "An example answer.")

    def test_run_evaluation_returns_summary(self):
        report = run_evaluation(
            FakeQueryEngine(),
            fake_query,
            qa_pairs=[
                {
                    "question": "How many remote days?",
                    "expected_keywords": ["three"],
                    "file_name": "sample_policy.txt",
                },
                {
                    "question": "Capital of France?",
                    "expected_keywords": ["don't know"],
                    "file_name": "sample_policy.txt",
                },
            ],
        )
        self.assertEqual(report["summary"]["total"], 2)
        self.assertIn("avg_latency_ms", report["summary"])
        self.assertEqual(len(report["results"]), 2)

    @patch("ragas.evaluate")
    def test_run_ragas_evaluation_returns_metric_summary(self, mock_evaluate):
        mock_evaluate.return_value.to_pandas.return_value = pd.DataFrame(
            {
                "faithfulness": [0.9, 0.8],
                "answer_relevancy": [0.85, 0.75],
                "context_recall": [0.95, 0.7],
            }
        )

        report = run_ragas_evaluation(
            FakeQueryEngine(),
            fake_query,
            qa_pairs=[
                {
                    "question": "How many remote days?",
                    "expected_keywords": ["three"],
                    "ground_truth": "Up to three days per week.",
                    "file_name": "sample_policy.txt",
                },
                {
                    "question": "Capital of France?",
                    "expected_keywords": ["don't know"],
                    "ground_truth": "I don't know.",
                    "file_name": "sample_policy.txt",
                },
            ],
        )

        self.assertEqual(report["summary"]["total"], 2)
        self.assertEqual(report["summary"]["faithfulness"], 0.85)
        self.assertEqual(report["summary"]["answer_relevancy"], 0.8)
        self.assertEqual(report["summary"]["context_recall"], 0.82)
        self.assertEqual(len(report["results"]), 2)
        mock_evaluate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
