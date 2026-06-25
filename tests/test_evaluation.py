import json
import unittest
from unittest.mock import patch

import pandas as pd

from src.evaluation import (
    answer_contains_keywords,
    answer_grounded_in_retrieval,
    compute_ir_metrics,
    load_qa_pairs,
    ndcg_at_k,
    parse_qa_pairs_json,
    recall_at_k,
    reciprocal_rank,
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

    def test_answer_grounded_requires_retrieval_unless_refusal(self):
        self.assertTrue(
            answer_grounded_in_retrieval(
                True,
                "Use the company VPN.",
                ["vpn"],
            )
        )
        self.assertFalse(
            answer_grounded_in_retrieval(
                False,
                "Use the company VPN.",
                ["vpn"],
            )
        )
        self.assertTrue(
            answer_grounded_in_retrieval(
                False,
                "I don't know.",
                ["don't know"],
                is_refusal=True,
            )
        )

    def test_ir_metrics_rank_relevant_chunk(self):
        item = {
            "expected_keywords": ["vpn"],
            "file_name": "policy.txt",
        }
        sources = [
            {"text": "General HR onboarding steps.", "file_name": "policy.txt", "chunk_id": 1},
            {"text": "Use the company VPN for remote access.", "file_name": "policy.txt", "chunk_id": 2},
            {"text": "Password rotation every 90 days.", "file_name": "policy.txt", "chunk_id": 3},
        ]
        self.assertEqual(recall_at_k(sources, item, 3), 1.0)
        self.assertEqual(reciprocal_rank(sources, item, 3), 0.5)
        self.assertAlmostEqual(ndcg_at_k(sources, item, 3), 0.631, places=2)

    def test_ir_metrics_zero_when_no_relevant_chunk(self):
        item = {"expected_keywords": ["jenkins"], "file_name": "policy.txt"}
        sources = [
            {"text": "Remote work policy.", "file_name": "policy.txt", "chunk_id": 1},
            {"text": "Expense reporting.", "file_name": "policy.txt", "chunk_id": 2},
        ]
        self.assertEqual(recall_at_k(sources, item, 2), 0.0)
        self.assertEqual(reciprocal_rank(sources, item, 2), 0.0)
        self.assertEqual(ndcg_at_k(sources, item, 2), 0.0)

    def test_ir_metrics_support_relevant_chunk_ids(self):
        item = {"relevant_chunk_ids": [2, 5], "file_name": "policy.txt"}
        sources = [
            {"text": "A", "file_name": "policy.txt", "chunk_id": 1},
            {"text": "B", "file_name": "policy.txt", "chunk_id": 2},
            {"text": "C", "file_name": "policy.txt", "chunk_id": 3},
        ]
        self.assertEqual(recall_at_k(sources, item, 3), 0.5)
        self.assertEqual(reciprocal_rank(sources, item, 3), 0.5)

    def test_ir_metrics_skipped_for_refusal(self):
        self.assertIsNone(
            compute_ir_metrics(
                [{"text": "anything", "file_name": "policy.txt", "chunk_id": 1}],
                {"refusal": True, "expected_keywords": ["don't know"]},
                3,
            )
        )

    def test_grounded_fails_when_answer_hallucinates_keywords(self):
        def hallucinating_query(_engine, _question):
            return {
                "answer": "Employees may work remotely up to three days per week.",
                "latency_ms": 100.0,
                "sources": [
                    {
                        "text": "Managers must approve hybrid schedules.",
                        "file_name": "sample_policy.txt",
                    }
                ],
            }

        report = run_evaluation(
            FakeQueryEngine(),
            hallucinating_query,
            qa_pairs=[
                {
                    "question": "How many remote days?",
                    "expected_keywords": ["three"],
                    "file_name": "sample_policy.txt",
                }
            ],
        )
        row = report["results"][0]
        self.assertFalse(row["retrieval_hit"])
        self.assertFalse(row["answer_grounded"])

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
                    "refusal": True,
                    "file_name": "sample_policy.txt",
                },
            ],
        )
        self.assertEqual(report["summary"]["total"], 2)
        self.assertIn("avg_latency_ms", report["summary"])
        self.assertIn("recall_at_k", report["summary"])
        self.assertIn("mrr", report["summary"])
        self.assertIn("ndcg_at_k", report["summary"])
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
