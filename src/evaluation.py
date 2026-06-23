import json
from pathlib import Path

DEFAULT_EVAL_PATH = Path(__file__).resolve().parent.parent / "eval" / "qa_pairs.json"


def load_qa_pairs(path=DEFAULT_EVAL_PATH):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def retrieval_hit(expected_keywords, sources):
    """True if any retrieved chunk contains all expected keywords."""
    if not expected_keywords:
        return True

    combined = " ".join(source.get("text", "") for source in sources).lower()
    return all(keyword.lower() in combined for keyword in expected_keywords)


def answer_contains_keywords(answer, expected_keywords):
    if not expected_keywords:
        return True
    answer_lower = answer.lower()
    return all(keyword.lower() in answer_lower for keyword in expected_keywords)


def run_evaluation(query_engine, query_fn, qa_pairs=None, path=DEFAULT_EVAL_PATH):
    """Run golden Q&A checks and return aggregate metrics."""
    if qa_pairs is None:
        qa_pairs = load_qa_pairs(path)

    results = []
    latencies = []

    for item in qa_pairs:
        question = item["question"]
        expected_keywords = item.get("expected_keywords", [])
        required_file = item.get("file_name")

        response = query_fn(query_engine, question)
        latency_ms = response["latency_ms"]
        latencies.append(latency_ms)

        sources = response["sources"]
        if required_file:
            sources = [
                source
                for source in sources
                if source.get("file_name") == required_file
            ] or response["sources"]

        hit = retrieval_hit(expected_keywords, sources)
        grounded = answer_contains_keywords(response["answer"], expected_keywords)

        results.append(
            {
                "question": question,
                "latency_ms": round(latency_ms, 1),
                "retrieval_hit": hit,
                "answer_grounded": grounded,
                "answer_preview": response["answer"][:160],
            }
        )

    count = len(results) or 1
    return {
        "results": results,
        "summary": {
            "total": len(results),
            "avg_latency_ms": round(sum(latencies) / count, 1),
            "retrieval_hit_rate": round(
                sum(r["retrieval_hit"] for r in results) / count,
                2,
            ),
            "answer_grounded_rate": round(
                sum(r["answer_grounded"] for r in results) / count,
                2,
            ),
        },
    }
