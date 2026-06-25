import json
from pathlib import Path

DEFAULT_EVAL_PATH = Path(__file__).resolve().parent.parent / "eval" / "qa_pairs.json"

RAGAS_METRIC_KEYS = ("faithfulness", "answer_relevancy", "context_recall")


def load_qa_pairs(path=DEFAULT_EVAL_PATH):
    with open(path, encoding="utf-8") as handle:
        return parse_qa_pairs_json(handle.read())


def parse_qa_pairs_json(raw_json):
    """Parse and validate a golden Q&A JSON payload."""
    data = json.loads(raw_json)
    if not isinstance(data, list):
        raise ValueError("Evaluation JSON must be a list of question objects.")

    validated = []
    for index, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Item {index} must be an object.")
        if "question" not in item:
            raise ValueError(f"Item {index} is missing 'question'.")
        entry = {
            "question": str(item["question"]),
            "expected_keywords": list(item.get("expected_keywords", [])),
            "file_name": item.get("file_name"),
        }
        if item.get("ground_truth") is not None:
            entry["ground_truth"] = str(item["ground_truth"])
        validated.append(entry)
    return validated


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


def _filter_sources(sources, required_file, fallback_sources):
    if not required_file:
        return sources
    filtered = [
        source for source in sources if source.get("file_name") == required_file
    ]
    return filtered or fallback_sources


def _query_response(query_engine, query_fn, question, full_context=False):
    try:
        return query_fn(query_engine, question, full_context=full_context)
    except TypeError:
        return query_fn(query_engine, question)


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

        response = _query_response(query_engine, query_fn, question)
        latency_ms = response["latency_ms"]
        latencies.append(latency_ms)

        sources = _filter_sources(
            response["sources"],
            required_file,
            response["sources"],
        )

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


def _build_ragas_dataset(rows):
    from datasets import Dataset

    dataset_dict = {
        "user_input": [row["user_input"] for row in rows],
        "response": [row["response"] for row in rows],
        "retrieved_contexts": [row["retrieved_contexts"] for row in rows],
    }
    if all(row.get("reference") for row in rows):
        dataset_dict["reference"] = [row["reference"] for row in rows]
    return Dataset.from_dict(dataset_dict)


def _select_ragas_metrics(rows):
    from ragas.metrics import answer_relevancy, context_recall, faithfulness

    metrics = [faithfulness, answer_relevancy]
    if all(row.get("reference") for row in rows):
        metrics.append(context_recall)
    return metrics


def _ragas_summary_from_df(df, metric_keys):
    summary = {}
    for key in metric_keys:
        if key in df.columns:
            summary[key] = round(float(df[key].mean()), 2)
    return summary


def run_ragas_evaluation(query_engine, query_fn, qa_pairs=None, path=DEFAULT_EVAL_PATH):
    """Run RAGAS LLM-as-judge metrics on the golden Q&A set."""
    try:
        from ragas import evaluate
    except ImportError as exc:
        raise ImportError(
            "RAGAS is not installed. Run: pip install ragas langchain-community"
        ) from exc

    if qa_pairs is None:
        qa_pairs = load_qa_pairs(path)

    rows = []
    latencies = []

    for item in qa_pairs:
        question = item["question"]
        required_file = item.get("file_name")

        response = _query_response(
            query_engine,
            query_fn,
            question,
            full_context=True,
        )
        latency_ms = response["latency_ms"]
        latencies.append(latency_ms)

        sources = _filter_sources(
            response["sources"],
            required_file,
            response["sources"],
        )

        row = {
            "question": question,
            "user_input": question,
            "response": response["answer"],
            "retrieved_contexts": [source.get("text", "") for source in sources],
            "latency_ms": round(latency_ms, 1),
        }
        ground_truth = item.get("ground_truth")
        if ground_truth:
            row["reference"] = ground_truth
        rows.append(row)

    dataset = _build_ragas_dataset(rows)
    metrics = _select_ragas_metrics(rows)
    metric_keys = [metric.name for metric in metrics]

    ragas_result = evaluate(dataset, metrics=metrics, show_progress=False)
    scores_df = ragas_result.to_pandas()

    summary = _ragas_summary_from_df(scores_df, metric_keys)
    summary["total"] = len(rows)
    summary["avg_latency_ms"] = round(sum(latencies) / (len(rows) or 1), 1)

    results = []
    for index, row in enumerate(rows):
        result = {
            "question": row["question"],
            "latency_ms": row["latency_ms"],
            "answer_preview": row["response"][:160],
        }
        for key in metric_keys:
            if key in scores_df.columns:
                result[key] = round(float(scores_df.iloc[index][key]), 2)
        results.append(result)

    return {"results": results, "summary": summary, "metrics": metric_keys}
