import json
import math
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
        if item.get("refusal"):
            entry["refusal"] = True
        if item.get("relevant_chunk_ids") is not None:
            entry["relevant_chunk_ids"] = list(item["relevant_chunk_ids"])
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


def answer_grounded_in_retrieval(retrieval_hit, answer, expected_keywords, is_refusal=False):
    """True when the answer reflects retrieved evidence, not outside knowledge."""
    has_keywords = answer_contains_keywords(answer, expected_keywords)
    if is_refusal:
        return has_keywords
    if not expected_keywords:
        return True
    return retrieval_hit and has_keywords


def chunk_relevance(source, expected_keywords, file_name=None):
    """Binary relevance for a single retrieved chunk."""
    if file_name and source.get("file_name") != file_name:
        return 0
    if not expected_keywords:
        return 1
    text = source.get("text", "").lower()
    return int(all(keyword.lower() in text for keyword in expected_keywords))


def chunk_relevance_by_id(source, relevant_chunk_ids, file_name=None):
    if file_name and source.get("file_name") != file_name:
        return 0
    return int(source.get("chunk_id") in relevant_chunk_ids)


def ranked_relevances(sources, item, k):
    """Relevance grades for the top-k ranked retrieval list."""
    relevances = []
    relevant_chunk_ids = item.get("relevant_chunk_ids")
    for source in sources[:k]:
        if relevant_chunk_ids:
            rel = chunk_relevance_by_id(
                source,
                relevant_chunk_ids,
                item.get("file_name"),
            )
        else:
            rel = chunk_relevance(
                source,
                item.get("expected_keywords", []),
                item.get("file_name"),
            )
        relevances.append(rel)
    return relevances


def recall_at_k(sources, item, k):
    """Fraction of known relevant chunks found in the top-k results."""
    relevant_chunk_ids = item.get("relevant_chunk_ids")
    if relevant_chunk_ids:
        retrieved_ids = [source.get("chunk_id") for source in sources[:k]]
        found = sum(1 for chunk_id in relevant_chunk_ids if chunk_id in retrieved_ids)
        return found / len(relevant_chunk_ids)

    relevances = ranked_relevances(sources, item, k)
    return 1.0 if any(relevances) else 0.0


def reciprocal_rank(sources, item, k):
    """Reciprocal rank of the first relevant chunk in top-k (1/rank, or 0)."""
    for rank, rel in enumerate(ranked_relevances(sources, item, k), start=1):
        if rel:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(sources, item, k):
    """Normalized DCG@k using binary chunk relevance."""
    relevances = ranked_relevances(sources, item, k)

    def dcg(scores):
        return sum(rel / math.log2(index + 2) for index, rel in enumerate(scores[:k]))

    actual = dcg(relevances)
    ideal = dcg(sorted(relevances, reverse=True))
    if ideal == 0:
        return 0.0
    return actual / ideal


def compute_ir_metrics(sources, item, k):
    """Compute Recall@k, MRR, and NDCG@k for one ranked retrieval list."""
    if item.get("refusal"):
        return None
    return {
        "recall_at_k": round(recall_at_k(sources, item, k), 3),
        "mrr": round(reciprocal_rank(sources, item, k), 3),
        "ndcg_at_k": round(ndcg_at_k(sources, item, k), 3),
    }


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


def run_evaluation(
    query_engine,
    query_fn,
    qa_pairs=None,
    path=DEFAULT_EVAL_PATH,
    index=None,
    top_k=3,
):
    """Run golden Q&A checks and return aggregate metrics."""
    if qa_pairs is None:
        qa_pairs = load_qa_pairs(path)

    results = []
    latencies = []
    ir_scores = []

    for item in qa_pairs:
        question = item["question"]
        expected_keywords = item.get("expected_keywords", [])
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

        if index is not None:
            from src.query_engine import retrieve_ranked

            ranked = retrieve_ranked(
                index,
                question,
                similarity_top_k=top_k,
                full_context=True,
            )
            ranked_sources = _filter_sources(
                ranked["sources"],
                required_file,
                ranked["sources"],
            )
        else:
            ranked_sources = sources

        hit = retrieval_hit(expected_keywords, sources)
        is_refusal = bool(item.get("refusal"))
        grounded = answer_grounded_in_retrieval(
            hit,
            response["answer"],
            expected_keywords,
            is_refusal=is_refusal,
        )
        ir_metrics = compute_ir_metrics(ranked_sources, item, top_k)

        row = {
            "question": question,
            "latency_ms": round(latency_ms, 1),
            "retrieval_hit": hit,
            "answer_grounded": grounded,
            "answer_preview": response["answer"][:160],
        }
        if ir_metrics:
            row.update(ir_metrics)
            ir_scores.append(ir_metrics)
        results.append(row)

    count = len(results) or 1
    summary = {
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
        "top_k": top_k,
    }
    if ir_scores:
        summary["recall_at_k"] = round(
            sum(score["recall_at_k"] for score in ir_scores) / len(ir_scores),
            3,
        )
        summary["mrr"] = round(
            sum(score["mrr"] for score in ir_scores) / len(ir_scores),
            3,
        )
        summary["ndcg_at_k"] = round(
            sum(score["ndcg_at_k"] for score in ir_scores) / len(ir_scores),
            3,
        )

    return {"results": results, "summary": summary}


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
