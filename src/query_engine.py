import time

from src.prompts import QA_PROMPT, REFINE_PROMPT

DEFAULT_TOP_K = 3


def get_query_engine(index, similarity_top_k=DEFAULT_TOP_K):
    return index.as_query_engine(
        text_qa_template=QA_PROMPT,
        refine_template=REFINE_PROMPT,
        similarity_top_k=similarity_top_k,
    )


def query_index(query_engine, query_text):
    start = time.perf_counter()
    response = query_engine.query(query_text)
    latency_ms = (time.perf_counter() - start) * 1000

    sources = []
    for node in response.source_nodes or []:
        sources.append(
            {
                "file_name": node.metadata.get("file_name", "unknown"),
                "chunk_id": node.metadata.get("chunk_id", "?"),
                "text": node.get_content()[:300],
                "score": getattr(node, "score", None),
            }
        )

    return {
        "answer": str(response),
        "latency_ms": latency_ms,
        "sources": sources,
    }
