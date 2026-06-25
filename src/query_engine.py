import time

from src.prompts import QA_PROMPT, REFINE_PROMPT

DEFAULT_TOP_K = 3


def get_query_engine(index, similarity_top_k=DEFAULT_TOP_K):
    return index.as_query_engine(
        text_qa_template=QA_PROMPT,
        refine_template=REFINE_PROMPT,
        similarity_top_k=similarity_top_k,
    )


def _source_from_node(node, full_context=False):
    content = node.get_content()
    if not full_context:
        content = content[:300]
    return {
        "file_name": node.metadata.get("file_name", "unknown"),
        "chunk_id": node.metadata.get("chunk_id", "?"),
        "text": content,
        "score": getattr(node, "score", None),
    }


def retrieve_ranked(index, query_text, similarity_top_k=DEFAULT_TOP_K, full_context=True):
    """Retrieve ranked chunks without running the LLM (for IR metrics)."""
    start = time.perf_counter()
    retriever = index.as_retriever(similarity_top_k=similarity_top_k)
    nodes = retriever.retrieve(query_text)
    latency_ms = (time.perf_counter() - start) * 1000
    sources = [_source_from_node(node, full_context=full_context) for node in nodes]
    return {"sources": sources, "latency_ms": latency_ms}


def query_index(query_engine, query_text, full_context=False):
    start = time.perf_counter()
    response = query_engine.query(query_text)
    latency_ms = (time.perf_counter() - start) * 1000

    sources = []
    for node in response.source_nodes or []:
        sources.append(_source_from_node(node, full_context=full_context))

    return {
        "answer": str(response),
        "latency_ms": latency_ms,
        "sources": sources,
    }
