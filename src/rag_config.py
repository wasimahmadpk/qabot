import hashlib

DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 128
DEFAULT_CHUNK_STRATEGY = "sentence"
DEFAULT_TOP_K = 3

CHUNK_STRATEGIES = {
    "sentence": "Sentence-aware splitting (respects sentence boundaries)",
    "token": "Token-based splitting (fixed token windows)",
}


def default_rag_settings():
    return {
        "chunk_size": DEFAULT_CHUNK_SIZE,
        "chunk_overlap": DEFAULT_CHUNK_OVERLAP,
        "chunk_strategy": DEFAULT_CHUNK_STRATEGY,
        "top_k": DEFAULT_TOP_K,
    }


def index_config_key(upload_signature, rag_settings):
    """Stable key for uploads + chunk/index configuration."""
    payload = (
        upload_signature,
        rag_settings["chunk_size"],
        rag_settings["chunk_overlap"],
        rag_settings["chunk_strategy"],
    )
    return payload


def collection_name_from_key(index_key):
    digest = hashlib.sha256(repr(index_key).encode()).hexdigest()
    return f"qabot_{digest[:16]}"
