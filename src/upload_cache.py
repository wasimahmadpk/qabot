import hashlib


def upload_signature(uploaded_files):
    """Return a stable cache key for a set of uploaded files."""
    parts = []
    for uploaded_file in uploaded_files:
        content = uploaded_file.getvalue()
        digest = hashlib.sha256(content).hexdigest()
        parts.append((uploaded_file.name, digest))
    return tuple(parts)
