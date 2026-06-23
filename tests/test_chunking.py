import unittest

from llama_index.core import Document

from src.chunking import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    chunk_documents,
    chunk_stats,
)


class ChunkingTests(unittest.TestCase):
    def test_splits_long_document_into_multiple_chunks(self):
        text = "word " * 400
        docs = [Document(text=text, metadata={"file_name": "long.txt"})]
        nodes = chunk_documents(docs, chunk_size=128, chunk_overlap=16)
        self.assertGreater(len(nodes), 1)

    def test_short_document_stays_single_chunk(self):
        docs = [Document(text="short text", metadata={"file_name": "short.txt"})]
        nodes = chunk_documents(docs, chunk_size=512, chunk_overlap=128)
        self.assertEqual(len(nodes), 1)

    def test_chunk_metadata_includes_file_name_and_chunk_id(self):
        docs = [Document(text="alpha beta gamma", metadata={"file_name": "a.txt"})]
        nodes = chunk_documents(docs)
        self.assertEqual(nodes[0].metadata["file_name"], "a.txt")
        self.assertEqual(nodes[0].metadata["chunk_id"], 1)

    def test_chunk_stats_groups_by_file(self):
        docs = [
            Document(text="one two three four", metadata={"file_name": "a.txt"}),
            Document(text="five six seven eight", metadata={"file_name": "b.txt"}),
        ]
        nodes = chunk_documents(docs, chunk_size=10, chunk_overlap=0)
        stats = chunk_stats(nodes)
        self.assertEqual(stats["total_chunks"], len(nodes))
        self.assertIn("a.txt", stats["chunks_by_file"])
        self.assertIn("b.txt", stats["chunks_by_file"])

    def test_defaults_match_documented_strategy(self):
        self.assertEqual(DEFAULT_CHUNK_SIZE, 512)
        self.assertEqual(DEFAULT_CHUNK_OVERLAP, 128)


if __name__ == "__main__":
    unittest.main()
