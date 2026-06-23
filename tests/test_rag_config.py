import unittest

from src.rag_config import default_rag_settings, index_config_key


class RagConfigTests(unittest.TestCase):
    def test_index_config_key_changes_with_chunk_settings(self):
        upload_sig = (("notes.txt", "abc123"),)
        base = default_rag_settings()
        first = index_config_key(upload_sig, base)
        changed = index_config_key(
            upload_sig,
            {**base, "chunk_size": 256},
        )
        self.assertNotEqual(first, changed)


if __name__ == "__main__":
    unittest.main()
