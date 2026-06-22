import unittest

from src.upload_cache import upload_signature


class FakeUpload:
    def __init__(self, name, content):
        self.name = name
        self._content = content

    def getvalue(self):
        return self._content


class UploadSignatureTests(unittest.TestCase):
    def test_signature_is_stable_for_same_files(self):
        files = [FakeUpload("notes.txt", b"hello world")]
        self.assertEqual(upload_signature(files), upload_signature(files))

    def test_signature_changes_when_content_changes(self):
        first = [FakeUpload("notes.txt", b"version one")]
        second = [FakeUpload("notes.txt", b"version two")]
        self.assertNotEqual(upload_signature(first), upload_signature(second))

    def test_signature_changes_when_file_set_changes(self):
        one_file = [FakeUpload("a.txt", b"same")]
        two_files = [
            FakeUpload("a.txt", b"same"),
            FakeUpload("b.txt", b"other"),
        ]
        self.assertNotEqual(upload_signature(one_file), upload_signature(two_files))


if __name__ == "__main__":
    unittest.main()
