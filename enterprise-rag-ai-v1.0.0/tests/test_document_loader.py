from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from onprem_rag.services.document_loader import load_document


class DocumentLoaderTests(TestCase):
    def test_load_text_document(self):
        with TemporaryDirectory() as temp_dir:
            document = Path(temp_dir) / "policy.txt"
            document.write_text("Approved retention period: seven years.", encoding="utf-8")

            result = load_document(document, registry={})

        self.assertTrue(result["text"].startswith("Approved retention"))
        self.assertEqual(result["metadata"]["document_name"], "policy.txt")

    def test_rejects_unsupported_extension(self):
        with TemporaryDirectory() as temp_dir:
            document = Path(temp_dir) / "payload.exe"
            document.write_bytes(b"not a document")

            with self.assertRaisesRegex(ValueError, "Unsupported file type"):
                load_document(document, registry={})
