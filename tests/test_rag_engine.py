from unittest import TestCase
from unittest.mock import MagicMock, patch

from onprem_rag.services import rag_engine


class RetrievalBehaviorTests(TestCase):
    def test_vague_requests_receive_clarification(self):
        response = rag_engine.clarification_response("Help me.")

        self.assertIsNotNone(response)
        self.assertIn("summarizing", response)
        self.assertIsNone(
            rag_engine.clarification_response("Help me summarize the quality manual")
        )

    def test_summary_queries_use_broader_retrieval(self):
        with (
            patch.object(rag_engine, "TOP_K", 3),
            patch.object(rag_engine, "SUMMARY_TOP_K", 12),
        ):
            self.assertEqual(rag_engine._retrieval_limit("Where is clause 7?"), 3)
            self.assertEqual(
                rag_engine._retrieval_limit("Summarize the entire manual"),
                12,
            )

    def test_vague_request_skips_vector_collection(self):
        with patch.object(rag_engine, "_get_collection") as get_collection:
            result = rag_engine.retrieve("help me")

        self.assertFalse(result["found"])
        self.assertEqual(result["chunks"], [])
        get_collection.assert_not_called()

    def test_search_uses_summary_limit(self):
        collection = MagicMock()
        collection.count.return_value = 20
        collection.query.return_value = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }
        collection.get.return_value = {
            "ids": [],
            "documents": [],
            "metadatas": [],
        }

        with (
            patch.object(rag_engine, "TOP_K", 3),
            patch.object(rag_engine, "SUMMARY_TOP_K", 12),
        ):
            rag_engine._search_chunks(
                collection,
                embedding=[0.1, 0.2],
                question="Give me an overview of the whole document",
            )

        self.assertEqual(collection.query.call_args.kwargs["n_results"], 20)

    def test_exact_document_term_prioritizes_body_over_front_matter(self):
        collection = MagicMock()
        collection.count.return_value = 3
        collection.query.return_value = {
            "ids": [["other"]],
            "documents": [["Unrelated process material"]],
            "metadatas": [[{"document_name": "Manual.pdf", "content_type": "body"}]],
            "distances": [[0.4]],
        }
        records = [
            (
                "front",
                "OCP document control sheet",
                {
                    "document_name": "OCP of Electric hazards.pdf",
                    "content_type": "front_matter",
                    "page_start": "1",
                    "page_end": "4",
                    "section_number": "",
                    "section_title": "Front matter",
                },
            ),
            (
                "hazards",
                "Wet electrical equipment can cause electric shock.",
                {
                    "document_name": "OCP of Electric hazards.pdf",
                    "content_type": "body",
                    "page_start": "5",
                    "page_end": "5",
                    "section_number": "1.0",
                    "section_title": "Potential Hazards",
                },
            ),
            (
                "practices",
                "Electrical tools should be inspected before use.",
                {
                    "document_name": "OCP of Electric hazards.pdf",
                    "content_type": "body",
                    "page_start": "6",
                    "page_end": "6",
                    "section_number": "3.0",
                    "section_title": "Best Practices",
                },
            ),
        ]
        collection.get.return_value = {
            "ids": [record[0] for record in records],
            "documents": [record[1] for record in records],
            "metadatas": [record[2] for record in records],
        }

        chunks = rag_engine._search_chunks(
            collection,
            embedding=[0.1],
            question="Help me learn OCP",
        )

        self.assertEqual(
            [chunk["metadata"]["content_type"] for chunk in chunks[:2]],
            ["body", "body"],
        )
        self.assertIn("Section", chunks[0]["source"])
        self.assertNotIn("chunk", chunks[0]["source"].lower())

    def test_document_title_derives_natural_acronym(self):
        aliases = rag_engine._document_alias_terms(
            "Organization's Standard Process.pdf"
        )

        self.assertIn("osp", aliases)

    def test_unrelated_semantic_results_are_rejected(self):
        collection = MagicMock()
        collection.count.return_value = 3
        collection.query.return_value = {
            "ids": [["fire", "manual", "burns"]],
            "documents": [
                [
                    "Fire prevention requirements.",
                    "Document control procedure.",
                    "First aid for burns.",
                ]
            ],
            "metadatas": [[
                {"document_name": "Fire.pdf", "content_type": "body"},
                {"document_name": "Manual.pdf", "content_type": "body"},
                {"document_name": "Emergency.pdf", "content_type": "body"},
            ]],
            "distances": [[0.59, 0.57, 0.55]],
        }
        collection.get.return_value = {
            "ids": ["fire", "manual", "burns"],
            "documents": [
                "Fire prevention requirements.",
                "Document control procedure.",
                "First aid for burns.",
            ],
            "metadatas": [
                {"document_name": "Fire.pdf", "content_type": "body"},
                {"document_name": "Manual.pdf", "content_type": "body"},
                {"document_name": "Emergency.pdf", "content_type": "body"},
            ],
        }

        chunks = rag_engine._search_chunks(
            collection,
            embedding=[0.1],
            question="How do I bake a chocolate cake?",
        )

        self.assertEqual(chunks, [])

    def test_each_matching_document_keeps_its_own_best_content_type(self):
        collection = MagicMock()
        collection.count.return_value = 2
        records = [
            (
                "matrix",
                "Operational controls and monitoring responsibilities.",
                {
                    "document_name": "Operational control matrix 0.9.pdf",
                    "content_type": "front_matter",
                    "page_start": "1",
                    "page_end": "1",
                },
            ),
            (
                "voc",
                "VOC control precautions.",
                {
                    "document_name": "Control of VOC.pdf",
                    "content_type": "body",
                    "section_number": "2.0",
                    "section_title": "Precautions",
                    "page_start": "5",
                    "page_end": "5",
                },
            ),
        ]
        collection.query.return_value = {
            "ids": [["voc"]],
            "documents": [[records[1][1]]],
            "metadatas": [[records[1][2]]],
            "distances": [[0.4]],
        }
        collection.get.return_value = {
            "ids": [record[0] for record in records],
            "documents": [record[1] for record in records],
            "metadatas": [record[2] for record in records],
        }

        chunks = rag_engine._search_chunks(
            collection,
            embedding=[0.1],
            question="Explain the operational control matrix",
        )

        self.assertIn(
            "Operational control matrix 0.9.pdf",
            [chunk["document_name"] for chunk in chunks],
        )
        self.assertNotIn(
            "Control of VOC.pdf",
            [chunk["document_name"] for chunk in chunks],
        )

    def test_fallback_followed_by_grounded_answer_keeps_continuation(self):
        client = MagicMock()
        client.chat.return_value = iter(
            [
                {
                    "message": {
                        "content": (
                            "I could not find reliable supporting information in "
                            "the provided documents. However, OSP defines the "
                            "organization's standard processes and shows how its "
                            "procedures interact with each other."
                        )
                    }
                }
            ]
        )
        chunks = [
            {
                "text": "Unrelated context.",
                "source": "manual.pdf — section 1",
                "document_name": "manual.pdf",
                "metadata": {},
            }
        ]

        with patch.object(rag_engine.ollama, "Client", return_value=client):
            stream = rag_engine.ask_stream("Explain OSP", chunks=chunks)
            streamed_text = "".join(stream)

        self.assertIn("However", streamed_text)
        self.assertEqual(
            stream.result["answer"],
            (
                "OSP defines the organization's standard processes and shows how "
                "its procedures interact with each other."
            ),
        )
        self.assertTrue(stream.result["found"])
        self.assertEqual(len(stream.result["sources"]), 1)

    def test_bare_fallback_remains_unavailable(self):
        answer, found = rag_engine._normalize_grounded_answer(
            "I could not find reliable supporting information in the provided documents.",
            "I could not find reliable supporting information in the provided documents.",
            has_evidence=True,
        )

        self.assertFalse(found)
        self.assertEqual(
            answer,
            "I could not find reliable supporting information in the provided documents.",
        )
