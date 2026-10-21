import unittest

from research.experiments.stage3.run_stage3 import citation_validity, evaluate_text


class Stage3EvaluationTests(unittest.TestCase):
    def test_citation_metadata_requires_unique_structured_sources(self):
        sources = [
            {"doc_id": "doc-a", "section_id": 1, "chunk_id": 0, "rank": 1, "text": "evidence"}
        ]
        result = citation_validity(sources)
        self.assertTrue(result["citation_record_exists"])
        self.assertTrue(result["citation_metadata_valid"])
        self.assertEqual(result["semantic_support"], "unscored_manual_review")

    def test_duplicate_source_records_are_not_valid_citations(self):
        source = {"doc_id": "doc-a", "section_id": 1, "chunk_id": 0, "rank": 1, "text": "evidence"}
        result = citation_validity([source, dict(source)])
        self.assertFalse(result["citation_metadata_valid"])

    def test_text_diagnostics_are_explicitly_descriptive(self):
        result = evaluate_text(
            "The detector uses evidence from the passage.",
            "What does the detector use?",
            "The detector uses evidence from the passage.",
            [{"text": "The detector uses evidence from the passage."}],
        )
        self.assertTrue(result["answer_nonempty"])
        self.assertGreater(result["evidence_token_overlap"], 0.5)
        self.assertIn("reference_token_recall_descriptive", result)


if __name__ == "__main__":
    unittest.main()
