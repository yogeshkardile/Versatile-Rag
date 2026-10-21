import unittest

from app.services.generation import (
    ExtractiveFallbackProvider,
    GenerationContext,
    GenerationResult,
    generate_with_fallback,
)


class MockProvider:
    name = "mock"
    model_name = "mock-model"
    initialization_ms = 1.0

    def generate(self, question, contexts):
        return GenerationResult(
            answer=f"{question}: {contexts[0].filename} page {contexts[0].page}",
            status="generated",
            provider=self.name,
            model=self.model_name,
            generation_latency_ms=2.0,
            initialization_ms=self.initialization_ms,
        )


class FailingProvider(MockProvider):
    def generate(self, question, contexts):
        del question, contexts
        raise RuntimeError("provider unavailable")


class GenerationTests(unittest.TestCase):
    def setUp(self):
        self.context = GenerationContext("Evidence text", "paper.pdf", 3, 1)

    def test_context_preserves_source_metadata(self):
        self.assertIn("paper.pdf, page 3", self.context.render())

    def test_mock_provider_returns_structured_generation(self):
        result = MockProvider().generate("What?", [self.context])
        self.assertEqual(result.status, "generated")
        self.assertIn("paper.pdf page 3", result.answer)

    def test_fallback_is_explicit(self):
        result = ExtractiveFallbackProvider().generate("What?", [self.context])
        self.assertEqual(result.status, "fallback")
        self.assertEqual(result.provider, "extractive_context_fallback")
        self.assertEqual(result.answer, "Evidence text")

    def test_provider_failure_can_be_detected_for_fallback(self):
        result = generate_with_fallback(FailingProvider(), "What?", [self.context])
        self.assertEqual(result.status, "fallback_after_generation_error")
        self.assertIn("unavailable", result.error or "")
        self.assertEqual(result.answer, "Evidence text")


if __name__ == "__main__":
    unittest.main()
