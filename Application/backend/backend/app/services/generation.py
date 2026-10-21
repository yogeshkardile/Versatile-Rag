from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Protocol, Sequence


@dataclass(frozen=True)
class GenerationContext:
    text: str
    filename: str
    page: int
    rank: int

    def render(self) -> str:
        return f"[Source {self.rank}: {self.filename}, page {self.page}]\n{self.text}"


@dataclass(frozen=True)
class GenerationResult:
    answer: str
    status: str
    provider: str
    model: str | None
    generation_latency_ms: float
    initialization_ms: float | None = None
    error: str | None = None


class GenerationProvider(Protocol):
    name: str
    model_name: str | None
    initialization_ms: float | None

    def generate(self, question: str, contexts: Sequence[GenerationContext]) -> GenerationResult: ...


@dataclass(frozen=True)
class ExtractiveFallbackProvider:
    name: str = "extractive_context_fallback"
    model_name: str | None = None
    initialization_ms: float | None = 0.0

    def generate(self, question: str, contexts: Sequence[GenerationContext]) -> GenerationResult:
        del question
        started = time.perf_counter()
        answer = contexts[0].text if contexts else "No supporting evidence was retrieved."
        return GenerationResult(
            answer=answer,
            status="fallback",
            provider=self.name,
            model=self.model_name,
            generation_latency_ms=(time.perf_counter() - started) * 1000,
            initialization_ms=self.initialization_ms,
        )


class LocalSeq2SeqGenerationProvider:
    name = "local_transformers"

    def __init__(self, model_name: str) -> None:
        started = time.perf_counter()
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, local_files_only=True)
        self.model.eval()
        self.initialization_ms = (time.perf_counter() - started) * 1000

    def generate(self, question: str, contexts: Sequence[GenerationContext]) -> GenerationResult:
        import torch

        started = time.perf_counter()
        evidence = "\n\n".join(context.render() for context in contexts)
        prompt = (
            "Answer the question using only the provided evidence. "
            "If the evidence is insufficient, say that it is insufficient. "
            "Do not invent facts or citations. Keep the answer concise.\n\n"
            f"Question: {question}\n\nEvidence:\n{evidence}\n\nAnswer:"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=128, do_sample=False)
        answer = self.tokenizer.decode(output[0], skip_special_tokens=True).strip()
        return GenerationResult(
            answer=answer or "The model returned no answer from the provided evidence.",
            status="generated",
            provider=self.name,
            model=self.model_name,
            generation_latency_ms=(time.perf_counter() - started) * 1000,
            initialization_ms=self.initialization_ms,
        )


def generate_with_fallback(
    provider: GenerationProvider,
    question: str,
    contexts: Sequence[GenerationContext],
    error: str | None = None,
) -> GenerationResult:
    try:
        return provider.generate(question, contexts)
    except Exception as exc:
        fallback = ExtractiveFallbackProvider().generate(question, contexts)
        return GenerationResult(
            answer=fallback.answer,
            status="fallback_after_generation_error",
            provider=fallback.provider,
            model=provider.model_name,
            generation_latency_ms=fallback.generation_latency_ms,
            initialization_ms=provider.initialization_ms,
            error=error or str(exc),
        )


def configured_provider() -> GenerationProvider:
    provider = os.getenv("VERSATILE_RAG_GENERATION_PROVIDER", "fallback").strip().lower()
    if provider == "local_transformers":
        return LocalSeq2SeqGenerationProvider(
            os.getenv("VERSATILE_RAG_GENERATION_MODEL", "google/flan-t5-base")
        )
    return ExtractiveFallbackProvider()
