"""Producers turn model output into *validated* ``ImageUnderstanding`` objects.

ClassifierProducer: deterministic enrichment of a supervised classifier's
(top class + softmax confidence) into the structured JSON schema.
VisionModelProducer: a stub for the capstone Phase-2 vision-model path
(Gemini Flash / Ollama) that emits the same schema, so callers never depend on
which producer is active.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import class_metadata
from schemas import ImageUnderstanding, LOW_CONFIDENCE


class Producer(ABC):
    @abstractmethod
    def produce(self, subject: str, confidence: float) -> ImageUnderstanding:
        """Build a validated ImageUnderstanding for a predicted subject."""


class ClassifierProducer(Producer):
    """Deterministic producer built on a supervised classifier's output."""

    def __init__(self, extra_attributes: Optional[Dict[str, List[str]]] = None):
        self._extra = extra_attributes or {}

    def produce(self, subject: str, confidence: float) -> ImageUnderstanding:
        attributes = class_metadata.class_attributes(subject) + self._extra.get(subject, [])
        return ImageUnderstanding(
            subject=subject,
            category="animal",
            attributes=attributes,
            caption=class_metadata.build_caption(subject),
            confidence=confidence,
        )


class VisionModelProducer(Producer):
    """STUB: real vision-model structured output (capstone Phase 2).

    When implemented, this will call a vision model (e.g. Gemini Flash free tier,
    or a local Ollama model such as LLaVA/Moondream), parse its response, and
    validate it against ``ImageUnderstanding`` before returning.
    """

    def produce(self, subject: str, confidence: float) -> ImageUnderstanding:
        raise NotImplementedError(
            "VisionModelProducer is a stub. Configure a vision model or use ClassifierProducer."
        )


def flag_low_confidence(understanding: ImageUnderstanding) -> Optional[str]:
    """Human-readable reason if confidence is low, otherwise None. Never silently accept."""
    if understanding.is_low_confidence:
        return (
            f"Low confidence ({understanding.confidence:.2f} < {LOW_CONFIDENCE:.2f}) "
            "— flag for review, do not trust."
        )
    return None