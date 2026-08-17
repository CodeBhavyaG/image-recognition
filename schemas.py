"""Pydantic schema for the capstone Phase-2 structured vision output.

This is the single source of truth for what a *validated* model response looks
like. Every producer (classifier baseline today, a vision model later) must emit
this shape, and every response is validated at the boundary before it is
trusted (capstone "validate at the boundary" rule).
"""
from typing import List

from pydantic import BaseModel, Field, field_validator

LOW_CONFIDENCE = 0.6  # below this, a prediction is flagged, not accepted


class ImageUnderstanding(BaseModel):
    subject: str = Field(..., min_length=1, description="Detected subject (English label).")
    category: str = Field("animal", description="High-level category of the subject.")
    attributes: List[str] = Field(default_factory=list, description="Known attributes of the subject.")
    caption: str = Field(..., min_length=1, description="Human-readable description sentence.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence in [0, 1].")

    @property
    def is_low_confidence(self) -> bool:
        return self.confidence < LOW_CONFIDENCE

    @field_validator("subject")
    @classmethod
    def _subject_not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("subject must not be blank")
        return v

    @field_validator("caption")
    @classmethod
    def _caption_not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("caption must not be blank")
        return v