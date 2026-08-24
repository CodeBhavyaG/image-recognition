from pydantic import BaseModel
from typing import List

class ImageUnderstanding(BaseModel):
    subject: str
    category: str
    attributes: List[str]
    caption: str
    confidence: float
