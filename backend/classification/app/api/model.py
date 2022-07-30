from pydantic import BaseModel
from typing import Optional, List
from enum import Enum


# Model for recieveing input
class Input(BaseModel):
    model: str = None
    text: str


# Model for classification response
class ClassResponse(BaseModel):
    category: str
    confidence: float
