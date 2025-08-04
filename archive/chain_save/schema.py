from pydantic import BaseModel, Field
from typing import List, Literal, Optional

# class ParsedQuestion(BaseModel):
#     question_type: Literal["overall_summary", "metric_summary"]
#     products: List[str] = Field(default=[])  # empty = no filter
#     metrics: List[str] = Field(default=[])
#     earliest_date: Optional[str] = Field(default="latest")
#     latest_date: Optional[str] = Field(default="latest")

class TimeSpan(BaseModel):
    """
    Represents a time span with a start and end (YYYY-MM), and the original text from which it was parsed.
    """
    start: str  # e.g., "2024-01"
    end: str    # e.g., "2024-03"
    original_text: str

class ParsedQuestion(BaseModel):
    """
    Represents a parsed question with type, product/metric filters, and time spans.
    """
    question_type: Literal["metric_summary", "overall_summary"]
    products: List[str] = []
    metrics: List[str] = []
    time_spans: List[TimeSpan] = []