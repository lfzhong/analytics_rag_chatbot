from typing import List, Literal
from pydantic import BaseModel, RootModel

# Individual time span
class TimeSpan(BaseModel):
    start: str
    end: str
    original_text: str

# A single query object
class QueryObject(BaseModel):
    product: str
    metric: str
    time_spans: List[TimeSpan]
    question_type: Literal["metric_summary", "overall_summary"] = "overall_summary"

# Root model for a list of QueryObject (for robust LLM output parsing)
class QueryObjectList(RootModel[List[QueryObject]]):
    pass


