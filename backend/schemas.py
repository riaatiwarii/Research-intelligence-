from pydantic import BaseModel
from typing import List, Optional


# -------- QA --------
class QARequest(BaseModel):
    question: str


class QAResponse(BaseModel):
    answer: str
    citations: List[str]


# -------- Search --------
class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 10


class SearchResult(BaseModel):
    text: str
    paper_id: str
    section: str
    score: float


# -------- PPT --------
class PPTRequest(BaseModel):
    topic: str
