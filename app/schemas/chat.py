from pydantic import BaseModel
from app.config.constants import SearchType

class ChatRequest(BaseModel):
    question: str
    type: SearchType = SearchType.HYBRID

class ChatResponse(BaseModel):
    answer: str
