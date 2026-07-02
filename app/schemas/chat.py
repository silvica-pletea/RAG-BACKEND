from pydantic import BaseModel
from app.config.constants import SearchType

class ChatRequest(BaseModel):
    question: str
    type: SearchType = SearchType.SEMANTIC

class ChatResponse(BaseModel):
    answer: str
