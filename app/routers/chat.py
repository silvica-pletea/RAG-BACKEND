from fastapi import APIRouter, HTTPException
from app.services.rag_service import RAGService
from app.schemas.chat import ChatRequest, ChatResponse

rag_service = RAGService();
router = APIRouter(prefix="/chat", tags=["chat"])

@router.post("", response_model=ChatResponse)
def chat(data: ChatRequest):
    try:
        answer = rag_service.ask(data.question, data.type)
        return ChatResponse(answer=answer)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Failed to retrieve the answer",
                "errors": str(e),
            },
        )