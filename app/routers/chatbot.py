from fastapi import APIRouter, HTTPException
from app.schemas import ChatbotRequest, ChatbotResponse
from app.services.chatbot_service import get_chatbot_response

router = APIRouter(prefix="/chatbot", tags=["Chatbot Support"])

@router.post("/", response_model=ChatbotResponse)
async def api_chatbot(req: ChatbotRequest):
    result = get_chatbot_response(req.message)
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
        
    return ChatbotResponse(**result)
