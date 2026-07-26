from fastapi import APIRouter, HTTPException
from app.schemas import SentimentRequest, SentimentResponse
from app.services.nlp_service import analyze_sentiment

router = APIRouter(prefix="/nlp", tags=["Natural Language Processing"])

@router.post("/analyze-sentiment", response_model=SentimentResponse)
async def api_analyze_sentiment(req: SentimentRequest):
    result = analyze_sentiment(req.text)
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
        
    return SentimentResponse(**result)
