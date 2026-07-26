from pydantic import BaseModel
from typing import Optional

class SentimentRequest(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float

class ChatbotRequest(BaseModel):
    message: str

class ChatbotResponse(BaseModel):
    message: str
    intent: str
    reply: str

class FaceRecognitionResponse(BaseModel):
    customer_id: str
    status: str
    message: str

class ProductClassificationResponse(BaseModel):
    category: str
    confidence: float

class DashboardStatsResponse(BaseModel):
    total_visits: int
    returning_customers: int
    new_customers: int
    sentiment_breakdown: dict
