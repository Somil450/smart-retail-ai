from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import vision, nlp, chatbot
from app.schemas import DashboardStatsResponse
import random

app = FastAPI(
    title="Smart Retail & Customer Intelligence Platform API",
    description="API for Face Recognition, Product Classification, Sentiment Analysis, and Chatbot Support.",
    version="1.0.0"
)

# Enable CORS for frontend applications
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(vision.router)
app.include_router(nlp.router)
app.include_router(chatbot.router)

@app.get("/dashboard/stats", response_model=DashboardStatsResponse, tags=["Dashboard"])
async def get_dashboard_stats():
    """
    Returns aggregate stats for the dashboard.
    In a real app, this would query a database.
    """
    total_visits = random.randint(100, 500)
    returning = random.randint(20, total_visits)
    
    return DashboardStatsResponse(
        total_visits=total_visits,
        returning_customers=returning,
        new_customers=total_visits - returning,
        sentiment_breakdown={
            "positive": random.randint(50, 200),
            "neutral": random.randint(20, 100),
            "negative": random.randint(5, 50)
        }
    )

@app.get("/", tags=["Health"])
async def root():
    return {"message": "Welcome to Smart Retail API. Check /docs for API documentation."}
