from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers.predict import router as predict_router
from app.routers.schedule import router as schedule_router

app = FastAPI(
    title="LCK win rate predictor",
    description="LCK win rate predictor based on machine learning",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict_router)
app.include_router(schedule_router)

@app.get("/health")
def health_check():
    return {"status": "healthy"}