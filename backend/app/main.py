from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.routes_airguard import router as airguard_router


app = FastAPI(
    title="AirGuard AI API",
    description="Urban air quality intelligence API for real-data forecasting, evidence tools, and Groq supervisor decisions.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(airguard_router)


@app.get("/")
def root():
    return {
        "service": "AirGuard AI",
        "status": "running",
        "docs": "/docs",
    }