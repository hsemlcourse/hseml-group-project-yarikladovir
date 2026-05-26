from fastapi import FastAPI

from src.predictor import predict_text
from src.schemas import PredictionRequest, PredictionResponse


app = FastAPI(
    title="Fake Job Posting Detection API",
    description="API for detecting potentially fraudulent job postings using a TF-IDF + Linear SVC model.",
    version="1.0.0",
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    result = predict_text(request.text)
    return PredictionResponse(**result)
