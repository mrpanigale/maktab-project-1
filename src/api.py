"""API"""

# ==============imports====================
import json
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict
from src.predict import predict


# ==============request class====================
class FraudRequest(BaseModel):
    """Exactly 30 features no more!"""

    model_config = ConfigDict(extra="forbid")

    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float


# =========API===================
app = FastAPI(title="Fraud Detection API", version="1.0.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict_endpoint(request: FraudRequest):
    """JSON in -> JSON out"""

    result_json = predict(request.model_dump_json())
    return JSONResponse(content=json.loads(result_json))
