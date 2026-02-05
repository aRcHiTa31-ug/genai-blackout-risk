# ===================================
# backend/main.py
# ===================================

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ---- ML imports ----
from ml_code.ml import rf, X
import pandas as pd

# ---- GenAI import ----
from genai.genaiproject import explain_blackout_risk

# ---- FastAPI imports ----
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

# ===================================
# FastAPI App
# ===================================
app = FastAPI(
    title="Blackout Forecaster API",
    description="AI-powered power outage explanation system",
    version="1.0"
)

# ===================================
# Helper Function
# ===================================
def get_severity(probability: float) -> str:
    if probability >= 0.75:
        return "HIGH"
    elif probability >= 0.4:
        return "MEDIUM"
    else:
        return "LOW"

# ===================================
# Input Schema
# ===================================
class BlackoutRequest(BaseModel):
    Ia: float
    Ib: float
    Ic: float
    Va: float
    Vb: float
    Vc: float
    location: str = "Sector 21, Delhi"
    factors: List[str] = ["Peak electricity demand", "Aging transformers", "Extreme heat conditions"]

# ===================================
# Output Schema
# ===================================
class BlackoutResponse(BaseModel):
    location: str
    risk_level: str
    probability: float
    severity: str
    explanation: str

# ===================================
# Predict + Explain Endpoint
# ===================================
@app.post("/predict", response_model=BlackoutResponse)
def predict_blackout(data: BlackoutRequest):
    """
    1️⃣ Use ML (Random Forest) to predict blackout probability
    2️⃣ Determine severity
    3️⃣ Generate GenAI explanation
    """

    # ---- Prepare ML input ----
    input_data = pd.DataFrame([[
        data.Ia, data.Ib, data.Ic, data.Va, data.Vb, data.Vc
    ]], columns=X.columns)

    # ---- ML Prediction ----
    probability = rf.predict_proba(input_data)[0][1]

    # ---- Determine severity ----
    severity = get_severity(probability)

    # ---- GenAI Explanation ----
    explanation = explain_blackout_risk(
        location=data.location,
        risk_level="High Risk of Power Blackout" if probability > 0.5 else "Low/Medium Risk",
        probability=probability,
        factors=data.factors
    )

    # ---- Return JSON ----
    return {
        "location": data.location,
        "risk_level": "High Risk of Power Blackout" if probability > 0.5 else "Low/Medium Risk",
        "probability": probability,
        "severity": severity,
        "explanation": explanation
    }

# ===================================
# Root Endpoint (optional health check)
# ===================================
@app.get("/")
def root():
    return {"status": "Blackout Forecaster Backend Running"}

