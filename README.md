# Genai-blackout-risk
This project predicts power blackout risk using a machine learning model and explains the results using Generative AI. It classifies risk severity as low, medium, or high and generates clear, actionable insights for authorities based on system conditions, demand, and environmental factors.

## Methodology
1) Loads CSV / structured datasets using pandas (taken from kaggle)
Handles:
Historical power outage data
Infrastructure & weather indicators
Performs basic preprocessing (cleaning, feature selection)

2️) Machine Learning Engine
Implemented in Python
Uses trained ML model to:
Predict blackout probability
Classify risk level (Low / Medium / High)
Outputs:
Probability score
Risk category
Key contributing factors

3️)  Risk Analysis Module
Converts probability into severity levels
Applies rule-based logic:
HIGH / MEDIUM / LOW severity mapping
Ensures consistency and interpretability of ML results

4️) Generative AI Explanation Module
Uses Google Gemini API (google-genai)
Transforms technical ML outputs into:
Clear explanations
Impact assessment
Preventive recommendations
Includes fallback logic for reliability if API fails

5️) Backend API Layer
Built using FastAPI
Exposes endpoints to:
Accept input parameters (location, conditions)
Trigger ML prediction
Generate AI explanations
Returns structured JSON response

6️)  Error Handling & Rate Control
Exception handling for:
API failures
Invalid inputs
time.sleep() used to manage free-tier rate limits ( can add)

## Backend Flow
Input → Data Processing → ML Prediction → Severity Mapping → GenAI Explanation → API Response
