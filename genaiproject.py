# genai/genai-project.py


from google import genai  # Updated to the 2026 unified library

# =========================
# LOAD API KEY SAFELY
# =========================

# Using the key you provided directly (ensure this is kept private in production!)
api_key = "AIzaSyBhd-HXkQyex_45ciMAQ_qM6uzEH2qtHUo"

# =========================
# INITIALIZATION
# =========================
# The new SDK uses a Client-based approach
client = genai.Client(api_key=api_key)

# =========================
# SEVERITY MAPPING
# =========================
def get_severity(probability: float) -> str:
    if probability >= 0.75:
        return "HIGH"
    elif probability >= 0.4:
        return "MEDIUM"
    else:
        return "LOW"

# =========================
# FALLBACK EXPLANATION
# =========================
def fallback_explanation(location, risk_level, severity):
    return f"""
⚠️ AI Explanation (Fallback Mode)

Location: {location}
Risk Level: {risk_level}
Severity: {severity}

This area shows indicators of potential power disruption based on system conditions.
Authorities should monitor load demand, inspect infrastructure, and prepare contingency plans.

(This explanation was generated without GenAI due to temporary unavailability.)
"""

# =========================
# CORE GENAI FUNCTION
# =========================
def explain_blackout_risk(
    location: str,
    risk_level: str,
    probability: float,
    factors: list
) -> str:
    """
    Generates explainable AI insights for power blackout risk using google-genai.
    """
    severity = get_severity(probability)

    prompt = f"""
You are an expert AI system assisting power grid operators and government authorities.

A machine learning model has predicted a power outage risk. Your job is to explain it clearly and responsibly.

Details:
Location: {location}
Predicted Risk: {risk_level}
Severity Level: {severity}
Confidence Score: {probability * 100:.1f}%
Key Contributing Factors: {", ".join(factors)}

Generate a structured explanation with:
1. Plain-language summary of the risk
2. Main reasons contributing to this risk
3. Potential impact if ignored
4. Clear and actionable preventive recommendations

Keep the explanation professional, concise, and suitable for decision-makers.
"""

    try:
        # Using the new client.models.generate_content method
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt
        )
        # response.text is directly accessible in the new SDK
        return response.text

    except Exception as e:
        print(f"DEBUG: API Error: {e}")
        return fallback_explanation(location, risk_level, severity)


# =========================
# LOCAL TESTING
# =========================
if __name__ == "__main__":
    result = explain_blackout_risk(
        location="Sector 21, Delhi",
        risk_level="High Risk of Power Blackout",
        probability=0.87,
        factors=[
            "Peak electricity demand",
            "Aging transformers",
            "Extreme heat conditions",
            "Heavy storm incoming",
            "Short circuit in Main line"
        ]
    )

    print("===== AI GENERATED OUTPUT =====\n")
    print(result)