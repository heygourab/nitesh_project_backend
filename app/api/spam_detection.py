from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
import sys
from pathlib import Path

# Add the model directory to Python path
sys.path.append(str(Path(__file__).parent.parent))
from model.preprocessing import preprocess_text

app = FastAPI()

# Load models
base_path = Path(__file__).parent.parent
sms_model_path = base_path / "model" / "spam_classifier_sms.pkl"
email_model_path = base_path / "model" / "spam_classifier_email.pkl"

sms_model = joblib.load(sms_model_path)
email_model = joblib.load(email_model_path)

class MessageRequest(BaseModel):
    message: str

def get_spam_prediction(text: str, model, spam_type: str) -> dict:
    # Preprocess the text
    cleaned_text = preprocess_text(text)
    
    # Get prediction and probability
    is_spam = bool(model.predict([cleaned_text])[0])
    probabilities = model.predict_proba([cleaned_text])[0]
    
    # Get confidence score (probability of the predicted class)
    spam_confidence = float(probabilities[1] if is_spam else probabilities[0])
    spam_score = float(probabilities[1])  # Probability of being spam
    
    return {
        "isSpam": is_spam,
        "spamScore": spam_score,
        "spamType": spam_type,
        "spamConfidence": spam_confidence
    }

@app.post("/detect-sms-spam")
async def detect_sms_spam(request: MessageRequest):
    try:
        return get_spam_prediction(request.message, sms_model, "sms")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect-email-spam")
async def detect_email_spam(request: MessageRequest):
    try:
        return get_spam_prediction(request.message, email_model, "email")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
