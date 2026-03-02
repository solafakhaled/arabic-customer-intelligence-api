from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# 1. Define the Input (Matches your dataset's 'text' column)
class FeedbackInput(BaseModel):
    text: str

# 2. Define the Output (Matches your project requirements)
class FeedbackOutput(BaseModel):
    sentiment: str       # Positive/Negative
    confidence: float    # A score (0.0 - 1.0)

# 3. Create the Endpoint
@app.post("/predict", response_model=FeedbackOutput)
def predict_feedback(feedback: FeedbackInput):
    # Dummy logic for Day 1 (Model comes tomorrow!)
    return {
        "sentiment": "Positive", 
        "confidence": 0.99
    }

@app.get("/health")
def health_check():
    return {"status": "ok"}