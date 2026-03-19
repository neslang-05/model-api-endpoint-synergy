from fastapi import FastAPI, UploadFile, File, HTTPException
from app.model import get_model, get_prediction
import os

app = FastAPI(title="Paddy Seed Quality Classifier API", description="FastAPI service for predicting Paddy Seed Quality")

# Automatically detect model path assuming app is run from project root
MODEL_PATH = os.environ.get('MODEL_PATH', os.path.join(os.path.dirname(__file__), '..', 'model', 'paddy_seed_model_final.pth'))

# Ensure model exists before loading
if not os.path.exists(MODEL_PATH):
    print(f"CRITICAL: Model file not found at {MODEL_PATH}")
    # Fallback to absolute container path if relative fails
    MODEL_PATH = "/workspace/model/paddy_seed_model_final.pth"

print(f"Loading model from: {MODEL_PATH}")
model = get_model(MODEL_PATH)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Paddy Seed Classifier API. Use /predict to classify seeds."}

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image.")
    
    try:
        image_bytes = await file.read()
        prediction, confidence = get_prediction(model, image_bytes)
        
        return {
            "filename": file.filename, 
            "prediction": prediction,
            "confidence": float(confidence)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
