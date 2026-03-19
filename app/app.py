from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
from threading import Lock

app = FastAPI(title="Paddy Seed Quality Classifier API", description="FastAPI service for predicting Paddy Seed Quality")


def _get_allowed_origins():
    configured_origins = os.environ.get('CORS_ALLOW_ORIGINS', '*').strip()
    if configured_origins == '*':
        return ['*']

    return [origin.strip() for origin in configured_origins.split(',') if origin.strip()]


app.add_middleware(
    CORSMiddleware,
    allow_origins=_get_allowed_origins(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_model_lock = Lock()
_model_module = None


def _resolve_model_path():
    configured_path = os.environ.get('MODEL_PATH')
    candidates = []

    if configured_path:
        candidates.append(configured_path)

    candidates.extend([
        os.path.join(os.path.dirname(__file__), '..', 'model', 'paddy_seed_model_final.pth'),
        '/workspace/model/paddy_seed_model_final.pth'
    ])

    for candidate in candidates:
        normalized = os.path.abspath(candidate)
        if os.path.exists(normalized):
            return normalized

    return os.path.abspath(candidates[0])


def _get_model_module():
    global _model_module
    if _model_module is None:
        from app import model as model_module
        _model_module = model_module
    return _model_module


def _get_or_load_model():
    model = getattr(app.state, 'model', None)
    if model is not None:
        return model

    with _model_lock:
        model = getattr(app.state, 'model', None)
        if model is not None:
            return model

        model_path = _resolve_model_path()
        print(f"Loading model from: {model_path}")

        model_module = _get_model_module()
        app.state.model = model_module.get_model(model_path)
        app.state.model_path = model_path

        return app.state.model

@app.get("/")
def read_root():
    return {
        "message": "Welcome to the Paddy Seed Classifier API. Use /predict to classify seeds.",
        "model_loaded": getattr(app.state, 'model', None) is not None
    }

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image.")
    
    try:
        model = _get_or_load_model()
        model_module = _get_model_module()

        image_bytes = await file.read()
        prediction, confidence = model_module.get_prediction(model, image_bytes)
        
        return {
            "filename": file.filename, 
            "prediction": prediction,
            "confidence": float(confidence)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
