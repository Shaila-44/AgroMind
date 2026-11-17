from fastapi import FastAPI, UploadFile, File, HTTPException
import tensorflow as tf
import os
from utils import preprocess_image

app = FastAPI(
    title="Tomato Ripeness Classification API",
    description="API to classify tomatoes as Ripe or Unripe using a custom CNN.",
    version="1.0.0"
)

# Global variable to hold the model
MODEL = None
MODEL_PATH = "model/best_model.h5"

@app.on_event("startup")
async def load_model():
    """
    Loads the trained CNN model into memory during FastAPI startup.
    """
    global MODEL
    if os.path.exists(MODEL_PATH):
        try:
            # Load the model without compiling since we only need it for inference
            MODEL = tf.keras.models.load_model(MODEL_PATH, compile=False)
            print(f"Model loaded successfully from {MODEL_PATH}")
        except Exception as e:
            print(f"Error loading model: {e}")
    else:
        print(f"Warning: Model file not found at {MODEL_PATH}. Prediction will fail until a model is provided.")

@app.get("/")
def read_root():
    """
    Health check endpoint.
    """
    return {"message": "Welcome to the Tomato Ripeness Classification API. Use POST /predict to classify an image."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Endpoint to predict tomato ripeness from an uploaded image.
    Accepts an image file, preprocesses it, and runs inference using the CNN.
    """
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Model is not loaded. Ensure the model file exists.")
        
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")
        
    try:
        # Read file contents
        contents = await file.read()
        
        # Preprocess the image
        processed_image = preprocess_image(contents, target_size=(128, 128))
        
        # Run inference
        prediction_prob = MODEL.predict(processed_image)
        prob_ripe = float(prediction_prob[0][0])
        prob_unripe = 1.0 - prob_ripe
        
        # Determine class based on threshold 0.5 (1=Ripe, 0=Unripe)
        if prob_ripe > 0.5:
            predicted_class = "Ripe"
            confidence = prob_ripe
        else:
            predicted_class = "Unripe"
            confidence = prob_unripe
            
        return {
            "predicted_class": predicted_class,
            "confidence": round(confidence, 4),
            "probabilities": {
                "Ripe": round(prob_ripe, 4),
                "Unripe": round(prob_unripe, 4)
            },
            "demand_forecast_kg": 1250.5  # Simulated value demonstrating Day 4 LSTM integration
        }
        
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
