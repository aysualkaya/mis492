from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from climate_utils import get_weighted_climate
from train_model.soil_utils import get_soil_data_with_fallback, map_texture_to_soil_type, encode_soil_type, SOIL_TYPE_ENCODER

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AgroMind API",
    description="Crop recommendation based on soil and climate data.",
    version="1.0.0"
)

# Load model
model = joblib.load("final_model.pkl")

class LocationRequest(BaseModel):
    latitude: float
    longitude: float
    month: int

    class Config:
        schema_extra = {
            "example": {
                "latitude": 41.0082,
                "longitude": 28.9784,
                "month": 6
            }
        }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "ok", "message": "AgroMind API is running"}

@app.post("/predict")
def predict_crop(request: LocationRequest):
    """
    Predict optimal crops for given location and month.
    
    Args:
        request: LocationRequest containing latitude, longitude, and month
        
    Returns:
        Crop predictions with probabilities and location info
    """
    try:
        input_vector, soil_label, location_info = prepare_input_vector(
            request.latitude, request.longitude, request.month
        )

        # Get predictions
        proba = model.predict_proba([input_vector])[0]
        top_indices = np.argsort(proba)[::-1][:3]
        top_crops = [
            {"crop": model.classes_[i], "probability": round(float(proba[i]), 3)}
            for i in top_indices
        ]

        return {
            "prediction": model.classes_[top_indices[0]],
            "confidence": round(float(proba[top_indices[0]]), 3),
            "top_3_predictions": top_crops,
            "soil_type": soil_label,
            "location_info": location_info,
            "status": "success"
        }

    except ValueError as ve:
        logger.warning(f"Validation error: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.exception("Prediction error occurred")
        raise HTTPException(status_code=500, detail="Internal server error occurred during prediction")

def prepare_input_vector(lat, lon, month):
    """
    Prepare input vector for model prediction.
    
    Args:
        lat (float): Latitude
        lon (float): Longitude
        month (int): Target month (1-12)
        
    Returns:
        tuple: (input_vector, soil_label, location_info)
    """
    logger.info(f"Processing coordinates: ({lat}, {lon}) for month {month}")

    # Validate month
    if not (1 <= month <= 12):
        raise ValueError("Month must be between 1 and 12.")

    # Validate coordinates
    if not (-90 <= lat <= 90):
        raise ValueError("Latitude must be between -90 and 90.")
    if not (-180 <= lon <= 180):
        raise ValueError("Longitude must be between -180 and 180.")

    try:
        # Get climate data
        climate_data = get_weighted_climate(lat, lon, month)
        logger.info(f"Climate data: {climate_data}")
        
        # Get soil data with fallback
        soil_data = get_soil_data_with_fallback(lat, lon)
        
        # Validate if the location is unplantable
        if not soil_data and climate_data["temperature"] < -5:
            raise ValueError("🌍 This location appears to be water, frozen, or agriculturally unsuitable.")

        if not soil_data:
            raise ValueError("❌ Soil data unavailable for the given location and nearby areas.")

        # Process soil data
        soil_label = map_texture_to_soil_type(
            soil_data["clay"], 
            soil_data["sand"], 
            soil_data["silt"]
        )
        encoded_soil = encode_soil_type(soil_label, SOIL_TYPE_ENCODER)

        # Create input vector (same order as training data)
        input_vector = [
            encoded_soil,
            soil_data.get("ph", 0.0),
            soil_data.get("k", 0.0),
            soil_data.get("p", 0.0),
            soil_data.get("n", 0.0),
            climate_data["temperature"],
            climate_data["humidity"]
        ]

        # Validate input vector
        if any(x is None for x in input_vector):
            raise ValueError("Some required soil or climate data is missing.")

        location_info = {
            "coordinates": {"lat": lat, "lon": lon},
            "month": month,
            "soil_data": {
                "ph": soil_data.get("ph"),
                "nitrogen": soil_data.get("n"),
                "phosphorus": soil_data.get("p"),
                "potassium": soil_data.get("k"),
                "clay": soil_data.get("clay"),
                "sand": soil_data.get("sand"),
                "silt": soil_data.get("silt")
            },
            "climate_data": climate_data
        }

        logger.info(f"Input vector prepared successfully: {input_vector}")
        return input_vector, soil_label, location_info

    except Exception as e:
        logger.error(f"Error preparing input vector: {e}")
        raise

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")