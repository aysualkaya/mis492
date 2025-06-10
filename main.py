from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from train_model.climate_utils import get_weighted_climate, get_location_details
from train_model.soil_utils import (
    get_partial_soil_data,
    encode_soil_type,
    SOIL_TYPE_ENCODER,
    map_texture_to_soil_type
)

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AgroMind API",
    description="Crop recommendation based on soil and climate data.",
    version="1.0.0"
)

# Load model once (updated to ensemble)
model = joblib.load("models/ensemble_model.pkl")
scaler = joblib.load("models/scaler.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

class LocationRequest(BaseModel):
    latitude: float
    longitude: float
    month: int

    class Config:
        json_schema_extra = {
            "example": {
                "latitude": 41.0082,
                "longitude": 28.9784,
                "month": 6
            }
        }

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "AgroMind API is running"}

@app.post("/predict")
def predict_crop(request: LocationRequest):
    try:
        input_vector, soil_label, location_info = prepare_input_vector(
            request.latitude, request.longitude, request.month
        )

        # Scale input vector
        input_vector_scaled = scaler.transform([input_vector])

        # Predict
        proba = model.predict_proba(input_vector_scaled)[0]
        top_indices = np.argsort(proba)[::-1][:3]
        top_crops = [
            {"crop": label_encoder.inverse_transform([i])[0], "probability": round(float(proba[i]), 3)}
            for i in top_indices
        ]

        return {
            "prediction": top_crops[0]["crop"],
            "confidence": top_crops[0]["probability"],
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
    logger.info(f"Processing coordinates: ({lat}, {lon}) for month {month}")

    if not (1 <= month <= 12):
        raise ValueError("Month must be between 1 and 12.")
    if not (-90 <= lat <= 90):
        raise ValueError("Latitude must be between -90 and 90.")
    if not (-180 <= lon <= 180):
        raise ValueError("Longitude must be between -180 and 180.")

    try:
        # Climate
        try:
            climate_data = get_weighted_climate(lat, lon, month)
            logger.info("✅ Climate data retrieved successfully")
        except Exception:
            climate_data = {"temperature": 20.0, "humidity": 70.0}
            logger.warning("⚠️ Fallback to default climate data.")

        # Soil
        soil_data = None
        soil_data_source = "default"
        try:
            soil_data = get_partial_soil_data(lat, lon)
            if soil_data is not None:
                soil_data_source = "SoilGrids API"
                logger.info("✅ SoilGrids data retrieved successfully")
        except Exception as e:
            logger.warning(f"❌ SoilGrids unavailable: {e}")

        if soil_data is None and climate_data["temperature"] < -5:
            raise ValueError("This region is unplantable due to extreme climate and missing soil data.")

        if soil_data is None:
            soil_data = {
                "ph": 6.5,
                "n": 15.0,
                "p": 25.0,
                "k": 180.0,
                "clay_percent": 20.0,
                "sand_percent": 40.0,
                "silt_percent": 40.0
            }
            soil_data_source = "default"
            logger.info("Using default soil data values")

        if 'clay_percent' in soil_data and 'sand_percent' in soil_data and 'silt_percent' in soil_data:
            soil_label = map_texture_to_soil_type(
                soil_data['clay_percent'], 
                soil_data['sand_percent'], 
                soil_data['silt_percent']
            )
        elif 'clay' in soil_data and 'sand' in soil_data and 'silt' in soil_data:
            soil_label = map_texture_to_soil_type(
                soil_data['clay'], 
                soil_data['sand'], 
                soil_data['silt']
            )
        else:
            soil_label = "Loamy"

        encoded_soil = encode_soil_type(soil_label, SOIL_TYPE_ENCODER)

        input_vector = [
            encoded_soil,
            soil_data.get("ph", 6.5),
            soil_data.get("k", 180.0),
            soil_data.get("p", 25.0),
            soil_data.get("n", 15.0),
            climate_data["temperature"],
            climate_data["humidity"]
        ]

        if any(x is None for x in input_vector):
            raise ValueError("Some required soil or climate data is missing.")

        try:
            location_str = get_location_details(lat, lon)
        except:
            location_str = f"({lat:.4f}, {lon:.4f})"

        location_info = {
            "coordinates": {"lat": lat, "lon": lon},
            "month": month,
            "location_name": location_str,
            "soil_data": soil_data,
            "soil_data_source": soil_data_source,
            "climate_data": climate_data,
            "data_quality": {
                "soil_source": soil_data_source,
                "climate_source": "API" if "temperature" in climate_data else "default"
            }
        }

        logger.info("✅ Input vector prepared successfully.")
        return input_vector, soil_label, location_info

    except Exception as e:
        logger.error(f"❌ Error preparing input vector: {e}")
        raise

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
