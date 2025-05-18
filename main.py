# API
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from train_model.climate_utils import get_location_details, prepare_input_vector, fetch_soil_with_fallback
import joblib
import os

app = FastAPI()

# Model ve encoder dosyalarının yüklenmesi
MODEL_PATH = "models/final_model.pkl"
ENCODER_PATH = "models/soil_type_encoder.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
    raise FileNotFoundError(f"Model or encoder file not found at: {MODEL_PATH} or {ENCODER_PATH}")

model = joblib.load(MODEL_PATH)
soil_encoder = joblib.load(ENCODER_PATH)


class LocationData(BaseModel):
    lat: float
    lon: float
    month: int = Query(0, ge=0, le=12, description="Month for future prediction (0 for current month)")


@app.post("/predict")
async def predict(data: LocationData):
    try:
        # Lokasyon detayları
        location_name = get_location_details(data.lat, data.lon)
        if location_name == "Unknown Location":
            raise HTTPException(status_code=400, detail="Location not found or invalid coordinates.")

        # Girdi vektörü hazırlama
        try:
            input_vector = prepare_input_vector(data.lat, data.lon, data.month)
        except ValueError as e:
            raise HTTPException(status_code=500, detail=str(e))

        if len(input_vector) != 7:
            raise HTTPException(status_code=400, detail="Invalid input vector length.")

        # Tahmin yapma
        prediction = model.predict([input_vector])[0]
        predicted_soil_type = soil_encoder.inverse_transform([int(input_vector[0])])[0]

        return {
            "location": location_name,
            "latitude": data.lat,
            "longitude": data.lon,
            "soil_type": predicted_soil_type,
            "recommended_crop": prediction,
            "target_month": data.month
        }

    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=f"Model or encoder file missing: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/")
async def root():
    return {"message": "Welcome to the AgroMind Crop Recommendation API!"}


@app.get("/health")
async def health():
    return {"status": "ok", "message": "AgroMind API is running smoothly."}
