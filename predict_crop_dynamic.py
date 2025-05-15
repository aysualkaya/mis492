import sys
import os
import joblib
from train_model.climate_utils import get_location, get_location_details, prepare_input_vector, SOIL_TYPE_ENCODER


def main():
    print("\n🚀 Starting AgroMind Dynamic Crop Prediction...")

    # 1. Kullanıcı konumu alımı
    try:
        lat, lon = get_location()
        if lat == 0.0 and lon == 0.0:
            raise ValueError("⚠️ Location not found. Please enable GPS or enter a valid location.")
        location_name = get_location_details(lat, lon)
        print(f"📍 Location Detected: Latitude={lat:.4f}, Longitude={lon:.4f} ({location_name})")
    except Exception as e:
        print(f"❌ Failed to get location: {e}")
        sys.exit(1)

    # 2. Girdi vektörü hazırlama
    try:
        input_vector = prepare_input_vector(lat, lon)
    except Exception as e:
        print(f"⚠️ Failed to prepare input vector: {e}")
        sys.exit(1)

    # 3. Modeli yükleme
    model_path = os.path.join("models", "final_model.pkl")
    encoder_path = os.path.join("models", "soil_type_encoder.pkl")

    if not os.path.exists(model_path) or not os.path.exists(encoder_path):
        print(f"❌ Model or encoder file not found at: {model_path} or {encoder_path}")
        sys.exit(1)

    try:
        model = joblib.load(model_path)
        soil_encoder = joblib.load(encoder_path)
    except Exception as e:
        print(f"❌ Failed to load model or encoder: {e}")
        sys.exit(1)

    # 4. Tahmin yapma
    try:
        prediction = model.predict([input_vector])[0]
        predicted_soil_type = soil_encoder.inverse_transform([input_vector[0]])[0]
        print(f"🌱 Recommended Crop: {prediction} (Soil Type: {predicted_soil_type})")
    except Exception as e:
        print(f"⚠️ Prediction failed: {e}")


if __name__ == "__main__":
    main()
