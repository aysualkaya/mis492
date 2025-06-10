# 🌱 AgroMind - Smart Crop Recommendation API

AgroMind is an intelligent crop recommendation system designed to support farmers and agricultural planners by suggesting suitable crops based on real-time **soil** and **climate** data. It combines machine learning, soil science, and meteorological analysis in a seamless FastAPI backend integrated with a Flutter mobile frontend.

---

## 🚀 Key Features
- Predicts the **top 3 most suitable crops** for a given location and planting month.
- Uses real-time **SoilGrids** and **Google Earth Engine** data.
- Gracefully handles missing soil or climate data with fallback logic.
- Modular, scalable backend ready for API deployment.

---

## ⚙️ Technologies Used
- **Python** (scikit-learn, xgboost, imbalanced-learn, FastAPI, joblib)
- **Machine Learning**:
  - Ensemble Voting Classifier (RandomForest + XGBoost + LogisticRegression)
  - Borderline-SMOTE for class balancing
- **APIs**:
  - SoilGrids REST API (ISDA)
  - Google Earth Engine for 24-year climate averages
- **Model Metrics**:
  - Weighted F1-Score (CV=5)

---

## 📦 Final Model & Files in Use

✅ **Currently Used**:
- `models/ensemble_model.pkl` → Best-performing classifier
- `models/scaler.pkl`, `models/label_encoder.pkl` → For preprocessing
- `train_model/train.py` → Final training script with Borderline-SMOTE + Stratified CV
- `main.py` → FastAPI backend serving predictions
- `soil_utils.py`, `climate_utils.py` → Dynamic real-time feature fetching

🗑️ **Deprecated / No Longer Used**:
- `final_model.pkl` → Previously used single-model version, now replaced
- `compare_models.py` → Used during model evaluation phase, now obsolete
- `datasets_all.xlsx/csv` → Raw data used during development, not loaded by API

---

## 🧪 API Usage

### 🔎 `/predict` (POST)
Predict crops for a location and month.
```json
{
  "latitude": 36.9081,
  "longitude": 30.6956,
  "month": 3
}
```
Response:
```json
{
  "prediction": "Wheat",
  "confidence": 0.781,
  "top_3_predictions": [
    {"crop": "Wheat", "probability": 0.781},
    {"crop": "Barley", "probability": 0.142},
    {"crop": "Potato", "probability": 0.044}
  ],
  "soil_type": "Loamy",
  "location_info": { ... }
}
```

### 🧬 `/health` (GET)
Check if the API is running.
```json
{ "status": "ok" }
```

---

## 🧠 How It Works (Behind the Scenes)
1. Takes user coordinates + month.
2. Fetches **climate data** from GEE (2000–2024 weighted avg).
3. Retrieves **soil data** from SoilGrids or defaults if unavailable.
4. Derives soil type via **texture triangle** if raw % values exist.
5. Scales input and runs **ensemble model**.
6. Returns top-3 crops with confidence + location and data metadata.

---

## 📌 Notes
- Designed to work even when partial data is missing (graceful degradation).
- All prediction input is dynamically fetched (no hardcoded data).
- Realistic integration for mobile apps (Flutter frontend ready).

---

## 👤 Developer
Aysu Alkaya  
Boğaziçi University – MIS Senior Project  

For questions or contributions, feel free to open an issue or contact the maintainer.
