import pandas as pd
import numpy as np
import os
import joblib
import logging
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import BorderlineSMOTE

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    try:
        logger.info("Veri yukleniyor...")
        df = pd.read_excel("datasets_all.xlsx")

        # LABEL TEMIZLEME (case sensitive sorunlari duzelt)
        label_mapping = {'rice': 'Rice', 'maize': 'Maize', 'cotton': 'Cotton'}
        df['label'] = df['label'].replace(label_mapping)

        df.dropna(subset=["soil_type", "ph", "k", "p", "n", "temperature", "humidity", "label"], inplace=True)

        logger.info("Soil type encoding...")
        soil_encoder = joblib.load("models/soil_type_encoder.pkl")
        df["soil_type"] = df["soil_type"].apply(lambda x: x if x in soil_encoder.classes_ else "Unknown")
        df["soil_type"] = soil_encoder.transform(df["soil_type"])

        features = ["soil_type", "ph", "k", "p", "n", "temperature", "humidity"]
        X = df[features]
        y = df["label"]

        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
        )

        logger.info("Borderline-SMOTE uygulanıyor...")
        smote = BorderlineSMOTE(sampling_strategy='auto', k_neighbors=3, random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

        logger.info("Ozellik olceklendirme...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_res)
        X_test_scaled = scaler.transform(X_test)

        rf = RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_split=20,
            min_samples_leaf=10,
            class_weight='balanced_subsample',
            random_state=42,
            n_jobs=-1
        )

        xgb = XGBClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='multi:softprob',
            num_class=len(label_encoder.classes_),
            eval_metric='mlogloss',
            n_jobs=-1,
            random_state=42
        )

        logger.info("Egitim baslatiliyor (CV=5, f1_weighted)...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        ensemble = VotingClassifier(estimators=[('rf', rf), ('xgb', xgb)], voting='soft', n_jobs=-1)
        ensemble.fit(X_train_scaled, y_train_res)

        y_pred = ensemble.predict(X_test_scaled)
        y_test_original = label_encoder.inverse_transform(y_test)
        y_pred_original = label_encoder.inverse_transform(y_pred)

        logger.info(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(classification_report(y_test_original, y_pred_original))

        os.makedirs("models", exist_ok=True)
        joblib.dump(ensemble, "models/ensemble_model.pkl")
        joblib.dump(scaler, "models/scaler.pkl")
        joblib.dump(label_encoder, "models/label_encoder.pkl")

        metadata = {
            'features': features,
            'n_classes': len(label_encoder.classes_),
            'classes': label_encoder.classes_.tolist(),
            'model_type': type(ensemble).__name__
        }
        joblib.dump(metadata, "models/metadata.pkl")
        logger.info("Model, scaler, encoder ve metadata kaydedildi.")

    except Exception as e:
        logger.error(f"Hata olustu: {e}")
        raise

if __name__ == "__main__":
    main()
