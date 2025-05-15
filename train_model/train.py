import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

# Veri yükleme
df = pd.read_excel("datasets_all.xlsx")

# NaN değerlerini temizleme
df.dropna(subset=["soil_type", "ph", "k", "p", "n", "temperature", "humidity"], inplace=True)

# Toprak türü temizleme (Clayey -> Clay)
df['soil_type'] = df['soil_type'].str.replace("Clayey", "Clay")

# Soil Type encoding (LabelEncoder)
SOIL_TYPES = ["Black", "Red", "Peaty", "Saline", "Sandy", "Clay", "Loamy", "Silty", "Unknown"]
label_encoder = LabelEncoder()
label_encoder.fit(SOIL_TYPES)
df['soil_type'] = label_encoder.transform(df['soil_type'])

# Özellikler ve hedef değişkeni ayırma
features = ['soil_type', 'ph', 'k', 'p', 'n', 'temperature', 'humidity']
X = df[features]
y = df['label']

# Veriyi eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Özellikleri ölçeklendirme
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Modeli oluşturma ve eğitme
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Modeli kaydetme
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/final_model.pkl")
joblib.dump(label_encoder, "models/soil_type_encoder.pkl")

# Sonuçları değerlendirme
predictions = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, predictions))
print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))
