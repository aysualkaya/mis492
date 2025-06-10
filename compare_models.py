import pandas as pd 
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# 1. Veriyi yükle
df = pd.read_excel("datasets_all_cleaned.xlsx")
# Eksik değerleri kontrol et
print("Eksik veri var mı?\n", df.isnull().sum())

# Tüm satırlarda eksik veri varsa onları sil
df = df.dropna()
  # Dosya adını uygun şekilde güncelle

# 2. One-hot encoded soil columns'ları otomatik olarak bul
soil_cols = [col for col in df.columns if col.startswith("SoilType_")]

# 3. Özellikler (X) ve hedef (y)
X = df[['Ph', 'N', 'P', 'K', 'Temperature', 'Humidity'] + soil_cols]
y = df['Label']  # veya 'Crop' sütununun adı neyse onu yaz

# 4. Label encode hedef (y)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 5. Train-test böl
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)

# 6. Model seti
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
    "Decision Tree": DecisionTreeClassifier(class_weight='balanced'),
    "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced'),
    "SVM": SVC(kernel='rbf', class_weight='balanced'),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
}

# 7. Model eğitim ve değerlendirme
results = []

for name, model in models.items():
    print(f"\n🔄 Training: {name}")
    start_train = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_train

    start_pred = time.time()
    y_pred = model.predict(X_test)
    pred_time = time.time() - start_pred

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')

    # Azınlık sınıf örneği: pea ve bean class index'leri varsa
    try:
        minor_f1 = f1_score(y_test, y_pred, average=None)[[5, 6]].mean()
    except:
        minor_f1 = "N/A"

    results.append({
        'Model': name,
        'Accuracy': round(accuracy, 4),
        'Macro F1': round(macro_f1, 4),
        'Minor Crop F1': round(minor_f1, 4) if minor_f1 != "N/A" else "N/A",
        'Train Time (s)': round(train_time, 2),
        'Predict Time (s)': round(pred_time, 4)
    })

# 8. Sonuçları göster ve kaydet
results_df = pd.DataFrame(results)
print("\n📊 Model Karşılaştırma Tablosu:")
print(results_df.sort_values(by="Macro F1", ascending=False))

results_df.to_excel("model_comparison_results.xlsx", index=False)
