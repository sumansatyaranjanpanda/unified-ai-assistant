# ml_models/healthcare_model.py

import pandas as pd
import pickle, os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Paths
DATA_PATH = "data/healthcare/dataset.csv"

MODEL_PATH = "models/healthcare_model.pkl"
SYMPTOMS_PATH = "models/symptom_list.pkl"

# 1️⃣ Load data
df = pd.read_csv(DATA_PATH)

# 2️⃣ Build master symptom list
symptom_cols = [c for c in df.columns if c.startswith("Symptom_")]
all_symptoms = set()
for col in symptom_cols:
    all_symptoms.update(df[col].dropna().str.strip().str.lower().tolist())
all_symptoms = sorted(all_symptoms)

# 3️⃣ Create binary feature matrix/one hot encoding
def encode_symptoms(df):
    X = pd.DataFrame(0, index=df.index, columns=all_symptoms)
    for col in symptom_cols:
        for i, symptom in df[col].dropna().str.strip().str.lower().items():
            X.at[i, symptom] = 1
    return X

X = encode_symptoms(df)
y = df["Disease"]

# 4️⃣ Train‑test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5️⃣ Train classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6️⃣ Save model and symptom list
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)
with open(SYMPTOMS_PATH, "wb") as f:
    pickle.dump(all_symptoms, f)

print(f"[✅] Model saved to {MODEL_PATH}")
print(f"[✅] Symptom list saved to {SYMPTOMS_PATH}")
