# ml_models/healthcare_predictor.py

import pickle, os
import pandas as pd

MODEL_PATH = "models/healthcare_model.pkl"
SYMPTOMS_PATH = "models/symptom_list.pkl"

# Load artifacts
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
with open(SYMPTOMS_PATH, "rb") as f:
    all_symptoms = pickle.load(f)

def predict_disease(selected_symptoms):
    """
    selected_symptoms: list of strings (normalized to lower-case)
    """
    # Build single-row DataFrame
    X = pd.DataFrame(0, index=[0], columns=all_symptoms)
    for sym in selected_symptoms:
        sym = sym.strip().lower()
        if sym in X.columns:
            X.at[0, sym] = 1
    pred = model.predict(X)[0]
    return pred