# utils/healthcare_data.py

import pandas as pd
import os

# You might want to compute ROOT_DIR as shown before
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "healthcare"))

# 1️⃣ Load the main symptom→disease dataset
df_dataset = pd.read_csv(os.path.join(DATA_DIR, "dataset.csv"))

# 2️⃣ Load symptom severity (assuming columns: Symptom, weight)
df_severity = pd.read_csv(os.path.join(DATA_DIR, "Symptom-severity.csv"))
severity_map = dict(zip(df_severity.Symptom.str.lower(), df_severity.weight))

# 3️⃣ Load symptom descriptions (Disease, Description)
df_desc = pd.read_csv(os.path.join(DATA_DIR, "symptom_Description.csv"))
desc_map = dict(zip(df_desc.Disease.str.lower(), df_desc.Description))


# 4️⃣ Load disease precautions (Disease, Precaution_1, Precaution_2, …)
df_prec = pd.read_csv(os.path.join(DATA_DIR, "symptom_precaution.csv"))
prec_map = {
    row.Disease.lower(): [row[f"Precaution_{i}"] 
                           for i in range(1, 5) 
                           if pd.notna(row.get(f"Precaution_{i}"))]
    for _, row in df_prec.iterrows()
}


def get_description():
    """
    Returns a dictionary mapping diseases to their descriptions.
    """
    return desc_map

def get_symptom_details(symptoms: list[str]) -> list[dict]:
    """
    For each symptom, return a dict with:
    - name
    - description
    - severity
    """
    details = []
    for s in symptoms:
        key = s.lower()
        details.append({
            "symptom": s,
            "severity": severity_map.get(key, "Unknown")
        })
    return details

def get_disease_precautions(disease: str) -> list[str]:
    """
    Return the list of precautions for the given disease.
    """
    return prec_map.get(disease.lower(), ["No precautions found."])
