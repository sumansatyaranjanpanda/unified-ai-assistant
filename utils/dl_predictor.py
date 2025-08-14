# utils/dl_predictor.py

import numpy as np
import pickle
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# 1️⃣ Paths to your artifacts
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
LSTM_MODEL_PATH = os.path.join(MODEL_DIR, "lstm_model.h5")
TOKENIZER_PATH   = os.path.join(MODEL_DIR, "tokenizer.pkl")
LABELENC_PATH    = os.path.join(MODEL_DIR, "le_disease.pkl")

MAX_SEQ_LEN = 50  # must match what you used in training

# 2️⃣ Load artifacts once
model = load_model(LSTM_MODEL_PATH)
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)
with open(LABELENC_PATH, "rb") as f:
    le = pickle.load(f)

def predict_disease_dl(text: str, top_k: int = 3):
    """
    Given a free-text symptom description, return top_k diseases with probabilities.
    """
    # 3️⃣ Text → sequence → padded
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_SEQ_LEN, padding="post", truncating="post")

    # 4️⃣ Inference
    probs = model.predict(padded)[0]  # shape = (num_classes,)
    
    # 5️⃣ Get top_k indices
    top_indices = probs.argsort()[-top_k:][::-1]
    top_probs   = probs[top_indices]
    top_labels  = le.inverse_transform(top_indices)

    # 6️⃣ Return list of (disease, probability)
    return list(zip(top_labels, top_probs.round(3)))
