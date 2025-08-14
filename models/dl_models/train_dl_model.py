import os
import random
import numpy as np
import pandas as pd
import pickle
import requests
from zipfile import ZipFile

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# ========================
# 1️⃣ Config
# ========================
DATA_PATH = "data/healthcare/disease_symptom_dataset.csv"
EMBEDDING_DIM = 100
GLOVE_NAME = f"glove.6B.{EMBEDDING_DIM}d.txt"
GLOVE_PATH = os.path.join("embeddings", GLOVE_NAME)
MAX_NUM_WORDS = 10000
MAX_SEQ_LEN = 50
TEST_SIZE = 0.2
RANDOM_STATE = 42
BATCH_SIZE = 32
EPOCHS = 5   # Keep small for Streamlit Cloud
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs("embeddings", exist_ok=True)

# ========================
# 2️⃣ Reproducibility
# ========================
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# ========================
## 3️⃣ Download GloVe if not present
# ========================
def download_glove():
    if not os.path.exists(GLOVE_PATH):
        print(f"[INFO] {GLOVE_NAME} not found. Downloading...")
        url = "http://nlp.stanford.edu/data/glove.6B.zip"
        zip_path = os.path.join("embeddings", "glove.6B.zip")

        r = requests.get(url, stream=True)
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(1024):
                f.write(chunk)

        with ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall("embeddings")
        print("[INFO] GloVe embeddings downloaded and extracted.")

download_glove()

# ========================
# 4️⃣ Load dataset
# ========================
print("[INFO] Loading dataset...")
df = pd.read_csv(DATA_PATH)
texts = df["description"].astype(str).tolist()
labels = df["disease"].tolist()

# ========================
# 5️⃣ Encode labels
# ========================
le = LabelEncoder()
y = le.fit_transform(labels)
num_classes = len(le.classes_)

with open(os.path.join(MODEL_DIR, "le_disease.pkl"), "wb") as f:
    pickle.dump(le, f)

# ========================
# 6️⃣ Tokenize texts
# ========================
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index

with open(os.path.join(MODEL_DIR, "tokenizer.pkl"), "wb") as f:
    pickle.dump(tokenizer, f)

# ========================
# 7️⃣ Pad sequences
# ========================
X = pad_sequences(sequences, maxlen=MAX_SEQ_LEN, padding="post", truncating="post")

# ========================
# 8️⃣ Prepare embedding matrix
# ========================
print("[INFO] Loading GloVe embeddings...")
embeddings_index = {}
with open(GLOVE_PATH, encoding="utf8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype="float32")
        embeddings_index[word] = coefs

num_words = min(MAX_NUM_WORDS, len(word_index) + 1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= num_words:
        continue
    vector = embeddings_index.get(word)
    if vector is not None:
        embedding_matrix[i] = vector

# ========================
# 9️⃣ Train-test split
# ========================
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

# ========================
# 🔟 Build LSTM model
# ========================
model = Sequential([
    Embedding(num_words, EMBEDDING_DIM, weights=[embedding_matrix],
              input_length=MAX_SEQ_LEN, trainable=False),
    Bidirectional(LSTM(64, return_sequences=False)),
    Dropout(0.5),
    Dense(64, activation="relu"),
    Dropout(0.5),
    Dense(num_classes, activation="softmax"),
])

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

model.summary()

# ========================
# 1️⃣1️⃣ Callbacks
# ========================
checkpoint = ModelCheckpoint(
    os.path.join(MODEL_DIR, "lstm_model"),
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)
earlystop = EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True)

# ========================
# 1️⃣2️⃣ Train
# ========================
print("[INFO] Starting training...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[checkpoint, earlystop]
)

print("[INFO] Training complete. Best model saved to:", MODEL_DIR)
