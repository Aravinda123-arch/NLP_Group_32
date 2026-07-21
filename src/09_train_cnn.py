import os
import ast
import pandas as pd
import numpy as np
import joblib

from gensim.models import Word2Vec
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

try:
    from tensorflow.keras.preprocessing.text import Tokenizer  # type: ignore
    from tensorflow.keras.preprocessing.sequence import pad_sequences  # type: ignore
except (ImportError, ModuleNotFoundError):
    from keras.src.legacy.preprocessing.text import Tokenizer  # type: ignore
    from keras.utils import pad_sequences

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

TRAIN_PATH = os.path.join(DATA_DIR, "train_news.csv")
TEST_PATH = os.path.join(DATA_DIR, "test_news.csv")
W2V_MODEL_PATH = os.path.join(MODEL_DIR, "word2vec.model")
CNN_MODEL_PATH = os.path.join(MODEL_DIR, "cnn_model.keras")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.pkl")

print("Loading datasets and trained Word2Vec model...")
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
w2v_model = Word2Vec.load(W2V_MODEL_PATH)

X_train_text = train_df["processed_text"].astype(str).tolist()
X_test_text = test_df["processed_text"].astype(str).tolist()

y_train = train_df["label"].values
y_test = test_df["label"].values

MAX_WORDS = 20000
MAX_LENGTH = 300
EMBEDDING_DIM = w2v_model.vector_size

print("Fitting Tokenizer on training text...")
tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train_text)

X_train_seq = tokenizer.texts_to_sequences(X_train_text)
X_test_seq = tokenizer.texts_to_sequences(X_test_text)

X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LENGTH, padding="post")
X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LENGTH, padding="post")

# Construct Embedding Matrix from Word2Vec weights
print("Building Embedding Matrix from Word2Vec weights...")
word_index = tokenizer.word_index
vocab_size = min(MAX_WORDS, len(word_index) + 1)
embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))

for word, i in word_index.items():
    if i < vocab_size:
        if word in w2v_model.wv:
            embedding_matrix[i] = w2v_model.wv[word]
        else:
            embedding_matrix[i] = np.random.normal(scale=0.6, size=(EMBEDDING_DIM,))

print(f"Vocabulary Size: {vocab_size} | Embedding Dimension: {EMBEDDING_DIM}")

# Build 1D-CNN Architecture
print("Building 1D-CNN Model...")
model = Sequential([
    Embedding(
        input_dim=vocab_size,
        output_dim=EMBEDDING_DIM,
        weights=[embedding_matrix],
        input_length=MAX_LENGTH,
        trainable=True
    ),
    Conv1D(filters=128, kernel_size=5, activation="relu"),
    MaxPooling1D(pool_size=2),
    Conv1D(filters=64, kernel_size=3, activation="relu"),
    GlobalMaxPooling1D(),
    Dropout(0.5),
    Dense(64, activation="relu"),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Callbacks
early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
checkpoint = ModelCheckpoint(CNN_MODEL_PATH, save_best_only=True, monitor="val_loss")

# Train Model
print("\nTraining CNN Model...")
history = model.fit(
    X_train_pad,
    y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stop, checkpoint],
    verbose=1
)

# Evaluate Model
print("\nEvaluating CNN Model on test set...")
y_pred_prob = model.predict(X_test_pad, verbose=0).ravel()
y_pred = (y_pred_prob >= 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\n" + "="*50)
print("CNN PERFORMANCE METRICS (Word2Vec Embeddings)")
print("="*50)
print(f"Accuracy  : {acc:.4f}")
print(f"Precision : {prec:.4f}")
print(f"Recall    : {rec:.4f}")
print(f"F1-Score  : {f1:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=4))
print("Confusion Matrix:\n", cm)

joblib.dump(tokenizer, TOKENIZER_PATH)
print(f"\nTokenizer saved to: {TOKENIZER_PATH}")
print(f"CNN Model saved to: {CNN_MODEL_PATH}")
