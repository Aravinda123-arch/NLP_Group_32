import os
import ast
import pandas as pd
import numpy as np
import joblib

from gensim.models import Word2Vec
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

TRAIN_PATH = os.path.join(DATA_DIR, "train_news.csv")
TEST_PATH = os.path.join(DATA_DIR, "test_news.csv")
W2V_MODEL_PATH = os.path.join(MODEL_DIR, "word2vec.model")
SVM_MODEL_PATH = os.path.join(MODEL_DIR, "svm_model.pkl")

print("Loading train & test datasets...")
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

print("Loading trained Word2Vec model...")
w2v_model = Word2Vec.load(W2V_MODEL_PATH)
vector_size = w2v_model.vector_size

def get_tokens(df):
    if "lemmatized_tokens" in df.columns:
        return df["lemmatized_tokens"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else []).tolist()
    return df["processed_text"].astype(str).apply(lambda x: x.split()).tolist()

train_tokens = get_tokens(train_df)
test_tokens = get_tokens(test_df)

y_train = train_df["label"].values
y_test = test_df["label"].values

def document_vector(tokens, model, dim):
    valid_vectors = [model.wv[word] for word in tokens if word in model.wv]
    if len(valid_vectors) == 0:
        return np.zeros(dim)
    return np.mean(valid_vectors, axis=0)

print("Transforming documents to Average Word2Vec vectors...")
X_train_vec = np.array([document_vector(tokens, w2v_model, vector_size) for tokens in train_tokens])
X_test_vec = np.array([document_vector(tokens, w2v_model, vector_size) for tokens in test_tokens])

print("Training Linear Support Vector Machine (SVM)...")
svm = SVC(kernel="linear", C=1.0, probability=True, random_state=42)
svm.fit(X_train_vec, y_train)

print("Evaluating SVM on test set...")
y_pred = svm.predict(X_test_vec)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\n" + "="*50)
print("SVM PERFORMANCE METRICS (Word2Vec Features)")
print("="*50)
print(f"Accuracy  : {acc:.4f}")
print(f"Precision : {prec:.4f}")
print(f"Recall    : {rec:.4f}")
print(f"F1-Score  : {f1:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=4))
print("Confusion Matrix:\n", cm)

joblib.dump(svm, SVM_MODEL_PATH)
print(f"\nSVM Model saved to: {SVM_MODEL_PATH}")
