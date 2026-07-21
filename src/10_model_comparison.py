import os
import ast
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from gensim.models import Word2Vec
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

try:
    from tensorflow.keras.models import load_model  # type: ignore
    from tensorflow.keras.preprocessing.sequence import pad_sequences  # type: ignore
except (ImportError, ModuleNotFoundError):
    from keras.src.saving import load_model  # type: ignore
    from keras.utils import pad_sequences

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
REPORT_DIR = os.path.join(PROJECT_ROOT, "reports")
os.makedirs(REPORT_DIR, exist_ok=True)

TEST_PATH = os.path.join(DATA_DIR, "test_news.csv")
W2V_MODEL_PATH = os.path.join(MODEL_DIR, "word2vec.model")
SVM_MODEL_PATH = os.path.join(MODEL_DIR, "svm_model.pkl")
CNN_MODEL_PATH = os.path.join(MODEL_DIR, "cnn_model.keras")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.pkl")
PLOT_PATH = os.path.join(REPORT_DIR, "confusion_matrices.png")

print("Loading test dataset and trained models...")
test_df = pd.read_csv(TEST_PATH)
y_test = test_df["label"].values

w2v_model = Word2Vec.load(W2V_MODEL_PATH)
svm_model = joblib.load(SVM_MODEL_PATH)
cnn_model = load_model(CNN_MODEL_PATH)
tokenizer = joblib.load(TOKENIZER_PATH)

# --- SVM Predictions ---
print("Evaluating SVM Model...")
if "lemmatized_tokens" in test_df.columns:
    test_tokens = test_df["lemmatized_tokens"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else []).tolist()
else:
    test_tokens = test_df["processed_text"].astype(str).apply(lambda x: x.split()).tolist()

def document_vector(tokens, model, dim=100):
    valid_vectors = [model.wv[word] for word in tokens if word in model.wv]
    if len(valid_vectors) == 0:
        return np.zeros(dim)
    return np.mean(valid_vectors, axis=0)

X_test_svm = np.array([document_vector(t, w2v_model, w2v_model.vector_size) for t in test_tokens])
y_pred_svm = svm_model.predict(X_test_svm)

# --- CNN Predictions ---
print("Evaluating CNN Model...")
X_test_text = test_df["processed_text"].astype(str).tolist()
X_test_seq = tokenizer.texts_to_sequences(X_test_text)
X_test_cnn = pad_sequences(X_test_seq, maxlen=300, padding="post")
y_pred_prob_cnn = cnn_model.predict(X_test_cnn, verbose=0).ravel()
y_pred_cnn = (y_pred_prob_cnn >= 0.5).astype(int)

# --- Compute Metrics ---
def compute_metrics(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1-Score": f1_score(y_true, y_pred)
    }

svm_metrics = compute_metrics(y_test, y_pred_svm)
cnn_metrics = compute_metrics(y_test, y_pred_cnn)

comparison_df = pd.DataFrame([svm_metrics, cnn_metrics], index=["SVM (Word2Vec)", "CNN (Word2Vec)"])

print("\n" + "="*60)
print("              MODEL PERFORMANCE COMPARISON")
print("="*60)
print(comparison_df.round(4).to_string())
print("="*60)

# Save Comparison Report
comparison_df.to_csv(os.path.join(REPORT_DIR, "model_comparison.csv"))

# --- Confusion Matrices Plotting ---
cm_svm = confusion_matrix(y_test, y_pred_svm)
cm_cnn = confusion_matrix(y_test, y_pred_cnn)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.heatmap(cm_svm, annot=True, fmt="d", cmap="Blues", ax=axes[0], cbar=False,
            xticklabels=["Fake", "True"], yticklabels=["Fake", "True"])
axes[0].set_title("SVM Confusion Matrix")
axes[0].set_xlabel("Predicted Label")
axes[0].set_ylabel("True Label")

sns.heatmap(cm_cnn, annot=True, fmt="d", cmap="Greens", ax=axes[1], cbar=False,
            xticklabels=["Fake", "True"], yticklabels=["Fake", "True"])
axes[1].set_title("CNN Confusion Matrix")
axes[1].set_xlabel("Predicted Label")
axes[1].set_ylabel("True Label")

plt.tight_layout()
plt.savefig(PLOT_PATH, dpi=300)
plt.close()

print(f"\nConfusion Matrix plot saved to: {PLOT_PATH}")
print(f"Comparison report saved to: {os.path.join(REPORT_DIR, 'model_comparison.csv')}")
