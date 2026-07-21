import os
import ast
import pandas as pd
from gensim.models import Word2Vec

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

TRAIN_PATH = os.path.join(DATA_DIR, "train_news.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "word2vec.model")

print("Loading training dataset for Word2Vec training...")
df = pd.read_csv(TRAIN_PATH)

if "lemmatized_tokens" in df.columns:
    sentences = df["lemmatized_tokens"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else []).tolist()
else:
    sentences = df["processed_text"].astype(str).apply(lambda x: x.split()).tolist()

# Filter out empty token lists
sentences = [s for s in sentences if len(s) > 0]

print(f"Training Word2Vec strictly on {len(sentences)} training documents...")

w2v_model = Word2Vec(
    sentences=sentences,
    vector_size=100,
    window=5,
    min_count=2,
    workers=4,
    sg=1,          # Skip-Gram
    epochs=20,
    seed=42
)

os.makedirs(MODEL_DIR, exist_ok=True)
w2v_model.save(MODEL_PATH)

print("Word2Vec Model Saved Successfully")
print(f"Vocabulary Size: {len(w2v_model.wv)}")
print(f"Saved to: {MODEL_PATH}")
