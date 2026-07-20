import os
import ast
import pandas as pd
from gensim.models import Word2Vec

# ------------------------------------
# File Paths
# ------------------------------------

DATA_PATH = "data/stemmed_news.csv"
MODEL_PATH = "models/word2vec.model"

# ------------------------------------
# Load Dataset
# ------------------------------------

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

# Check required column
if "stemmed_tokens" not in df.columns:
    raise ValueError("'stemmed_tokens' column not found.")

# Convert string representation to list
df["stemmed_tokens"] = df["stemmed_tokens"].apply(ast.literal_eval)

sentences = df["stemmed_tokens"].tolist()

print(f"Total Documents : {len(sentences)}")

# ------------------------------------
# Train Word2Vec
# ------------------------------------

print("\nTraining Word2Vec...\n")

model = Word2Vec(
    sentences=sentences,
    vector_size=100,
    window=5,
    min_count=2,
    workers=4,
    sg=1,          # Skip-Gram
    epochs=20,     # Increased for better learning
    seed=42        # Reproducibility
)

# ------------------------------------
# Save Model
# ------------------------------------

os.makedirs("models", exist_ok=True)

model.save(MODEL_PATH)

print("Word2Vec Model Saved Successfully")
print(f"Vocabulary Size : {len(model.wv)}")

# ------------------------------------
# Example Word
# ------------------------------------

word = "trump"

if word in model.wv:
    print(f"\nEmbedding for '{word}':\n")
    print(model.wv[word])

    print("\nTop 10 Similar Words:\n")
    for similar_word, score in model.wv.most_similar(word, topn=10):
        print(f"{similar_word:<20} {score:.4f}")

else:
    print(f"\n'{word}' not found in vocabulary.")


print(model.wv.most_similar("trump", topn=10))