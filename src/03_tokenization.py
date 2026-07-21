import os
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize

nltk.download("punkt", quiet=True)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
INPUT_FILE = os.path.join(DATA_DIR, "cleaned_news.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "tokenized_news.csv")

# -------------------------
# Load cleaned dataset
# -------------------------
df = pd.read_csv(INPUT_FILE)

# -------------------------
# Tokenization Function
# -------------------------
def tokenize_text(text):
    if pd.isna(text):
        return []
    return word_tokenize(str(text))

print("Tokenizing text...")
df["tokens"] = df["clean_text"].apply(tokenize_text)

# Save
df.to_csv(OUTPUT_FILE, index=False)

print("Tokenization Completed Successfully")
print(f"Saved tokenized dataset to: {OUTPUT_FILE}")
print(df[["tokens"]].head())