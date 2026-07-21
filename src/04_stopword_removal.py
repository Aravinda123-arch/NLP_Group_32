import os
import pandas as pd
import ast
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
INPUT_FILE = os.path.join(DATA_DIR, "tokenized_news.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "stopwords_removed.csv")

# Load tokenized dataset
df = pd.read_csv(INPUT_FILE)

# Convert string representation back to list
df["tokens"] = df["tokens"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])

stop_words = set(stopwords.words("english"))

# Remove Stop Words
def remove_stopwords(tokens):
    return [word for word in tokens if word.lower() not in stop_words]

print("Removing stopwords...")
df["filtered_tokens"] = df["tokens"].apply(remove_stopwords)

# Save
df.to_csv(OUTPUT_FILE, index=False)

print("Stopword Removal Completed Successfully")
print(f"Saved dataset to: {OUTPUT_FILE}")
print(df[["filtered_tokens"]].head())