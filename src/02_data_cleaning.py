import os
import pandas as pd
import re

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
INPUT_FILE = os.path.join(DATA_DIR, "merged_news.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "cleaned_news.csv")

# Load merged dataset
df = pd.read_csv(INPUT_FILE)

# -----------------------------
# Combine title + text
# -----------------------------
df["text"] = df["text"].fillna("")
df["title"] = df["title"].fillna("")
df["content"] = df["title"] + " " + df["text"]

# -----------------------------
# Cleaning Function
# -----------------------------
def clean_text(text):
    if pd.isna(text):
        return ""

    # Remove Reuters / publisher metadata headers (target leakage fix)
    text = re.sub(r'^.*?\s*\([A-Za-z\s]+\)\s*-\s*', ' ', text)
    text = re.sub(r'\bReuters\b', ' ', text, flags=re.IGNORECASE)

    # lowercase
    text = text.lower()

    # remove html
    text = re.sub(r'<.*?>', ' ', text)

    # remove urls
    text = re.sub(r'http\S+|www\S+', ' ', text)

    # remove emails
    text = re.sub(r'\S+@\S+', ' ', text)

    # remove numbers
    text = re.sub(r'\d+', ' ', text)

    # remove punctuation / special chars
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    # remove multiple spaces
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

# Apply cleaning
print("Cleaning text and removing publisher leakage...")
df["clean_text"] = df["content"].apply(clean_text)

# Remove empty rows
df = df[df["clean_text"] != ""]

# Save dataset
df.to_csv(OUTPUT_FILE, index=False)

print("Cleaning Completed Successfully")
print(f"Cleaned dataset saved to: {OUTPUT_FILE}")
print(df[["clean_text"]].head())