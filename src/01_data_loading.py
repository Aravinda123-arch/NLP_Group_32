# ==========================================
# 01_data_loading.py
# NLP Fake News Detection Project
# ==========================================

import os
import nltk
import pandas as pd

# ------------------------------------------
# Download Required NLTK Resources (Quietly)
# ------------------------------------------
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

# ------------------------------------------
# Project Paths
# ------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")

FAKE_FILE = os.path.join(DATA_DIR, "Fake.csv")
TRUE_FILE = os.path.join(DATA_DIR, "True.csv")
MERGED_FILE = os.path.join(DATA_DIR, "merged_news.csv")

# ------------------------------------------
# Check Dataset Files
# ------------------------------------------
if not os.path.exists(FAKE_FILE):
    raise FileNotFoundError(f"Fake.csv not found:\n{FAKE_FILE}")

if not os.path.exists(TRUE_FILE):
    raise FileNotFoundError(f"True.csv not found:\n{TRUE_FILE}")

# ------------------------------------------
# Load Datasets
# ------------------------------------------
fake_df = pd.read_csv(FAKE_FILE)
true_df = pd.read_csv(TRUE_FILE)

# ------------------------------------------
# Add Labels
# Fake = 0
# True = 1
# ------------------------------------------
fake_df["label"] = 0
true_df["label"] = 1

# ------------------------------------------
# Merge Datasets
# ------------------------------------------
df = pd.concat([fake_df, true_df], ignore_index=True)

# ------------------------------------------
# Shuffle Dataset
# ------------------------------------------
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# ------------------------------------------
# Save Dataset
# ------------------------------------------
df.to_csv(MERGED_FILE, index=False)

# ------------------------------------------
# Display Information
# ------------------------------------------
print("=" * 60)
print("FAKE NEWS DATASET LOADED SUCCESSFULLY")
print("=" * 60)

print("\nFirst 5 Records")
print(df.head())

print("\n" + "=" * 60)
print(f"Dataset Shape : {df.shape}")

print("\nColumns")
print(df.columns.tolist())

print("\nMissing Values")
print(df.isnull().sum())

print("\nLabel Distribution")
print(df["label"].value_counts())

print("\nData Types")
print(df.dtypes)

print("\nMerged Dataset Saved Successfully")
print(MERGED_FILE)

print("\nProcess Completed Successfully.")