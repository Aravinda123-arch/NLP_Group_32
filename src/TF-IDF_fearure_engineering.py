from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


# File paths
TRAIN_FILE = Path("data/before_feature_engineering_dataset.csv")
TEST_FILE = Path("data/test_dataset.csv")
OUTPUT_DIR = Path("data/feature_engineering")
MODEL_DIR = Path("models")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# Load datasets
train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

required = {"fused_text", "label", "group_id"}

if not required.issubset(train_df.columns):
    raise ValueError(
        f"Training dataset missing: {required - set(train_df.columns)}"
    )

if not required.issubset(test_df.columns):
    raise ValueError(
        f"Test dataset missing: {required - set(test_df.columns)}"
    )


# Remove missing or invalid rows
for df in [train_df, test_df]:
    df["fused_text"] = (
        df["fused_text"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df["label"] = pd.to_numeric(
        df["label"],
        errors="coerce"
    )

train_df = train_df[
    (train_df["fused_text"] != "") &
    (train_df["label"].isin([0, 1])) &
    (train_df["group_id"].notna())
].copy()

test_df = test_df[
    (test_df["fused_text"] != "") &
    (test_df["label"].isin([0, 1])) &
    (test_df["group_id"].notna())
].copy()

train_df["label"] = train_df["label"].astype(int)
test_df["label"] = test_df["label"].astype(int)


# Check train-test leakage
group_overlap = (
    set(train_df["group_id"])
    & set(test_df["group_id"])
)

text_overlap = (
    set(train_df["fused_text"])
    & set(test_df["fused_text"])
)

if group_overlap:
    raise ValueError(
        f"Group leakage detected: {len(group_overlap)}"
    )

if text_overlap:
    raise ValueError(
        f"Text leakage detected: {len(text_overlap)}"
    )


# Separate text and labels
X_train = train_df["fused_text"]
y_train = train_df["label"]

X_test = test_df["fused_text"]
y_test = test_df["label"]


# Create TF-IDF converter
tfidf = TfidfVectorizer(
    lowercase=False,
    strip_accents="unicode",
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.95,
    max_features=5000,
    sublinear_tf=True,
    norm="l2",
    dtype=np.float32
)


# Fit only on training data
X_train_tfidf = tfidf.fit_transform(X_train)

# Transform test data using training vocabulary
X_test_tfidf = tfidf.transform(X_test)


# NEW: Save fitted TF-IDF vectorizer
joblib.dump(
    tfidf,
    MODEL_DIR / "tfidf_vectorizer.pkl"
)


# Convert matrices to DataFrames
features = tfidf.get_feature_names_out()

train_tfidf_df = pd.DataFrame.sparse.from_spmatrix(
    X_train_tfidf,
    columns=features
)

test_tfidf_df = pd.DataFrame.sparse.from_spmatrix(
    X_test_tfidf,
    columns=features
)


# Add labels and group IDs
train_tfidf_df["label"] = y_train.reset_index(drop=True)

train_tfidf_df["group_id"] = (
    train_df["group_id"]
    .reset_index(drop=True)
)

test_tfidf_df["label"] = y_test.reset_index(drop=True)

test_tfidf_df["group_id"] = (
    test_df["group_id"]
    .reset_index(drop=True)
)


# Save transformed datasets
train_tfidf_df.to_csv(
    OUTPUT_DIR / "train_tfidf_dataset.csv",
    index=False
)

test_tfidf_df.to_csv(
    OUTPUT_DIR / "test_tfidf_dataset.csv",
    index=False
)

pd.DataFrame({
    "feature_name": features
}).to_csv(
    OUTPUT_DIR / "tfidf_feature_names.csv",
    index=False
)


# Report
print("TF-IDF conversion completed.")
print("Training shape:", X_train_tfidf.shape)
print("Testing shape:", X_test_tfidf.shape)
print("Vocabulary size:", len(features))
print("Group overlap:", len(group_overlap))
print("Text overlap:", len(text_overlap))

print("\nSaved files:")
print("data/feature_engineering/train_tfidf_dataset.csv")
print("data/feature_engineering/test_tfidf_dataset.csv")
print("data/feature_engineering/tfidf_feature_names.csv")
print("models/tfidf_vectorizer.pkl")