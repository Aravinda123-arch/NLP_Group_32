from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


# ==================================================
# 1. File paths
# ==================================================
INPUT_FILE = Path(
    "data/before_feature_engineering_dataset.csv"
)

OUTPUT_DIR = Path("data/bert")

TRAIN_OUTPUT = OUTPUT_DIR / "bert_train_dataset.csv"
VALID_OUTPUT = OUTPUT_DIR / "bert_validation_dataset.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# 2. Load existing training pool
# ==================================================
if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input dataset not found: {INPUT_FILE}"
    )

print("Loading BERT training pool...")

df = pd.read_csv(
    INPUT_FILE,
    low_memory=False
)


# ==================================================
# 3. Check required columns
# ==================================================
required_columns = {
    "fused_text",
    "label",
    "group_id"
}

missing_columns = (
    required_columns
    - set(df.columns)
)

if missing_columns:
    raise ValueError(
        f"Dataset missing columns: {missing_columns}"
    )


# ==================================================
# 4. Validate text, labels and groups
# ==================================================
df["fused_text"] = (
    df["fused_text"]
    .fillna("")
    .astype(str)
    .str.strip()
)

df["group_id"] = (
    df["group_id"]
    .fillna("")
    .astype(str)
    .str.strip()
)

df["label"] = pd.to_numeric(
    df["label"],
    errors="coerce"
)


# Keep only valid records
df = df[
    (df["fused_text"] != "") &
    (df["group_id"] != "") &
    (df["label"].isin([0, 1]))
].copy()

df["label"] = (
    df["label"]
    .astype(int)
)

df = df.reset_index(
    drop=True
)


# ==================================================
# 5. Basic validation
# ==================================================
if df.empty:
    raise ValueError(
        "No valid records remain."
    )

if df["label"].nunique() != 2:
    raise ValueError(
        "Both Fake and Real classes are required."
    )


print("\nOriginal BERT training pool:")
print("Records:", len(df))

print("\nClass distribution:")
print(
    df["label"]
    .value_counts()
    .sort_index()
    .rename(
        index={
            0: "Fake",
            1: "Real"
        }
    )
)


# ==================================================
# 6. Create BERT train / validation split
# ==================================================
# 10 folds:
# one fold ≈ 10% validation
# remaining folds ≈ 90% training
#
# group_id prevents related records from being
# placed in both training and validation.

splitter = StratifiedGroupKFold(
    n_splits=10,
    shuffle=True,
    random_state=42
)


train_index, validation_index = next(
    splitter.split(
        X=df["fused_text"],
        y=df["label"],
        groups=df["group_id"]
    )
)


bert_train_df = (
    df.iloc[train_index]
    .copy()
    .reset_index(drop=True)
)

bert_valid_df = (
    df.iloc[validation_index]
    .copy()
    .reset_index(drop=True)
)


# ==================================================
# 7. Check group leakage
# ==================================================
train_groups = set(
    bert_train_df["group_id"]
)

validation_groups = set(
    bert_valid_df["group_id"]
)

group_overlap = (
    train_groups
    & validation_groups
)

if group_overlap:
    raise ValueError(
        f"Group leakage detected: "
        f"{len(group_overlap)} groups overlap."
    )


# ==================================================
# 8. Check exact text leakage
# ==================================================
train_text = set(
    bert_train_df["fused_text"]
)

validation_text = set(
    bert_valid_df["fused_text"]
)

text_overlap = (
    train_text
    & validation_text
)

if text_overlap:
    raise ValueError(
        f"Exact text leakage detected: "
        f"{len(text_overlap)} records overlap."
    )


# ==================================================
# 9. Keep only columns BERT needs
# ==================================================
# group_id is retained for auditing.
#
# BERT model input later:
# fused_text -> tokenizer
# label      -> prediction target

columns_to_save = [
    "fused_text",
    "label",
    "group_id"
]

bert_train_df = bert_train_df[
    columns_to_save
]

bert_valid_df = bert_valid_df[
    columns_to_save
]


# ==================================================
# 10. Save BERT datasets
# ==================================================
bert_train_df.to_csv(
    TRAIN_OUTPUT,
    index=False,
    encoding="utf-8-sig"
)

bert_valid_df.to_csv(
    VALID_OUTPUT,
    index=False,
    encoding="utf-8-sig"
)


# ==================================================
# 11. Final validation report
# ==================================================
print(
    "\n========== BERT DATA PREPARATION =========="
)

print(
    "BERT training records:",
    len(bert_train_df)
)

print(
    "BERT validation records:",
    len(bert_valid_df)
)

print(
    "Total records:",
    len(bert_train_df)
    + len(bert_valid_df)
)


print("\nBERT training class distribution:")

print(
    bert_train_df["label"]
    .value_counts()
    .sort_index()
    .rename(
        index={
            0: "Fake",
            1: "Real"
        }
    )
)


print("\nBERT validation class distribution:")

print(
    bert_valid_df["label"]
    .value_counts()
    .sort_index()
    .rename(
        index={
            0: "Fake",
            1: "Real"
        }
    )
)


print(
    "\nGroup overlap:",
    len(group_overlap)
)

print(
    "Exact text overlap:",
    len(text_overlap)
)


print("\nSaved files:")

print(
    TRAIN_OUTPUT
)

print(
    VALID_OUTPUT
)

print(
    "\nBERT data preparation completed successfully."
)