from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

INPUT_FILE = Path("data/advanced_preprocessed_dataset.csv")

TRAIN_FILE = Path("data/before_feature_engineering_dataset.csv")
TEST_FILE = Path("data/test_dataset.csv")


# Load preprocessed dataset
df = pd.read_csv(INPUT_FILE)

required = {"fused_text", "label", "group_id"}

missing = required - set(df.columns)

if missing:
    raise ValueError(
        f"Advanced dataset missing columns: {missing}"
    )

# Remove invalid rows
df = df.dropna(
    subset=["fused_text", "label", "group_id"]
).copy()

df["label"] = pd.to_numeric(
    df["label"],
    errors="coerce"
)

df = df[
    df["label"].isin([0, 1])
].copy()

df["label"] = df["label"].astype(int)


# 80% train / 20% test
splitter = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

train_index, test_index = next(
    splitter.split(
        df["fused_text"],
        df["label"],
        groups=df["group_id"]
    )
)

train_df = df.iloc[train_index].reset_index(drop=True)
test_df = df.iloc[test_index].reset_index(drop=True)


# Check leakage
overlap = (
    set(train_df["group_id"])
    & set(test_df["group_id"])
)

if overlap:
    raise ValueError(
        f"Group leakage detected: {len(overlap)}"
    )


# Save
train_df.to_csv(
    TRAIN_FILE,
    index=False
)

test_df.to_csv(
    TEST_FILE,
    index=False
)


print("Data split completed.")

print("\nTraining columns:")
print(train_df.columns.tolist())

print("\nTesting columns:")
print(test_df.columns.tolist())

print("\nTraining rows:", len(train_df))
print("Testing rows:", len(test_df))
print("Group overlap:", len(overlap))

print("\nSaved:")
print(TRAIN_FILE)
print(TEST_FILE)