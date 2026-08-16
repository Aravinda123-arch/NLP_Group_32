from pathlib import Path
import pandas as pd

# =========================================================
# Project Paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

DATA_DIR.mkdir(parents=True, exist_ok=True)

FAKE_PATH = DATA_DIR / "Fake.csv"
TRUE_PATH = DATA_DIR / "True.csv"

OUTPUT_PATH = DATA_DIR / "merged_news.csv"


# =========================================================
# Load Dataset
# =========================================================

print("=" * 60)
print("Loading datasets...")
print("=" * 60)

try:
    fake_df = pd.read_csv(FAKE_PATH)
    true_df = pd.read_csv(TRUE_PATH)

except FileNotFoundError as e:
    raise FileNotFoundError(
        f"\nDataset not found.\nPlease place Fake.csv and True.csv inside:\n{DATA_DIR}"
    ) from e


# =========================================================
# Add Labels
# =========================================================

fake_df["label"] = 0
true_df["label"] = 1


# =========================================================
# Select Columns
# =========================================================

required_columns = [
    "title",
    "text",
    "subject",
    "date",
    "label"
]

fake_df = fake_df[required_columns]
true_df = true_df[required_columns]


# =========================================================
# Merge Dataset
# =========================================================

merged_df = pd.concat(
    [fake_df, true_df],
    ignore_index=True
)


# =========================================================
# Remove Empty Records
# =========================================================

merged_df.dropna(
    subset=["title", "text"],
    inplace=True
)

merged_df["title"] = merged_df["title"].astype(str)
merged_df["text"] = merged_df["text"].astype(str)

merged_df = merged_df[
    merged_df["title"].str.strip() != ""
]

merged_df = merged_df[
    merged_df["text"].str.strip() != ""
]


# =========================================================
# Remove Duplicate Articles
# =========================================================

merged_df.drop_duplicates(
    subset=["title", "text"],
    inplace=True
)


# =========================================================
# Shuffle Dataset
# =========================================================

merged_df = merged_df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# =========================================================
# Save Dataset
# =========================================================

merged_df.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8"
)


# =========================================================
# Display Information
# =========================================================

print("\nDataset Successfully Merged")
print("-" * 60)

print(f"Total Records : {len(merged_df):,}")

print("\nClass Distribution")

print(
    merged_df["label"]
    .value_counts()
    .rename({
        0: "Fake",
        1: "Real"
    })
)

print("\nColumns")

print(list(merged_df.columns))

print("\nDataset Preview")

print(
    merged_df.head()
)


print("\nCompleted Successfully")
print("=" * 60)