import os
import pandas as pd
from sklearn.model_selection import train_test_split

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
INPUT_FILE = os.path.join(DATA_DIR, "lemmatized_news.csv")

TRAIN_FILE = os.path.join(DATA_DIR, "train_news.csv")
TEST_FILE = os.path.join(DATA_DIR, "test_news.csv")

print("Loading lemmatized dataset for train-test split...")
df = pd.read_csv(INPUT_FILE)

# Fill nulls if any
df["processed_text"] = df["processed_text"].fillna("")

print("Performing 80/20 Stratified Train-Test Split...")
train_df, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["label"]
)

train_df.to_csv(TRAIN_FILE, index=False)
test_df.to_csv(TEST_FILE, index=False)

print("Train-Test Split Completed Successfully")
print(f"Train samples: {len(train_df)} | Saved to: {TRAIN_FILE}")
print(f"Test samples : {len(test_df)}  | Saved to: {TEST_FILE}")
