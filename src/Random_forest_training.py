from pathlib import Path
import time

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


# ==================================================
# File paths
# ==================================================
TRAIN_FILE = Path(
    "data/feature_engineering/train_tfidf_dataset.csv"
)

MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")

MODEL_FILE = MODEL_DIR / "random_forest_model.pkl"
TRAIN_REPORT = REPORT_DIR / "random_forest_training_report.csv"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# Load training dataset
# ==================================================
if not TRAIN_FILE.exists():
    raise FileNotFoundError(
        f"Training dataset not found: {TRAIN_FILE}"
    )

print("Loading TF-IDF training dataset...")

train_df = pd.read_csv(TRAIN_FILE)

required_columns = {"label", "group_id"}
missing_columns = required_columns - set(train_df.columns)

if missing_columns:
    raise ValueError(
        f"Training dataset missing columns: {missing_columns}"
    )


# ==================================================
# Separate features and labels
# ==================================================
feature_names = [
    column
    for column in train_df.columns
    if column not in ["label", "group_id"]
]

if not feature_names:
    raise ValueError("No TF-IDF feature columns were found.")

train_df["label"] = pd.to_numeric(
    train_df["label"],
    errors="coerce"
)

train_df = train_df[
    train_df["label"].isin([0, 1])
].copy()

train_df["label"] = train_df["label"].astype(int)

# Convert TF-IDF features to memory-efficient float32
X_train = (
    train_df[feature_names]
    .fillna(0)
    .astype("float32")
)

y_train = train_df["label"]


# ==================================================
# Validate training data
# ==================================================
if len(X_train) != len(y_train):
    raise ValueError(
        "Training features and labels do not have matching rows."
    )

if y_train.nunique() != 2:
    raise ValueError(
        "Training data must contain both Fake and Real labels."
    )

print("\nTraining records:", len(X_train))
print("Number of TF-IDF features:", len(feature_names))

print("\nTraining class distribution:")
print(
    y_train.value_counts()
    .sort_index()
    .rename(index={0: "Fake", 1: "Real"})
)


# ==================================================
# Develop Random Forest model
# ==================================================
random_forest = RandomForestClassifier(
    n_estimators=300,
    criterion="gini",
    max_depth=None,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features="sqrt",
    class_weight="balanced_subsample",
    bootstrap=True,
    oob_score=True,
    random_state=42,
    n_jobs=-1,
    verbose=1
)


# ==================================================
# Train model
# ==================================================
print("\nTraining Random Forest model...")

start_time = time.time()

random_forest.fit(
    X_train,
    y_train
)

training_time = time.time() - start_time

print("\nTraining completed.")
print(f"Training time: {training_time / 60:.2f} minutes")
print(f"OOB score: {random_forest.oob_score_:.4f}")


# ==================================================
# Save model and feature order
# ==================================================
model_package = {
    "model": random_forest,
    "feature_names": feature_names,
    "label_mapping": {
        0: "Fake",
        1: "Real"
    }
}

joblib.dump(
    model_package,
    MODEL_FILE,
    compress=3
)


# ==================================================
# Save training report
# ==================================================
training_report = pd.DataFrame([{
    "model": "Random Forest",
    "training_records": len(X_train),
    "number_of_features": len(feature_names),
    "fake_training_records": int((y_train == 0).sum()),
    "real_training_records": int((y_train == 1).sum()),
    "n_estimators": random_forest.n_estimators,
    "min_samples_split": random_forest.min_samples_split,
    "min_samples_leaf": random_forest.min_samples_leaf,
    "max_features": random_forest.max_features,
    "class_weight": "balanced_subsample",
    "oob_score": random_forest.oob_score_,
    "training_time_seconds": training_time
}])

training_report.to_csv(
    TRAIN_REPORT,
    index=False
)


print("\nFiles saved:")
print("Model:", MODEL_FILE)
print("Training report:", TRAIN_REPORT)