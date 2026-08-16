from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    classification_report,
    confusion_matrix
)


# ==================================================
# 1. File paths
# ==================================================
MODEL_FILE = Path(
    "models/random_forest_model.pkl"
)

TEST_FILE = Path(
    "data/feature_engineering/test_tfidf_dataset.csv"
)


# ==================================================
# 2. Check files
# ==================================================
if not MODEL_FILE.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_FILE}"
    )

if not TEST_FILE.exists():
    raise FileNotFoundError(
        f"Test dataset not found: {TEST_FILE}"
    )


# ==================================================
# 3. Load trained Random Forest model
# ==================================================
print("Loading trained Random Forest...")

package = joblib.load(MODEL_FILE)

model = package["model"]
feature_names = package["feature_names"]


# ==================================================
# 4. Load ONLY test dataset
# ==================================================
print("Loading test dataset...")

test_df = pd.read_csv(TEST_FILE)

required = {"label", "group_id"}

missing = required - set(test_df.columns)

if missing:
    raise ValueError(
        f"Test dataset missing columns: {missing}"
    )


# ==================================================
# 5. Validate model features
# ==================================================
missing_features = (
    set(feature_names)
    - set(test_df.columns)
)

if missing_features:
    raise ValueError(
        f"Test dataset missing "
        f"{len(missing_features)} model features.\n"
        "The model and test TF-IDF dataset "
        "must come from the same TF-IDF run."
    )


# ==================================================
# 6. Prepare test features
# ==================================================
test_df["label"] = pd.to_numeric(
    test_df["label"],
    errors="coerce"
)

test_df = test_df[
    test_df["label"].isin([0, 1])
].reset_index(drop=True)


# Use exact same feature order as training
X_test = (
    test_df[feature_names]
    .fillna(0)
    .astype("float32")
)

y_test = (
    test_df["label"]
    .astype(int)
    .to_numpy()
)


# ==================================================
# 7. Final feature check
# ==================================================
if X_test.shape[1] != model.n_features_in_:
    raise ValueError(
        f"Feature mismatch.\n"
        f"Model expects: {model.n_features_in_}\n"
        f"Test contains: {X_test.shape[1]}"
    )


print("\n========== TEST DATA ==========")
print("Test records:", len(y_test))
print("Features:", X_test.shape[1])

print("\nClass distribution:")
print(
    test_df["label"]
    .value_counts()
    .sort_index()
    .rename(index={
        0: "Fake",
        1: "Real"
    })
)


# ==================================================
# 8. Predict test data
# ==================================================
print("\nTesting Random Forest...")

y_pred = model.predict(X_test)


# Probability of Real = class 1
classes = list(model.classes_)

if 1 not in classes:
    raise ValueError(
        "Real class (1) not found in model."
    )

real_index = classes.index(1)

y_probability_real = (
    model.predict_proba(X_test)
    [:, real_index]
)


# ==================================================
# 9. Calculate metrics
# ==================================================
accuracy = accuracy_score(
    y_test,
    y_pred
)

balanced_accuracy = balanced_accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_probability_real
)

mcc = matthews_corrcoef(
    y_test,
    y_pred
)


# ==================================================
# 10. 95% accuracy confidence interval
# ==================================================
rng = np.random.default_rng(42)

bootstrap_scores = []

for _ in range(1000):

    indexes = rng.integers(
        0,
        len(y_test),
        len(y_test)
    )

    bootstrap_scores.append(
        accuracy_score(
            y_test[indexes],
            y_pred[indexes]
        )
    )


lower_ci, upper_ci = np.percentile(
    bootstrap_scores,
    [2.5, 97.5]
)


# ==================================================
# 11. Final test results
# ==================================================
print(
    "\n========== FINAL RANDOM FOREST TEST =========="
)

print(f"Accuracy:          {accuracy:.4f}")

print(
    f"95% Accuracy CI:   "
    f"{lower_ci:.4f} - {upper_ci:.4f}"
)

print(
    f"Balanced Accuracy: {balanced_accuracy:.4f}"
)

print(
    f"Macro Precision:   {precision:.4f}"
)

print(
    f"Macro Recall:      {recall:.4f}"
)

print(
    f"Macro F1-score:    {f1:.4f}"
)

print(
    f"ROC-AUC:           {roc_auc:.4f}"
)

print(
    f"MCC:               {mcc:.4f}"
)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Fake",
            "Real"
        ],
        digits=4,
        zero_division=0
    )
)


print("Confusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)