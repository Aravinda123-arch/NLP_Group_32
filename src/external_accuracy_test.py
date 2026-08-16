from pathlib import Path
import re
import unicodedata

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)


# ==================================================
# 1. PROJECT PATHS
# ==================================================

# external_accuracy_test.py is inside src/
# parent.parent = Fake-News-Detection project folder
BASE_DIR = Path(__file__).resolve().parent.parent

def get_validation_file(base_dir, name):
    folder = base_dir / "data" / "external_validation"
    for ext in [".xlsx", ".csv", ".xls"]:
        p = folder / f"{name}{ext}"
        if p.exists():
            return p
    return folder / f"{name}.csv"


FAKE_FILE = get_validation_file(BASE_DIR, "Fake")
REAL_FILE = get_validation_file(BASE_DIR, "Real")

MODEL_FILE = BASE_DIR / "models" / "random_forest_model.pkl"
TFIDF_FILE = BASE_DIR / "models" / "tfidf_vectorizer.pkl"

REPORT_DIR = BASE_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# 2. CHECK FILES
# ==================================================
print("Fake file:", FAKE_FILE)
print("Real file:", REAL_FILE)
print("RF model:", MODEL_FILE)
print("TF-IDF:", TFIDF_FILE)

for file_path in [
    FAKE_FILE,
    REAL_FILE,
    MODEL_FILE,
    TFIDF_FILE
]:
    if not file_path.exists():
        raise FileNotFoundError(
            f"\nFile not found:\n{file_path}\n"
            "Check the filename and folder location."
        )


# ==================================================
# 3. LOAD EXTERNAL DATA
# ==================================================
def load_file_safe(path):
    if str(path).lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path)

    # Try common CSV encodings
    for encoding in ["utf-8-sig", "utf-8", "cp1252", "latin1"]:
        try:
            return pd.read_csv(
                path,
                encoding=encoding,
                low_memory=False
            )
        except UnicodeDecodeError:
            continue

    raise ValueError(
        f"Unable to decode file: {path}"
    )


fake_df = load_file_safe(FAKE_FILE)
real_df = load_file_safe(REAL_FILE)

# Standardize headers
fake_df.columns = (
    fake_df.columns
    .str.strip()
    .str.lower()
)

real_df.columns = (
    real_df.columns
    .str.strip()
    .str.lower()
)

print("\nFake columns:")
print(fake_df.columns.tolist())

print("\nReal columns:")
print(real_df.columns.tolist())


# ==================================================
# 4. CHECK IMPORTANT COLUMNS
# ==================================================

# Based directly on your uploaded Polyglot dataset view
TEXT_COLUMN = "english translated version"

if TEXT_COLUMN not in fake_df.columns:
    raise ValueError(
        f"'{TEXT_COLUMN}' not found in Fake.csv"
    )

if TEXT_COLUMN not in real_df.columns:
    raise ValueError(
        f"'{TEXT_COLUMN}' not found in Real.csv"
    )


# ==================================================
# 5. LABELS
# ==================================================

# Your project:
# 0 = Fake
# 1 = Real

fake_df["label"] = 0
real_df["label"] = 1

external_df = pd.concat(
    [fake_df, real_df],
    ignore_index=True
)

print("\nOriginal external records:")
print("Fake:", len(fake_df))
print("Real:", len(real_df))
print("Total:", len(external_df))


# ==================================================
# 6. DATE HANDLING
# ==================================================

# Actual article publication date
if "news date" in external_df.columns:
    external_df["parsed_news_date"] = pd.to_datetime(
        external_df["news date"],
        errors="coerce"
    )
else:
    external_df["parsed_news_date"] = pd.NaT


# Dataset collection date
if "gathering date" in external_df.columns:
    external_df["parsed_gathering_date"] = pd.to_datetime(
        external_df["gathering date"],
        errors="coerce"
    )
else:
    external_df["parsed_gathering_date"] = pd.NaT


# Prefer news date.
# Use gathering date only when news date is unavailable.
external_df["validation_date"] = (
    external_df["parsed_news_date"]
    .fillna(external_df["parsed_gathering_date"])
)


# Keep 2020 onward
external_df = external_df[
    external_df["validation_date"]
    >= pd.Timestamp("2020-01-01")
].copy()

print("\nPost-2020 external records:")
print(len(external_df))

print(
    "Date range:",
    external_df["validation_date"].min(),
    "to",
    external_df["validation_date"].max()
)


# ==================================================
# 7. USE ENGLISH TRANSLATED TEXT
# ==================================================

# Do not use "news headline" here because the
# screenshot shows headlines are in original languages.
external_df["fused_text"] = (
    external_df["english translated version"]
    .fillna("")
    .astype(str)
)


# ==================================================
# 8. SAME BASIC PREPROCESSING
# ==================================================
def preprocess_text(text):

    if pd.isna(text):
        return ""

    text = unicodedata.normalize(
        "NFKC",
        str(text)
    )

    # Remove HTML
    text = re.sub(
        r"<[^>]+>",
        " ",
        text
    )

    # Remove URLs
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text,
        flags=re.I
    )

    # Remove emails
    text = re.sub(
        r"\S+@\S+\.\S+",
        " ",
        text
    )

    # Remove Reuters/AP/AFP datelines
    text = re.sub(
        r"^\s*[A-Z][A-Z\s,./'-]{1,80}"
        r"\s+\((Reuters|AP|AFP)\)"
        r"\s*[-–—:]?\s*",
        " ",
        text,
        flags=re.I
    )

    # Remove reporting credits
    text = re.sub(
        r"\b(reporting|writing|editing|additional reporting)"
        r"\s+by\s+[^.;]{1,120}[.;]?",
        " ",
        text,
        flags=re.I
    )

    # Remove control characters
    text = re.sub(
        r"[\x00-\x1f\x7f-\x9f]",
        " ",
        text
    )

    text = text.lower()

    # Normalize spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


external_df["fused_text"] = (
    external_df["fused_text"]
    .apply(preprocess_text)
)


# ==================================================
# 9. REMOVE INVALID RECORDS
# ==================================================
external_df = external_df[
    external_df["fused_text"]
    .str.split()
    .str.len()
    .ge(20)
].copy()

external_df = external_df.drop_duplicates(
    subset="fused_text",
    keep="first"
)

external_df = external_df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

print("\nFinal external records:")
print(len(external_df))

print("\nExternal class distribution:")
print(
    external_df["label"]
    .value_counts()
    .sort_index()
    .rename(index={
        0: "Fake",
        1: "Real"
    })
)


# ==================================================
# 10. LOAD OLD TRAINED MODEL
# ==================================================
model_package = joblib.load(
    MODEL_FILE
)

random_forest = model_package["model"]

model_features = model_package.get(
    "feature_names"
)


# ==================================================
# 11. LOAD OLD FITTED TF-IDF
# ==================================================
tfidf = joblib.load(
    TFIDF_FILE
)

tfidf_features = list(
    tfidf.get_feature_names_out()
)

# Confirm TF-IDF matches the RF model
if model_features is not None:
    if list(model_features) != tfidf_features:
        raise ValueError(
            "Random Forest and TF-IDF features do not match. "
            "Retrain the RF model using the current saved vectorizer."
        )


# ==================================================
# 12. TRANSFORM NEW DATA
# ==================================================

# Correct:
# transform() only
#
# Do NOT use fit()
# Do NOT use fit_transform()

X_external_sparse = tfidf.transform(
    external_df["fused_text"]
)

X_external = pd.DataFrame(
    X_external_sparse.toarray(),
    columns=tfidf.get_feature_names_out()
)

y_external = external_df["label"].to_numpy()

print("\nExternal TF-IDF shape:")
print(X_external.shape)

empty_vectors = int(
    (X_external_sparse.getnnz(axis=1) == 0).sum()
)

print(
    "External empty TF-IDF vectors:",
    empty_vectors
)


# ==================================================
# 13. PREDICT
# ==================================================
y_pred = random_forest.predict(
    X_external
)

class_positions = {
    label: index
    for index, label
    in enumerate(random_forest.classes_)
}

if 1 not in class_positions:
    raise ValueError(
        "Real class (1) is missing from the RF model."
    )

real_index = class_positions[1]

y_probability_real = (
    random_forest.predict_proba(
        X_external
    )[:, real_index]
)


# ==================================================
# 14. EXTERNAL ACCURACY
# ==================================================
accuracy = accuracy_score(
    y_external,
    y_pred
)

balanced_accuracy = balanced_accuracy_score(
    y_external,
    y_pred
)

precision = precision_score(
    y_external,
    y_pred,
    average="macro",
    zero_division=0
)

recall = recall_score(
    y_external,
    y_pred,
    average="macro",
    zero_division=0
)

f1 = f1_score(
    y_external,
    y_pred,
    average="macro",
    zero_division=0
)

roc_auc = roc_auc_score(
    y_external,
    y_probability_real
)


# ==================================================
# 15. RESULTS
# ==================================================
print(
    "\n========== POLYGLOT EXTERNAL TEST =========="
)

print(f"External Accuracy:          {accuracy:.4f}")
print(f"External Balanced Accuracy: {balanced_accuracy:.4f}")
print(f"External Macro Precision:   {precision:.4f}")
print(f"External Macro Recall:      {recall:.4f}")
print(f"External Macro F1-score:    {f1:.4f}")
print(f"External ROC-AUC:           {roc_auc:.4f}")

print("\nClassification Report:")

print(
    classification_report(
        y_external,
        y_pred,
        target_names=["Fake", "Real"],
        digits=4,
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_external,
        y_pred
    )
)


# ==================================================
# 16. SAVE EXTERNAL RESULTS
# ==================================================
external_df["predicted_label"] = y_pred
external_df["probability_real"] = y_probability_real

external_df["correct"] = (
    external_df["label"]
    == external_df["predicted_label"]
)

external_df.to_csv(
    REPORT_DIR / "polyglot_external_predictions.csv",
    index=False,
    encoding="utf-8-sig"
)

metrics_df = pd.DataFrame([{
    "model": "Random Forest",
    "dataset": "PolyglotFakeFacts v2.0",
    "records": len(external_df),
    "accuracy": accuracy,
    "balanced_accuracy": balanced_accuracy,
    "macro_precision": precision,
    "macro_recall": recall,
    "macro_f1": f1,
    "roc_auc": roc_auc
}])

metrics_df.to_csv(
    REPORT_DIR / "polyglot_external_metrics.csv",
    index=False
)

print("\nReports saved successfully.")