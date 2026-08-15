from pathlib import Path
import re
import unicodedata
import string

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure NLTK resources are available
nltk_resources = [
    ('stopwords', 'corpora/stopwords'),
    ('punkt', 'tokenizers/punkt'),
    ('punkt_tab', 'tokenizers/punkt_tab'),
    ('wordnet', 'corpora/wordnet'),
    ('omw-1.4', 'corpora/omw-1.4')
]

for resource, path in nltk_resources:
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(resource, quiet=True)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

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
    confusion_matrix,
    matthews_corrcoef
)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------
# External PolyglotFakeFacts files
# ------------------------------------------------------------

def get_validation_file(
    base_dir,
    name
):

    folder = (
        base_dir
        / "data"
        / "external_validation"
    )

    for extension in [
        ".xlsx",
        ".csv",
        ".xls"
    ]:

        file_path = (
            folder
            / f"{name}{extension}"
        )

        if file_path.exists():
            return file_path


    return (
        folder
        / f"{name}.csv"
    )


FAKE_FILE = get_validation_file(
    BASE_DIR,
    "Fake"
)

REAL_FILE = get_validation_file(
    BASE_DIR,
    "Real"
)


# ============================================================
# IMPORTANT:
#
# Change these two paths if your member used
# different filenames.
# ============================================================

MODEL_FILE = (
    BASE_DIR
    / "models"
    / "logistic_regression.pkl"
)


# If Logistic Regression was trained using the SAME shared
# TF-IDF vectorizer, this is correct:
TFIDF_FILE = (
    BASE_DIR
    / "models"
    / "tfidf_vectorizer.pkl"
)


# If the member has their own vectorizer instead, use:
#
# TFIDF_FILE = (
#     BASE_DIR
#     / "models"
#     / "logistic_regression_tfidf_vectorizer.pkl"
# )


REPORT_DIR = (
    BASE_DIR
    / "reports"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. CHECK REQUIRED FILES
# ============================================================

print("\n==========================================")
print(" LOGISTIC REGRESSION EXTERNAL VALIDATION")
print("==========================================")


print(
    "Fake dataset:",
    FAKE_FILE
)

print(
    "Real dataset:",
    REAL_FILE
)

print(
    "Logistic Regression model:",
    MODEL_FILE
)

print(
    "TF-IDF vectorizer:",
    TFIDF_FILE
)


for file_path in [

    FAKE_FILE,

    REAL_FILE,

    MODEL_FILE,

    TFIDF_FILE

]:

    if not file_path.exists():

        raise FileNotFoundError(
            f"\nRequired file not found:\n"
            f"{file_path}\n\n"
            "Check the filename and folder."
        )


# ============================================================
# 3. LOAD CSV / EXCEL SAFELY
# ============================================================

def load_file_safe(
    path
):

    suffix = (
        path.suffix.lower()
    )


    # --------------------------------------------------------
    # Excel
    # --------------------------------------------------------

    if suffix in {
        ".xlsx",
        ".xls"
    }:

        return pd.read_excel(
            path
        )


    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    for encoding in [

        "utf-8-sig",

        "utf-8",

        "cp1252",

        "latin1"

    ]:

        try:

            return pd.read_csv(
                path,
                encoding=encoding,
                low_memory=False
            )

        except UnicodeDecodeError:

            continue


    raise ValueError(
        f"Unable to read dataset: "
        f"{path}"
    )


# ============================================================
# 4. LOAD EXTERNAL DATASETS
# ============================================================

print(
    "\nLoading PolyglotFakeFacts..."
)


fake_df = load_file_safe(
    FAKE_FILE
)

real_df = load_file_safe(
    REAL_FILE
)


# Standardize column names
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


print(
    "Raw Fake records:",
    len(fake_df)
)

print(
    "Raw Real records:",
    len(real_df)
)


# ============================================================
# 5. CHECK TEXT COLUMN
# ============================================================

TEXT_COLUMN = (
    "english translated version"
)


if TEXT_COLUMN not in fake_df.columns:

    raise ValueError(
        f"'{TEXT_COLUMN}' column "
        "not found in Fake dataset."
    )


if TEXT_COLUMN not in real_df.columns:

    raise ValueError(
        f"'{TEXT_COLUMN}' column "
        "not found in Real dataset."
    )


# ============================================================
# 6. CREATE LABELS
# ============================================================

# Project label mapping:
#
# 0 = Fake
# 1 = Real

fake_df["label"] = 0

real_df["label"] = 1


external_df = pd.concat(

    [
        fake_df,
        real_df
    ],

    ignore_index=True
)


print(
    "\nOriginal external records:"
)

print(
    "Fake:",
    len(fake_df)
)

print(
    "Real:",
    len(real_df)
)

print(
    "Total:",
    len(external_df)
)


# ============================================================
# 7. DATE HANDLING
# ============================================================

# ------------------------------------------------------------
# Actual news publication date
# ------------------------------------------------------------

if (
    "news date"
    in external_df.columns
):

    external_df[
        "parsed_news_date"
    ] = pd.to_datetime(

        external_df[
            "news date"
        ],

        errors="coerce"
    )

else:

    external_df[
        "parsed_news_date"
    ] = pd.NaT


# ------------------------------------------------------------
# Dataset gathering/collection date
# ------------------------------------------------------------

if (
    "gathering date"
    in external_df.columns
):

    external_df[
        "parsed_gathering_date"
    ] = pd.to_datetime(

        external_df[
            "gathering date"
        ],

        errors="coerce"
    )

else:

    external_df[
        "parsed_gathering_date"
    ] = pd.NaT


# Prefer actual publication date.
# Fall back to gathering date when unavailable.

external_df[
    "validation_date"
] = (

    external_df[
        "parsed_news_date"
    ]

    .fillna(

        external_df[
            "parsed_gathering_date"
        ]
    )
)


# ============================================================
# 8. KEEP 2020+ EXTERNAL NEWS
# ============================================================

external_df = external_df[

    external_df[
        "validation_date"
    ]

    >=

    pd.Timestamp(
        "2020-01-01"
    )

].copy()


print(
    "\nPost-2020 external records:",
    len(external_df)
)


print(
    "Date range:",
    external_df[
        "validation_date"
    ].min(),
    "to",
    external_df[
        "validation_date"
    ].max()
)


# ============================================================
# 9. USE ENGLISH TRANSLATED VERSION
# ============================================================

external_df[
    "fused_text"
] = (

    external_df[
        TEXT_COLUMN
    ]

    .fillna("")

    .astype(str)
)


# ============================================================
# 10. BASIC EXTERNAL TEXT PREPROCESSING
# ============================================================

def preprocess_text(
    text
):

    if pd.isna(text):
        return ""

    text = unicodedata.normalize(
        "NFKC",
        str(text)
    )

    text = str(text).lower()

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
        text
    )

    # Remove emails
    text = re.sub(
        r"\S+@\S+\.\S+",
        " ",
        text
    )

    # Remove Reuters/AP/AFP datelines
    text = re.sub(
        r"^\s*[a-z][a-z\s,./'-]{1,80}\s+\((reuters|ap|afp)\)\s*[-–—:]?\s*",
        " ",
        text
    )

    # Remove reporting credits
    text = re.sub(
        r"\b(reporting|writing|editing|additional reporting)\s+by\s+[^.;]{1,120}[.;]?",
        " ",
        text
    )

    # Remove control characters
    text = re.sub(
        r"[\x00-\x1f\x7f-\x9f]",
        " ",
        text
    )
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove numbers/digits
    text = re.sub(r'\d+', '', text)

    # Tokenization
    tokens = word_tokenize(text)

    # Stop Word Removal & Lemmatization
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 1]

    text = " ".join(tokens)

    # Normalize whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


external_df[
    "fused_text"
] = (

    external_df[
        "fused_text"
    ]

    .apply(
        preprocess_text
    )
)


# ============================================================
# 11. REMOVE INVALID / SHORT RECORDS
# ============================================================

external_df = external_df[

    external_df[
        "fused_text"
    ]

    .str.split()

    .str.len()

    .ge(20)

].copy()


# ============================================================
# 12. REMOVE EXACT DUPLICATES
# ============================================================

external_df = (

    external_df

    .drop_duplicates(
        subset="fused_text",
        keep="first"
    )

    .copy()
)


# Shuffle deterministically
external_df = (

    external_df

    .sample(
        frac=1,
        random_state=42
    )

    .reset_index(
        drop=True
    )
)


if external_df.empty:

    raise ValueError(
        "No external records remain "
        "after filtering."
    )


if (
    external_df[
        "label"
    ].nunique()
    != 2
):

    raise ValueError(
        "External dataset must contain "
        "both Fake and Real classes."
    )


print(
    "\nFinal external records:",
    len(external_df)
)


print(
    "\nExternal class distribution:"
)


print(

    external_df[
        "label"
    ]

    .value_counts()

    .sort_index()

    .rename(
        index={
            0: "Fake",
            1: "Real"
        }
    )
)


# ============================================================
# 13. LOAD LOGISTIC REGRESSION MODEL
# ============================================================

print(
    "\nLoading Logistic Regression..."
)


model_package = joblib.load(
    MODEL_FILE
)


# ------------------------------------------------------------
# Supports:
#
# joblib.dump(model, ...)
#
# OR:
#
# joblib.dump({
#     "model": model,
#     "feature_names": ...
# }, ...)
# ------------------------------------------------------------

if isinstance(
    model_package,
    dict
):

    model = (
        model_package.get(
            "model"
        )
    )


    if model is None:

        # Optional support for another common key
        model = (
            model_package.get(
                "logistic_regression"
            )
        )


    if model is None:

        raise ValueError(
            "Logistic Regression model package "
            "does not contain a supported model key."
        )


    model_features = (
        model_package.get(
            "feature_names"
        )
    )


else:

    model = (
        model_package
    )

    model_features = None


print(
    "Loaded model type:",
    type(model).__name__
)


# ============================================================
# 14. CHECK MODEL TYPE
# ============================================================

if (
    type(model).__name__
    != "LogisticRegression"
):

    print(
        "WARNING:"
    )

    print(
        "Loaded object is not named "
        "'LogisticRegression'."
    )

    print(
        "Actual type:",
        type(model).__name__
    )


# ============================================================
# 15. LOAD FITTED TF-IDF VECTORIZER
# ============================================================

print(
    "\nLoading Logistic Regression "
    "TF-IDF vectorizer..."
)


tfidf = joblib.load(
    TFIDF_FILE
)


tfidf_features = list(

    tfidf
    .get_feature_names_out()
)


print(
    "TF-IDF features:",
    len(tfidf_features)
)


# ============================================================
# 16. CHECK MODEL / VECTORIZER COMPATIBILITY
# ============================================================

# Feature count check
if hasattr(
    model,
    "n_features_in_"
):

    if (
        model.n_features_in_
        != len(tfidf_features)
    ):

        raise ValueError(

            "\nLOGISTIC REGRESSION / TF-IDF MISMATCH!\n"

            f"Logistic Regression expects "
            f"{model.n_features_in_} features.\n"

            f"TF-IDF produces "
            f"{len(tfidf_features)} features.\n\n"

            "Use the fitted vectorizer from the "
            "same training run as this "
            "Logistic Regression model."
        )


# Exact feature-name check when model package
# contains saved feature names.
if model_features is not None:

    if (
        list(model_features)
        != tfidf_features
    ):

        raise ValueError(

            "Logistic Regression saved feature names "
            "do not match the TF-IDF vectorizer."
        )


print(
    "Model / TF-IDF compatibility: OK"
)


# ============================================================
# 17. TRANSFORM EXTERNAL DATA
# ============================================================

# IMPORTANT:
#
# transform() only.
#
# NEVER:
#
# fit()
# fit_transform()
#
# on the external test dataset.

X_external = tfidf.transform(

    external_df[
        "fused_text"
    ]
)


y_external = (

    external_df[
        "label"
    ]

    .to_numpy()
)


print(
    "\nExternal TF-IDF shape:",
    X_external.shape
)


empty_vectors = int(

    (
        X_external
        .getnnz(axis=1)
        == 0
    ).sum()
)


print(
    "Empty external TF-IDF vectors:",
    empty_vectors
)


# ============================================================
# 18. LOGISTIC REGRESSION PREDICTIONS
# ============================================================

print(
    "\nRunning Logistic Regression predictions..."
)


y_pred = model.predict(
    X_external
)


# ============================================================
# 19. PREDICTION PROBABILITIES
# ============================================================

if not hasattr(
    model,
    "predict_proba"
):

    raise ValueError(
        "The loaded Logistic Regression model "
        "does not support predict_proba()."
    )


classes = list(
    model.classes_
)


if (
    0 not in classes
    or
    1 not in classes
):

    raise ValueError(
        "Logistic Regression classes must contain "
        "0 = Fake and 1 = Real."
    )


real_class_index = (
    classes.index(1)
)


fake_class_index = (
    classes.index(0)
)


probabilities = (
    model.predict_proba(
        X_external
    )
)


y_probability_fake = (
    probabilities[
        :,
        fake_class_index
    ]
)


y_probability_real = (
    probabilities[
        :,
        real_class_index
    ]
)


# ============================================================
# 20. EXTERNAL EVALUATION METRICS
# ============================================================

accuracy = accuracy_score(
    y_external,
    y_pred
)


balanced_accuracy = (
    balanced_accuracy_score(
        y_external,
        y_pred
    )
)


macro_precision = (
    precision_score(
        y_external,
        y_pred,
        average="macro",
        zero_division=0
    )
)


macro_recall = (
    recall_score(
        y_external,
        y_pred,
        average="macro",
        zero_division=0
    )
)


macro_f1 = (
    f1_score(
        y_external,
        y_pred,
        average="macro",
        zero_division=0
    )
)


weighted_f1 = (
    f1_score(
        y_external,
        y_pred,
        average="weighted",
        zero_division=0
    )
)


roc_auc = (
    roc_auc_score(
        y_external,
        y_probability_real
    )
)


mcc = (
    matthews_corrcoef(
        y_external,
        y_pred
    )
)


# ============================================================
# 21. CLASSIFICATION REPORT
# ============================================================

classification_report_text = (
    classification_report(

        y_external,

        y_pred,

        labels=[
            0,
            1
        ],

        target_names=[
            "Fake",
            "Real"
        ],

        digits=4,

        zero_division=0
    )
)


# ============================================================
# 22. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(

    y_external,

    y_pred,

    labels=[
        0,
        1
    ]
)


# ============================================================
# 23. DISPLAY RESULTS
# ============================================================

print(
    "\n=========================================="
)

print(
    " LOGISTIC REGRESSION POLYGLOT EXTERNAL TEST"
)

print(
    "=========================================="
)


print(
    f"External records:          "
    f"{len(external_df)}"
)


print(
    f"External Accuracy:         "
    f"{accuracy:.4f} "
    f"({accuracy * 100:.2f}%)"
)


print(
    f"Balanced Accuracy:         "
    f"{balanced_accuracy:.4f}"
)


print(
    f"Macro Precision:           "
    f"{macro_precision:.4f}"
)


print(
    f"Macro Recall:              "
    f"{macro_recall:.4f}"
)


print(
    f"Macro F1-score:            "
    f"{macro_f1:.4f}"
)


print(
    f"Weighted F1-score:         "
    f"{weighted_f1:.4f}"
)


print(
    f"ROC-AUC:                   "
    f"{roc_auc:.4f}"
)


print(
    f"MCC:                       "
    f"{mcc:.4f}"
)


print(
    "\nClassification Report:"
)


print(
    classification_report_text
)


print(
    "Confusion Matrix:"
)


print(
    cm
)


print(
    "\nRows    = Actual [Fake, Real]"
)

print(
    "Columns = Predicted [Fake, Real]"
)


# ============================================================
# 24. SAVE EXTERNAL PREDICTIONS
# ============================================================

external_df[
    "predicted_label"
] = y_pred


external_df[
    "probability_fake"
] = y_probability_fake


external_df[
    "probability_real"
] = y_probability_real


external_df[
    "correct"
] = (

    external_df[
        "label"
    ]

    ==

    external_df[
        "predicted_label"
    ]
)


PREDICTIONS_FILE = (

    REPORT_DIR
    / "logistic_regression_polyglot_external_predictions.csv"
)


external_df.to_csv(

    PREDICTIONS_FILE,

    index=False,

    encoding="utf-8-sig"
)


# ============================================================
# 25. SAVE EXTERNAL METRICS
# ============================================================

metrics_df = pd.DataFrame([{

    "model":
        "Logistic Regression",

    "dataset":
        "PolyglotFakeFacts v2.0",

    "records":
        len(external_df),

    "fake_records":
        int(
            (
                y_external == 0
            ).sum()
        ),

    "real_records":
        int(
            (
                y_external == 1
            ).sum()
        ),

    "accuracy":
        accuracy,

    "accuracy_percent":
        accuracy * 100,

    "balanced_accuracy":
        balanced_accuracy,

    "macro_precision":
        macro_precision,

    "macro_recall":
        macro_recall,

    "macro_f1":
        macro_f1,

    "weighted_f1":
        weighted_f1,

    "roc_auc":
        roc_auc,

    "mcc":
        mcc

}])


METRICS_FILE = (

    REPORT_DIR
    / "logistic_regression_polyglot_external_metrics.csv"
)


metrics_df.to_csv(

    METRICS_FILE,

    index=False
)


# ============================================================
# 26. SAVE CONFUSION MATRIX
# ============================================================

confusion_df = pd.DataFrame(

    cm,

    index=[
        "Actual_Fake",
        "Actual_Real"
    ],

    columns=[
        "Predicted_Fake",
        "Predicted_Real"
    ]
)


CONFUSION_FILE = (

    REPORT_DIR
    / "logistic_regression_polyglot_confusion_matrix.csv"
)


confusion_df.to_csv(
    CONFUSION_FILE
)


# ============================================================
# 27. FINAL MESSAGE
# ============================================================

print(
    "\n=========================================="
)

print(
    " EXTERNAL VALIDATION COMPLETE"
)

print(
    "=========================================="
)


print(
    "\nLogistic Regression external accuracy:"
)

print(
    f"{accuracy * 100:.2f}%"
)


print(
    "\nReports saved:"
)

print(
    METRICS_FILE
)

print(
    PREDICTIONS_FILE
)

print(
    CONFUSION_FILE
)


print(
    "\nIMPORTANT:"
)

print(
    "PolyglotFakeFacts was used for "
    "evaluation only."
)

print(
    "Do not fit the TF-IDF vectorizer "
    "or retrain Logistic Regression "
    "using this external dataset."
)

print(
    "=========================================="
)