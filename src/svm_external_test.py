from pathlib import Path
import re
import string
import unicodedata
import joblib  # pyrefly: ignore [missing-import]
import numpy as np  # pyrefly: ignore [missing-import]
import pandas as pd  # pyrefly: ignore [missing-import]

import nltk  # pyrefly: ignore [missing-import]
from nltk.tokenize import word_tokenize  # pyrefly: ignore [missing-import]
from nltk.corpus import stopwords  # pyrefly: ignore [missing-import]
from nltk.stem import WordNetLemmatizer  # pyrefly: ignore [missing-import]

from sklearn.pipeline import Pipeline  # pyrefly: ignore [missing-import]
from sklearn.metrics import (  # pyrefly: ignore [missing-import]
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


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

EXTERNAL_DIR = (
    BASE_DIR
    / "data"
    / "external_validation"
)

MODEL_FILE = (
    BASE_DIR
    / "models"
    / "svm_model.pkl"
)

REPORT_DIR = (
    BASE_DIR
    / "reports"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. FIND EXTERNAL DATASETS
# ============================================================

def find_external_file(name):

    for extension in [
        ".xlsx",
        ".csv",
        ".xls"
    ]:

        file_path = (
            EXTERNAL_DIR
            / f"{name}{extension}"
        )

        if file_path.exists():
            return file_path

    raise FileNotFoundError(
        f"\nCould not find {name} dataset inside:\n"
        f"{EXTERNAL_DIR}"
    )


FAKE_FILE = find_external_file("Fake")
REAL_FILE = find_external_file("Real")


# ============================================================
# 3. NLTK RESOURCES
# ============================================================

nltk_resources = [
    ("stopwords", "corpora/stopwords"),
    ("punkt", "tokenizers/punkt"),
    ("punkt_tab", "tokenizers/punkt_tab"),
    ("wordnet", "corpora/wordnet"),
    ("omw-1.4", "corpora/omw-1.4")
]


for resource, path in nltk_resources:

    try:
        nltk.data.find(path)

    except LookupError:

        try:
            nltk.download(
                resource,
                quiet=True
            )
        except Exception:
            pass


stop_words = set(
    stopwords.words("english")
)

lemmatizer = WordNetLemmatizer()


# ============================================================
# 4. STARTUP INFORMATION
# ============================================================

print("\n==========================================")
print("          SVM EXTERNAL VALIDATION")
print("==========================================")

print("Fake dataset:", FAKE_FILE)
print("Real dataset:", REAL_FILE)
print("SVM model:", MODEL_FILE)


if not MODEL_FILE.exists():

    raise FileNotFoundError(
        f"\nSVM model not found:\n"
        f"{MODEL_FILE}"
    )


# ============================================================
# 5. LOAD CSV / EXCEL
# ============================================================

def load_file_safe(path):

    suffix = path.suffix.lower()

    if suffix in {
        ".xlsx",
        ".xls"
    }:

        return pd.read_excel(
            path
        )


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
        f"Unable to read: {path}"
    )


# ============================================================
# 6. LOAD POLYGLOT DATA
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


TEXT_COLUMN = (
    "english translated version"
)


if TEXT_COLUMN not in fake_df.columns:

    raise ValueError(
        f"'{TEXT_COLUMN}' not found "
        "in Fake dataset."
    )


if TEXT_COLUMN not in real_df.columns:

    raise ValueError(
        f"'{TEXT_COLUMN}' not found "
        "in Real dataset."
    )


# ============================================================
# 7. CREATE LABELS
# ============================================================

# Project labels:
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
    "\nOriginal records:"
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
# 8. DATE HANDLING
# ============================================================

if "news date" in external_df.columns:

    external_df[
        "parsed_news_date"
    ] = pd.to_datetime(
        external_df["news date"],
        errors="coerce"
    )

else:

    external_df[
        "parsed_news_date"
    ] = pd.NaT


if "gathering date" in external_df.columns:

    external_df[
        "parsed_gathering_date"
    ] = pd.to_datetime(
        external_df["gathering date"],
        errors="coerce"
    )

else:

    external_df[
        "parsed_gathering_date"
    ] = pd.NaT


# Prefer news date.
# Gathering date used only as fallback.

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
# 9. KEEP NEWS FROM 2020+
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
    "\nPost-2020 records:",
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
# 10. USE ENGLISH TRANSLATION
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
# 11. TEXT PREPROCESSING
# ============================================================

def preprocess_text(text):

    if pd.isna(text):
        return ""

    text = unicodedata.normalize(
        "NFKC",
        str(text)
    )

    text = text.lower()


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


    # Reuters / AP / AFP dateline
    text = re.sub(
        r"^\s*[a-z][a-z\s,./'-]{1,80}"
        r"\s+\((reuters|ap|afp)\)"
        r"\s*[-–—:]?\s*",
        " ",
        text
    )


    # Reporting credits
    text = re.sub(
        r"\b(reporting|writing|editing|additional reporting)"
        r"\s+by\s+[^.;]{1,120}[.;]?",
        " ",
        text
    )


    # Control characters
    text = re.sub(
        r"[\x00-\x1f\x7f-\x9f]",
        " ",
        text
    )


    # Remove punctuation
    text = text.translate(
        str.maketrans(
            "",
            "",
            string.punctuation
        )
    )


    # Remove digits
    text = re.sub(
        r"\d+",
        "",
        text
    )


    # Tokenize
    tokens = word_tokenize(
        text
    )


    # Stopwords + lemmatization
    tokens = [
        lemmatizer.lemmatize(word)

        for word in tokens

        if (
            word not in stop_words
            and
            len(word) > 1
        )
    ]


    text = " ".join(
        tokens
    )


    # Normalize whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()


    return text


print(
    "\nPreprocessing external news..."
)


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
# 12. REMOVE INVALID / SHORT ARTICLES
# ============================================================

external_df[
    "word_count"
] = (
    external_df[
        "fused_text"
    ]
    .str.split()
    .str.len()
)


external_df = external_df[
    external_df[
        "word_count"
    ] >= 20
].copy()


# ============================================================
# 13. REMOVE DUPLICATES
# ============================================================

external_df = (
    external_df
    .drop_duplicates(
        subset=[
            "fused_text"
        ],
        keep="first"
    )
    .copy()
)


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
        "No external records remain."
    )


if external_df[
    "label"
].nunique() != 2:

    raise ValueError(
        "External dataset must contain "
        "Fake and Real classes."
    )


print(
    "\nFinal external records:",
    len(external_df)
)


print(
    "\nClass distribution:"
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
# 14. LOAD SVM MODEL
# ============================================================

print(
    "\nLoading SVM model..."
)


model_package = joblib.load(
    MODEL_FILE
)


# ------------------------------------------------------------
# Support:
#
# 1. Direct saved SVM
# 2. Dictionary package
# 3. sklearn Pipeline
# ------------------------------------------------------------

vectorizer = None


if isinstance(
    model_package,
    dict
):

    svm_model = (
        model_package.get(
            "model"
        )
    )


    if svm_model is None:

        svm_model = (
            model_package.get(
                "svm"
            )
        )


    if svm_model is None:

        svm_model = (
            model_package.get(
                "classifier"
            )
        )


    if svm_model is None:

        raise ValueError(
            "SVM package does not contain "
            "a supported model key."
        )


    # Some members may save their vectorizer
    # inside the same dictionary.

    vectorizer = (
        model_package.get(
            "vectorizer"
        )
    )


    if vectorizer is None:

        vectorizer = (
            model_package.get(
                "tfidf"
            )
        )


else:

    svm_model = model_package


print(
    "Loaded object type:",
    type(svm_model).__name__
)


# ============================================================
# 15. CHECK IF SAVED MODEL IS A PIPELINE
# ============================================================

is_pipeline = isinstance(
    svm_model,
    Pipeline
)


print(
    "Saved sklearn Pipeline:",
    is_pipeline
)


# ============================================================
# 16. FIND VECTORIZER IF NEEDED
# ============================================================

if not is_pipeline:

    # --------------------------------------------------------
    # If no vectorizer was stored inside SVM package,
    # look for a saved vectorizer.
    # --------------------------------------------------------

    if vectorizer is None:

        possible_vectorizers = [

            BASE_DIR
            / "models"
            / "svm_tfidf_vectorizer.pkl",

            BASE_DIR
            / "models"
            / "svm_vectorizer.pkl",

            BASE_DIR
            / "models"
            / "tfidf_vectorizer.pkl",
            
            BASE_DIR
            / "data"
            / "features"
            / "tfidf_vectorizer.pkl"
        ]


        for vectorizer_file in possible_vectorizers:

            if vectorizer_file.exists():

                print(
                    "Found vectorizer:",
                    vectorizer_file
                )

                vectorizer = joblib.load(
                    vectorizer_file
                )

                break


# ============================================================
# 17. PREPARE EXTERNAL FEATURES
# ============================================================

y_external = (
    external_df[
        "label"
    ]
    .to_numpy()
)


# ------------------------------------------------------------
# CASE A:
# Complete Pipeline
#
# e.g.
# Pipeline([
#     ("tfidf", TfidfVectorizer(...)),
#     ("svm", LinearSVC(...))
# ])
#
# Pipeline handles transformation internally.
# ------------------------------------------------------------

if is_pipeline:

    X_external = (
        external_df[
            "fused_text"
        ]
    )


    print(
        "\nSVM Pipeline detected."
    )


    print(
        "Text will be passed directly "
        "to the saved pipeline."
    )


# ------------------------------------------------------------
# CASE B:
# Standalone SVM
# ------------------------------------------------------------

else:

    if vectorizer is None:

        raise FileNotFoundError(
            "\nSVM model is standalone, but its "
            "fitted vectorizer was not found.\n\n"
            "You need the EXACT fitted vectorizer "
            "used during SVM training.\n\n"
            "Expected something like:\n"
            "models/svm_tfidf_vectorizer.pkl\n"
            "or a model package containing "
            "the vectorizer."
        )


    if not hasattr(
        vectorizer,
        "transform"
    ):

        raise TypeError(
            "Loaded SVM vectorizer does not "
            "support transform()."
        )


    X_external = (
        vectorizer.transform(
            external_df[
                "fused_text"
            ]
        )
    )


    print(
        "\nExternal feature shape:",
        X_external.shape
    )


    # Feature compatibility
    if hasattr(
        svm_model,
        "n_features_in_"
    ):

        if (
            svm_model.n_features_in_
            != X_external.shape[1]
        ):

            raise ValueError(
                "\nSVM / vectorizer mismatch!\n"
                f"SVM expects "
                f"{svm_model.n_features_in_} features.\n"
                f"Vectorizer produced "
                f"{X_external.shape[1]} features.\n\n"
                "Use the vectorizer from the "
                "same SVM training run."
            )


# ============================================================
# 18. RUN SVM PREDICTION
# ============================================================

print(
    "\nRunning SVM external predictions..."
)


y_pred = svm_model.predict(
    X_external
)


# ============================================================
# 19. CHECK LABELS
# ============================================================

unique_predictions = set(
    np.asarray(
        y_pred
    ).tolist()
)


if not unique_predictions.issubset(
    {
        0,
        1
    }
):

    raise ValueError(
        "\nSVM predictions are not using "
        "0 = Fake / 1 = Real.\n"
        f"Predicted labels found: "
        f"{unique_predictions}"
    )


# ============================================================
# 20. GET SCORE FOR ROC-AUC
# ============================================================

score_type = None

y_score_real = None


# ------------------------------------------------------------
# OPTION A:
# SVM trained with probability=True
# ------------------------------------------------------------

if hasattr(
    svm_model,
    "predict_proba"
):

    try:

        probabilities = (
            svm_model.predict_proba(
                X_external
            )
        )


        classes = list(
            svm_model.classes_
        )


        if 1 not in classes:

            raise ValueError(
                "Real class (1) missing."
            )


        real_index = (
            classes.index(1)
        )


        y_score_real = (
            probabilities[
                :,
                real_index
            ]
        )


        score_type = (
            "probability"
        )


    except Exception:

        y_score_real = None


# ------------------------------------------------------------
# OPTION B:
# LinearSVC / SVC decision scores
# ------------------------------------------------------------

if (
    y_score_real is None
    and
    hasattr(
        svm_model,
        "decision_function"
    )
):

    decision_scores = (
        svm_model.decision_function(
            X_external
        )
    )


    decision_scores = np.asarray(
        decision_scores
    )


    # Binary SVM normally returns shape (N,)
    if decision_scores.ndim == 1:

        classes = list(
            svm_model.classes_
        )


        # sklearn binary decision_function:
        # positive score corresponds to classes_[1]

        if classes[1] == 1:

            y_score_real = (
                decision_scores
            )

        else:

            y_score_real = (
                -decision_scores
            )


    # Some models return (N,2)
    elif (
        decision_scores.ndim == 2
        and
        decision_scores.shape[1] == 2
    ):

        classes = list(
            svm_model.classes_
        )


        real_index = (
            classes.index(1)
        )


        y_score_real = (
            decision_scores[
                :,
                real_index
            ]
        )


    score_type = (
        "decision_function"
    )


# ============================================================
# 21. METRICS
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


mcc = matthews_corrcoef(
    y_external,
    y_pred
)


if y_score_real is not None:

    roc_auc = roc_auc_score(
        y_external,
        y_score_real
    )

else:

    roc_auc = float(
        "nan"
    )


# ============================================================
# 22. CLASSIFICATION REPORT
# ============================================================

report_text = classification_report(
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


report_dict = classification_report(
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
    output_dict=True,
    zero_division=0
)


# ============================================================
# 23. CONFUSION MATRIX
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
# 24. DISPLAY RESULTS
# ============================================================

print(
    "\n=========================================="
)

print(
    "          SVM POLYGLOT EXTERNAL TEST"
)

print(
    "=========================================="
)


print(
    f"External records:          "
    f"{len(y_external)}"
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
    "ROC-AUC score source:",
    score_type
)


print(
    "\nClassification Report:"
)


print(
    report_text
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
# 25. SAVE PREDICTIONS
# ============================================================

external_df[
    "predicted_label"
] = y_pred


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


if y_score_real is not None:

    external_df[
        "real_class_score"
    ] = y_score_real


PREDICTIONS_FILE = (
    REPORT_DIR
    / "svm_polyglot_external_predictions.csv"
)


external_df.to_csv(
    PREDICTIONS_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 26. SAVE METRICS
# ============================================================

metrics_df = pd.DataFrame([{

    "model":
        type(svm_model).__name__,

    "dataset":
        "PolyglotFakeFacts v2.0",

    "records":
        len(y_external),

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
        mcc,

    "roc_auc_score_type":
        score_type,

    "saved_pipeline":
        is_pipeline

}])


METRICS_FILE = (
    REPORT_DIR
    / "svm_polyglot_external_metrics.csv"
)


metrics_df.to_csv(
    METRICS_FILE,
    index=False
)


# ============================================================
# 27. SAVE CONFUSION MATRIX
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
    / "svm_polyglot_confusion_matrix.csv"
)


confusion_df.to_csv(
    CONFUSION_FILE
)


# ============================================================
# 28. SAVE CLASSIFICATION REPORT
# ============================================================

REPORT_FILE = (
    REPORT_DIR
    / "svm_polyglot_classification_report.csv"
)


pd.DataFrame(
    report_dict
).transpose().to_csv(
    REPORT_FILE
)


# ============================================================
# 29. FINAL MESSAGE
# ============================================================

print(
    "\n=========================================="
)

print(
    "          SVM EXTERNAL TEST COMPLETE"
)

print(
    "=========================================="
)


print(
    "\nSVM external accuracy:"
)

print(
    f"{accuracy * 100:.2f}%"
)


print(
    "\nSVM external Macro F1:"
)

print(
    f"{macro_f1:.4f}"
)


print(
    "\nSVM external ROC-AUC:"
)

print(
    f"{roc_auc:.4f}"
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
    REPORT_FILE
)


print(
    "\nIMPORTANT:"
)

print(
    "PolyglotFakeFacts was used for "
    "external evaluation only."
)

print(
    "Do not fit a new vectorizer or "
    "retrain/tune the SVM using this "
    "external test dataset."
)

print(
    "=========================================="
)