import sys
import os
import subprocess

try:
    import tensorflow as tf
except ModuleNotFoundError:
    if os.environ.get("_TF_RELAUNCH_ATTEMPTED"):
        print("Error: TensorFlow is not installed in the default python environment either. Please install it.")
        sys.exit(1)

    print("TensorFlow not found in the current Python environment.")
    print("Re-launching script using Python 3.12...")
    env = os.environ.copy()
    env["_TF_RELAUNCH_ATTEMPTED"] = "1"
    
    # Try the known working Python 3.12 executable
    python_exe = r"C:\Users\User\AppData\Local\Microsoft\WindowsApps\python.exe"
    if not os.path.exists(python_exe):
        python_exe = "py"
        args = ["-3.12", sys.argv[0]] + sys.argv[1:]
    else:
        args = [python_exe, sys.argv[0]] + sys.argv[1:]
        
    sys.exit(subprocess.call(args, env=env))

from pathlib import Path
import pickle
import re
import string
import unicodedata

import nltk
import numpy as np
import pandas as pd

from tensorflow import keras
pad_sequences = tf.keras.preprocessing.sequence.pad_sequences

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

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
    / "lstm_model.keras"
)


TOKENIZER_FILE = (
    BASE_DIR
    / "models"
    / "lstm_tokenizer.pickle"
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
# 2. LSTM SETTINGS
# ============================================================

# Your project labels:
#
# 0 = Fake
# 1 = Real

FAKE_LABEL = 0
REAL_LABEL = 1


# Binary classification threshold
THRESHOLD = 0.50


# Prediction batch size
BATCH_SIZE = 64


# ------------------------------------------------------------
# IMPORTANT
# ------------------------------------------------------------
#
# pad_sequences() defaults are:
#
# padding="pre"
# truncating="pre"
#
# Keep these values ONLY if the LSTM training code also
# used pre-padding / pre-truncation.
#
# If training used "post", change both to "post".
#
# They MUST match LSTM training.
# ------------------------------------------------------------

PADDING_TYPE = "post"

TRUNCATING_TYPE = "post"


# ------------------------------------------------------------
# Usually this can be detected automatically from model shape.
#
# If model input shape is variable, put the EXACT max_length
# used during LSTM training here.
#
# Example:
#
# MANUAL_MAX_LENGTH = 300
#
# ------------------------------------------------------------

MANUAL_MAX_LENGTH = None


# ============================================================
# 3. NLTK RESOURCES
# ============================================================

nltk_resources = [

    (
        "stopwords",
        "corpora/stopwords"
    ),

    (
        "punkt",
        "tokenizers/punkt"
    ),

    (
        "punkt_tab",
        "tokenizers/punkt_tab"
    ),

    (
        "wordnet",
        "corpora/wordnet"
    ),

    (
        "omw-1.4",
        "corpora/omw-1.4"
    )
]


for resource, resource_path in nltk_resources:

    try:

        nltk.data.find(
            resource_path
        )

    except LookupError:

        try:

            nltk.download(
                resource,
                quiet=True
            )

        except Exception:

            # Some NLTK versions may not need
            # every optional resource.
            pass


stop_words = set(
    stopwords.words(
        "english"
    )
)


lemmatizer = (
    WordNetLemmatizer()
)


# ============================================================
# 4. FIND EXTERNAL DATA FILES
# ============================================================

def find_external_file(
    name
):

    extensions = [

        ".xlsx",

        ".csv",

        ".xls"
    ]


    for extension in extensions:

        file_path = (

            EXTERNAL_DIR
            / f"{name}{extension}"
        )


        if file_path.exists():

            return file_path


    raise FileNotFoundError(

        f"\nCould not find {name} external dataset.\n"

        f"Expected inside:\n"

        f"{EXTERNAL_DIR}"
    )


FAKE_FILE = find_external_file(
    "Fake"
)


REAL_FILE = find_external_file(
    "Real"
)


# ============================================================
# 5. CHECK REQUIRED FILES
# ============================================================

print(
    "\n=========================================="
)

print(
    "       LSTM EXTERNAL VALIDATION"
)

print(
    "=========================================="
)


print(
    "Fake dataset:",
    FAKE_FILE
)


print(
    "Real dataset:",
    REAL_FILE
)


print(
    "LSTM model:",
    MODEL_FILE
)


print(
    "LSTM tokenizer:",
    TOKENIZER_FILE
)


for required_file in [

    FAKE_FILE,

    REAL_FILE,

    MODEL_FILE,

    TOKENIZER_FILE

]:

    if not required_file.exists():

        raise FileNotFoundError(

            f"\nRequired file not found:\n"

            f"{required_file}"
        )


# ============================================================
# 6. GPU CHECK
# ============================================================

gpus = (
    tf.config
    .list_physical_devices(
        "GPU"
    )
)


print(
    "\nTensorFlow version:",
    tf.__version__
)


print(
    "GPU available:",
    bool(gpus)
)


if gpus:

    print(
        "GPU:",
        gpus[0].name
    )


    # Avoid TensorFlow allocating all GPU
    # memory immediately.
    try:

        for gpu in gpus:

            tf.config.experimental.set_memory_growth(
                gpu,
                True
            )

    except RuntimeError:

        pass


else:

    print(
        "LSTM inference will use CPU."
    )


# ============================================================
# 7. LOAD CSV / EXCEL SAFELY
# ============================================================

def load_file_safe(
    path
):

    suffix = (
        path.suffix.lower()
    )


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

        f"Unable to read dataset: "

        f"{path}"
    )


# ============================================================
# 8. LOAD POLYGLOT DATASET
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


print(
    "Raw Fake records:",
    len(fake_df)
)


print(
    "Raw Real records:",
    len(real_df)
)


# ============================================================
# 9. CHECK REQUIRED TEXT COLUMN
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
# 10. CREATE LABELS
# ============================================================

fake_df[
    "label"
] = FAKE_LABEL


real_df[
    "label"
] = REAL_LABEL


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
# 11. DATE HANDLING
# ============================================================

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


# Prefer actual article date.
# Fall back to gathering date.

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
# 12. KEEP 2020+ NEWS
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
# 13. USE ENGLISH TRANSLATED VERSION
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
# 14. LSTM TEXT PREPROCESSING
# ============================================================

def preprocess_text(
    text
):

    if pd.isna(text):

        return ""


    # Unicode normalization
    text = unicodedata.normalize(
        "NFKC",
        str(text)
    )


    # Lowercase
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


    # Remove Reuters / AP / AFP dateline
    text = re.sub(

        r"^\s*[a-z][a-z\s,./'-]{1,80}"
        r"\s+\((reuters|ap|afp)\)"
        r"\s*[-–—:]?\s*",

        " ",

        text
    )


    # Remove reporting credits
    text = re.sub(

        r"\b(reporting|writing|editing|additional reporting)"
        r"\s+by\s+[^.;]{1,120}[.;]?",

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
    text = text.translate(

        str.maketrans(
            "",
            "",
            string.punctuation
        )
    )


    # Remove numbers
    text = re.sub(
        r"\d+",
        "",
        text
    )


    # Tokenization
    tokens = word_tokenize(
        text
    )


    # Stopword removal + lemmatization
    tokens = [

        lemmatizer.lemmatize(
            word
        )

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
    "\nPreprocessing external text..."
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
# 15. REMOVE INVALID / SHORT TEXT
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
    ]

    >= 20

].copy()


# ============================================================
# 16. REMOVE DUPLICATES
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

        "No external records remain "

        "after preprocessing."
    )


if (
    external_df[
        "label"
    ].nunique()
    != 2
):

    raise ValueError(

        "Both Fake and Real records "

        "are required."
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
# 17. LOAD SAVED LSTM TOKENIZER
# ============================================================

print(
    "\nLoading saved LSTM tokenizer..."
)


with open(
    TOKENIZER_FILE,
    "rb"
) as tokenizer_file:

    tokenizer = pickle.load(
        tokenizer_file
    )


if not hasattr(
    tokenizer,
    "texts_to_sequences"
):

    raise TypeError(

        "lstm_tokenizer.pickle does not "

        "contain a compatible Keras tokenizer."
    )


print(
    "Tokenizer loaded successfully."
)


if hasattr(
    tokenizer,
    "word_index"
):

    print(
        "Tokenizer vocabulary size:",
        len(
            tokenizer.word_index
        )
    )


# ============================================================
# 18. LOAD SAVED LSTM MODEL
# ============================================================

print(
    "\nLoading LSTM model..."
)


lstm_model = (
    keras.models.load_model(

        MODEL_FILE,

        compile=False
    )
)


print(
    "LSTM model loaded successfully."
)


print(
    "Model input shape:",
    lstm_model.input_shape
)


print(
    "Model output shape:",
    lstm_model.output_shape
)


# ============================================================
# 19. DETERMINE TRAINING SEQUENCE LENGTH
# ============================================================

def determine_max_length(
    model
):

    input_shape = (
        model.input_shape
    )


    # Handle models with list input shapes.
    if isinstance(
        input_shape,
        list
    ):

        if len(input_shape) != 1:

            raise ValueError(

                "This script expects a single-text-input "

                "LSTM model."
            )


        input_shape = (
            input_shape[0]
        )


    detected_length = None


    if (
        input_shape is not None

        and

        len(input_shape) >= 2
    ):

        detected_length = (
            input_shape[-1]
        )


    # Use fixed model input length when available.
    if (
        detected_length
        is not None
    ):

        return int(
            detected_length
        )


    # Otherwise require the training value manually.
    if (
        MANUAL_MAX_LENGTH
        is not None
    ):

        return int(
            MANUAL_MAX_LENGTH
        )


    raise ValueError(

        "\nUnable to automatically determine the "

        "LSTM sequence length.\n\n"

        "Open the LSTM training code and find the "

        "max_length / maxlen value used with "

        "pad_sequences().\n\n"

        "Then set:\n\n"

        "MANUAL_MAX_LENGTH = <training value>\n"
    )


MAX_LENGTH = (
    determine_max_length(
        lstm_model
    )
)


print(
    "Sequence length:",
    MAX_LENGTH
)


print(
    "Padding:",
    PADDING_TYPE
)


print(
    "Truncating:",
    TRUNCATING_TYPE
)


# ============================================================
# 20. CONVERT TEXT TO TOKEN SEQUENCES
# ============================================================

print(
    "\nConverting external news to token sequences..."
)


sequences = (
    tokenizer.texts_to_sequences(

        external_df[
            "fused_text"
        ].tolist()
    )
)


# Check how many articles became empty
empty_sequences = sum(

    1

    for sequence in sequences

    if len(sequence) == 0
)


print(
    "Empty token sequences:",
    empty_sequences
)


# ============================================================
# 21. PAD SEQUENCES
# ============================================================

X_external = pad_sequences(

    sequences,

    maxlen=MAX_LENGTH,

    padding=PADDING_TYPE,

    truncating=TRUNCATING_TYPE
)


y_external = (

    external_df[
        "label"
    ]

    .to_numpy(
        dtype=np.int32
    )
)


print(
    "External LSTM input shape:",
    X_external.shape
)


print(
    "External labels shape:",
    y_external.shape
)


# ============================================================
# 22. RUN LSTM PREDICTION
# ============================================================

print(
    "\nRunning LSTM external predictions..."
)


raw_predictions = (
    lstm_model.predict(

        X_external,

        batch_size=BATCH_SIZE,

        verbose=1
    )
)


raw_predictions = np.asarray(
    raw_predictions
)


print(
    "\nRaw prediction shape:",
    raw_predictions.shape
)


# ============================================================
# 23. CONVERT MODEL OUTPUT TO REAL PROBABILITY
# ============================================================

def sigmoid(
    values
):

    values = np.clip(
        values,
        -50,
        50
    )


    return (

        1.0

        /

        (
            1.0
            +
            np.exp(
                -values
            )
        )
    )


def softmax(
    values
):

    values = (

        values

        -

        np.max(

            values,

            axis=1,

            keepdims=True
        )
    )


    exp_values = np.exp(
        values
    )


    return (

        exp_values

        /

        np.sum(

            exp_values,

            axis=1,

            keepdims=True
        )
    )


# ------------------------------------------------------------
# CASE A:
# Model output:
#
# (N,)
# ------------------------------------------------------------

if raw_predictions.ndim == 1:

    real_probability = (
        raw_predictions
        .astype(float)
    )


    # If output looks like logits rather than probabilities
    if (

        np.any(
            real_probability < 0
        )

        or

        np.any(
            real_probability > 1
        )
    ):

        real_probability = sigmoid(
            real_probability
        )


# ------------------------------------------------------------
# CASE B:
# Sigmoid output:
#
# (N, 1)
# ------------------------------------------------------------

elif (

    raw_predictions.ndim == 2

    and

    raw_predictions.shape[1] == 1
):

    real_probability = (

        raw_predictions[
            :,
            0
        ]

        .astype(float)
    )


    if (

        np.any(
            real_probability < 0
        )

        or

        np.any(
            real_probability > 1
        )
    ):

        real_probability = sigmoid(
            real_probability
        )


# ------------------------------------------------------------
# CASE C:
# Two-class softmax:
#
# (N, 2)
# ------------------------------------------------------------

elif (

    raw_predictions.ndim == 2

    and

    raw_predictions.shape[1] == 2
):

    probabilities = (

        raw_predictions

        .astype(float)
    )


    row_sums = (
        probabilities.sum(
            axis=1
        )
    )


    looks_like_probabilities = (

        np.all(
            probabilities >= 0
        )

        and

        np.all(
            probabilities <= 1
        )

        and

        np.allclose(

            row_sums,

            1.0,

            atol=1e-3
        )
    )


    if not looks_like_probabilities:

        probabilities = softmax(
            probabilities
        )


    # Project labels:
    #
    # column 0 = Fake
    # column 1 = Real

    real_probability = (

        probabilities[
            :,
            1
        ]
    )


# ------------------------------------------------------------
# Unsupported model output
# ------------------------------------------------------------

else:

    raise ValueError(

        "\nUnsupported LSTM output shape:\n"

        f"{raw_predictions.shape}\n\n"

        "Expected binary sigmoid output (N,1) "

        "or two-class output (N,2)."
    )


# ============================================================
# 24. CREATE FINAL CLASS PREDICTIONS
# ============================================================

y_pred = (

    real_probability

    >=

    THRESHOLD

).astype(
    np.int32
)


fake_probability = (

    1.0

    -

    real_probability
)


# ============================================================
# 25. CHECK OUTPUT LENGTH
# ============================================================

if (
    len(y_pred)
    != len(y_external)
):

    raise RuntimeError(

        "Prediction count does not match "

        "external label count."
    )


# ============================================================
# 26. EXTERNAL METRICS
# ============================================================

accuracy = (
    accuracy_score(

        y_external,

        y_pred
    )
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

        real_probability
    )
)


mcc = (
    matthews_corrcoef(

        y_external,

        y_pred
    )
)


# ============================================================
# 27. CLASSIFICATION REPORT
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


classification_report_dict = (
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

        output_dict=True,

        zero_division=0
    )
)


# ============================================================
# 28. CONFUSION MATRIX
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
# 29. DISPLAY EXTERNAL RESULTS
# ============================================================

print(
    "\n=========================================="
)

print(
    "        LSTM POLYGLOT EXTERNAL TEST"
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
# 30. SAVE EXTERNAL PREDICTIONS
# ============================================================

external_df[
    "predicted_label"
] = y_pred


external_df[
    "probability_fake"
] = fake_probability


external_df[
    "probability_real"
] = real_probability


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

    / "lstm_polyglot_external_predictions.csv"
)


external_df.to_csv(

    PREDICTIONS_FILE,

    index=False,

    encoding="utf-8-sig"
)


# ============================================================
# 31. SAVE METRICS
# ============================================================

metrics_df = pd.DataFrame([{

    "model":
        "LSTM",

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

    "sequence_length":
        MAX_LENGTH,

    "padding":
        PADDING_TYPE,

    "truncating":
        TRUNCATING_TYPE,

    "classification_threshold":
        THRESHOLD,

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

    / "lstm_polyglot_external_metrics.csv"
)


metrics_df.to_csv(

    METRICS_FILE,

    index=False
)


# ============================================================
# 32. SAVE CONFUSION MATRIX
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

    / "lstm_polyglot_confusion_matrix.csv"
)


confusion_df.to_csv(
    CONFUSION_FILE
)


# ============================================================
# 33. SAVE CLASSIFICATION REPORT
# ============================================================

CLASSIFICATION_REPORT_FILE = (

    REPORT_DIR

    / "lstm_polyglot_classification_report.csv"
)


pd.DataFrame(

    classification_report_dict

).transpose().to_csv(

    CLASSIFICATION_REPORT_FILE
)


# ============================================================
# 34. FINAL MESSAGE
# ============================================================

print(
    "\n=========================================="
)

print(
    "      LSTM EXTERNAL TEST COMPLETE"
)

print(
    "=========================================="
)


print(
    "\nLSTM external accuracy:"
)


print(
    f"{accuracy * 100:.2f}%"
)


print(
    "\nLSTM external Macro F1:"
)


print(
    f"{macro_f1:.4f}"
)


print(
    "\nLSTM external ROC-AUC:"
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
    CLASSIFICATION_REPORT_FILE
)


print(
    "\nIMPORTANT:"
)


print(
    "PolyglotFakeFacts was used only "
    "for external evaluation."
)


print(
    "Do not train or fine-tune the LSTM "
    "using this external test dataset."
)


print(
    "=========================================="
)