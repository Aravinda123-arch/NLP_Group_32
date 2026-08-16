from pathlib import Path
import pickle
import re
import string
import unicodedata

import numpy as np  # pyrefly: ignore [missing-import]
import pandas as pd  # pyrefly: ignore [missing-import]
import tensorflow as tf  # pyrefly: ignore [missing-import]

from tensorflow import keras  # pyrefly: ignore [missing-import]
from tensorflow.keras.utils import pad_sequences  # pyrefly: ignore [missing-import]

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

MODELS_DIR = (
    BASE_DIR
    / "models"
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
# 2. GRU MODEL FILE
# ============================================================

# Prefer best trained model.
BEST_MODEL_FILE = (
    MODELS_DIR
    / "best_gru_model.h5"
)

FALLBACK_MODEL_FILE = (
    MODELS_DIR
    / "gru_model.h5"
)


if BEST_MODEL_FILE.exists():

    MODEL_FILE = (
        BEST_MODEL_FILE
    )

elif FALLBACK_MODEL_FILE.exists():

    MODEL_FILE = (
        FALLBACK_MODEL_FILE
    )

else:

    raise FileNotFoundError(
        "\nNo GRU model found.\n\n"
        "Expected one of:\n"
        f"{BEST_MODEL_FILE}\n"
        f"{FALLBACK_MODEL_FILE}"
    )


# ============================================================
# 3. POSSIBLE TOKENIZER FILES
# ============================================================

TOKENIZER_CANDIDATES = [

    MODELS_DIR / "gru_tokenizer.pickle",

    MODELS_DIR / "gru_tokenizer.pkl",

    MODELS_DIR / "tokenizer.pickle",

    MODELS_DIR / "tokenizer.pkl",
    
    BASE_DIR / "data" / "features" / "tokenizer.pkl"
]


def find_tokenizer():

    for file_path in TOKENIZER_CANDIDATES:

        if file_path.exists():

            return file_path

    return None


TOKENIZER_FILE = (
    find_tokenizer()
)


# ============================================================
# 4. SETTINGS
# ============================================================

# Project label mapping
#
# 0 = Fake
# 1 = Real

FAKE_LABEL = 0
REAL_LABEL = 1


# Binary threshold
THRESHOLD = 0.50


# Prediction batch size
BATCH_SIZE = 64


# ------------------------------------------------------------
# IMPORTANT
#
# These MUST match the GRU training code if the model expects
# pre-tokenized integer sequences.
# ------------------------------------------------------------

PADDING_TYPE = "pre"

TRUNCATING_TYPE = "pre"


# Set this ONLY if sequence length cannot be detected
# automatically from model.input_shape.
#
# Example:
#
# MANUAL_MAX_LENGTH = 300

MANUAL_MAX_LENGTH = None


# ============================================================
# 5. FIND EXTERNAL DATASETS
# ============================================================

def find_external_file(
    name
):

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
        f"\nCould not find {name} external dataset "
        f"inside:\n{EXTERNAL_DIR}"
    )


FAKE_FILE = find_external_file(
    "Fake"
)

REAL_FILE = find_external_file(
    "Real"
)


# ============================================================
# 6. START INFORMATION
# ============================================================

print(
    "\n=========================================="
)

print(
    "        GRU EXTERNAL VALIDATION"
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
    "GRU model:",
    MODEL_FILE
)

print(
    "Tokenizer:",
    (
        TOKENIZER_FILE
        if TOKENIZER_FILE
        else "Not found"
    )
)


# ============================================================
# 7. GPU CHECK
# ============================================================

gpus = tf.config.list_physical_devices(
    "GPU"
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
        "GRU external testing will use CPU."
    )


# ============================================================
# 8. LOAD CSV / EXCEL SAFELY
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
        f"Unable to read file: {path}"
    )


# ============================================================
# 9. LOAD POLYGLOT DATA
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


print(
    "Raw Fake records:",
    len(fake_df)
)

print(
    "Raw Real records:",
    len(real_df)
)


# ============================================================
# 10. LABELS
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


# ============================================================
# 11. DATE HANDLING
# ============================================================

if "news date" in external_df.columns:

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


if "gathering date" in external_df.columns:

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
# 12. KEEP 2020+
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
# 13. EXTERNAL ENGLISH TEXT
# ============================================================

external_df[
    "raw_text"
] = (

    external_df[
        TEXT_COLUMN
    ]

    .fillna("")

    .astype(str)
)


# ============================================================
# 14. BASIC CLEANING
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


    text = text.lower()


    # HTML
    text = re.sub(
        r"<[^>]+>",
        " ",
        text
    )


    # URLs
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )


    # Emails
    text = re.sub(
        r"\S+@\S+\.\S+",
        " ",
        text
    )


    # Reuters / AP / AFP datelines
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


    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Keep these only if the member's GRU training also
    # removed punctuation and numbers.
    # --------------------------------------------------------

    text = text.translate(

        str.maketrans(
            "",
            "",
            string.punctuation
        )
    )


    text = re.sub(
        r"\d+",
        "",
        text
    )


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
        "raw_text"
    ]

    .apply(
        preprocess_text
    )
)


# ============================================================
# 15. REMOVE INVALID / SHORT ARTICLES
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
# 16. REMOVE EXACT DUPLICATES
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
        "both Fake and Real."
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
# 17. LOAD GRU MODEL
# ============================================================

print(
    "\nLoading GRU model..."
)


# compile=False is enough for prediction/evaluation.
# TensorFlow supports loading saved Keras models from a path.
gru_model = tf.keras.models.load_model(

    MODEL_FILE,

    compile=False
)


print(
    "GRU model loaded."
)


print(
    "Model input shape:",
    gru_model.input_shape
)


print(
    "Model output shape:",
    gru_model.output_shape
)


# ============================================================
# 18. DETECT MODEL INPUT TYPE
# ============================================================

model_input = (
    gru_model.inputs[0]
)


input_dtype = (
    model_input.dtype
)


print(
    "Model input dtype:",
    input_dtype
)


is_string_input = (
    input_dtype
    == tf.string
)


# ============================================================
# 19. PREPARE GRU INPUT
# ============================================================

if is_string_input:

    # ========================================================
    # CASE A:
    # MODEL ACCEPTS RAW STRING INPUT
    #
    # This usually means preprocessing/TextVectorization
    # exists inside the saved model.
    # ========================================================

    print(
        "\nGRU accepts raw text input."
    )

    print(
        "Using preprocessing stored "
        "inside the model."
    )


    X_external = np.asarray(

        external_df[
            "fused_text"
        ].tolist(),

        dtype=object
    )


else:

    # ========================================================
    # CASE B:
    # MODEL EXPECTS INTEGER TOKEN SEQUENCES
    # ========================================================

    print(
        "\nGRU expects token sequences."
    )


    if TOKENIZER_FILE is None:

        raise FileNotFoundError(

            "\nIMPORTANT: GRU TOKENIZER IS MISSING.\n\n"

            "The saved GRU model expects numeric token "
            "sequences, but no tokenizer was found.\n\n"

            "You MUST obtain the exact tokenizer used "
            "during GRU training.\n\n"

            "Ask the member to save it as, for example:\n\n"

            "models/gru_tokenizer.pickle\n\n"

            "Do NOT create a new tokenizer from the "
            "external dataset because that would produce "
            "incorrect inputs and invalid accuracy."
        )


    # --------------------------------------------------------
    # Load saved tokenizer
    # --------------------------------------------------------

    print(
        "Loading GRU tokenizer:",
        TOKENIZER_FILE
    )


    with open(
        TOKENIZER_FILE,
        "rb"
    ) as file:

        tokenizer = pickle.load(
            file
        )


    if not hasattr(
        tokenizer,
        "texts_to_sequences"
    ):

        raise TypeError(
            "Saved GRU tokenizer does not support "
            "texts_to_sequences()."
        )


    # --------------------------------------------------------
    # Determine sequence length
    # --------------------------------------------------------

    input_shape = (
        gru_model.input_shape
    )


    if isinstance(
        input_shape,
        list
    ):

        input_shape = (
            input_shape[0]
        )


    detected_max_length = None


    if (
        input_shape is not None

        and

        len(input_shape) >= 2

        and

        input_shape[-1] is not None
    ):

        detected_max_length = int(
            input_shape[-1]
        )


    if detected_max_length is not None:

        MAX_LENGTH = (
            detected_max_length
        )


    elif MANUAL_MAX_LENGTH is not None:

        MAX_LENGTH = int(
            MANUAL_MAX_LENGTH
        )


    else:

        raise ValueError(

            "\nCould not determine GRU sequence length.\n\n"

            "Find the maxlen used in the member's "
            "GRU training code and set:\n\n"

            "MANUAL_MAX_LENGTH = <training maxlen>"
        )


    print(
        "GRU sequence length:",
        MAX_LENGTH
    )


    # --------------------------------------------------------
    # Convert text to integer token IDs
    # --------------------------------------------------------

    sequences = (
        tokenizer.texts_to_sequences(

            external_df[
                "fused_text"
            ].tolist()
        )
    )


    empty_sequences = sum(

        len(sequence) == 0

        for sequence in sequences
    )


    print(
        "Empty token sequences:",
        empty_sequences
    )


    # --------------------------------------------------------
    # Padding
    #
    # Must match GRU training.
    # --------------------------------------------------------

    X_external = pad_sequences(

        sequences,

        maxlen=MAX_LENGTH,

        padding=PADDING_TYPE,

        truncating=TRUNCATING_TYPE,

        dtype="int32"
    )


    print(
        "External GRU input shape:",
        X_external.shape
    )


# ============================================================
# 20. LABELS
# ============================================================

y_external = (

    external_df[
        "label"
    ]

    .to_numpy(
        dtype=np.int32
    )
)


# ============================================================
# 21. RUN GRU PREDICTION
# ============================================================

print(
    "\nRunning GRU external predictions..."
)


raw_predictions = (
    gru_model.predict(

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
# 22. CONVERT OUTPUT TO REAL CLASS SCORE
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

    shifted = (

        values

        -

        np.max(
            values,
            axis=1,
            keepdims=True
        )
    )


    exp_values = np.exp(
        shifted
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
# Binary output: (N,)
# ------------------------------------------------------------

if raw_predictions.ndim == 1:

    probability_real = (
        raw_predictions.astype(float)
    )


    if (

        np.any(
            probability_real < 0
        )

        or

        np.any(
            probability_real > 1
        )
    ):

        probability_real = sigmoid(
            probability_real
        )


# ------------------------------------------------------------
# Binary sigmoid output: (N, 1)
# ------------------------------------------------------------

elif (

    raw_predictions.ndim == 2

    and

    raw_predictions.shape[1] == 1
):

    probability_real = (

        raw_predictions[
            :,
            0
        ]

        .astype(float)
    )


    if (

        np.any(
            probability_real < 0
        )

        or

        np.any(
            probability_real > 1
        )
    ):

        probability_real = sigmoid(
            probability_real
        )


# ------------------------------------------------------------
# 2-class softmax output: (N, 2)
# ------------------------------------------------------------

elif (

    raw_predictions.ndim == 2

    and

    raw_predictions.shape[1] == 2
):

    probabilities = (
        raw_predictions.astype(float)
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


    # Assumes:
    #
    # column 0 = Fake
    # column 1 = Real

    probability_real = (
        probabilities[
            :,
            1
        ]
    )


else:

    raise ValueError(
        "\nUnsupported GRU output shape: "
        f"{raw_predictions.shape}"
    )


# ============================================================
# 23. CREATE CLASS PREDICTIONS
# ============================================================

y_pred = (

    probability_real

    >=

    THRESHOLD

).astype(
    np.int32
)


probability_fake = (

    1.0

    -

    probability_real
)


# ============================================================
# 24. METRICS
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


macro_precision = precision_score(

    y_external,

    y_pred,

    average="macro",

    zero_division=0
)


macro_recall = recall_score(

    y_external,

    y_pred,

    average="macro",

    zero_division=0
)


macro_f1 = f1_score(

    y_external,

    y_pred,

    average="macro",

    zero_division=0
)


weighted_f1 = f1_score(

    y_external,

    y_pred,

    average="weighted",

    zero_division=0
)


roc_auc = roc_auc_score(

    y_external,

    probability_real
)


mcc = matthews_corrcoef(

    y_external,

    y_pred
)


# ============================================================
# 25. CLASSIFICATION REPORT
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
# 26. CONFUSION MATRIX
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
# 27. DISPLAY RESULTS
# ============================================================

print(
    "\n=========================================="
)

print(
    "          GRU POLYGLOT EXTERNAL TEST"
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
# 28. SAVE PREDICTIONS
# ============================================================

external_df[
    "predicted_label"
] = y_pred


external_df[
    "probability_fake"
] = probability_fake


external_df[
    "probability_real"
] = probability_real


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

    / "gru_polyglot_external_predictions.csv"
)


external_df.to_csv(

    PREDICTIONS_FILE,

    index=False,

    encoding="utf-8-sig"
)


# ============================================================
# 29. SAVE METRICS
# ============================================================

metrics_df = pd.DataFrame([{

    "model":
        "GRU",

    "model_file":
        MODEL_FILE.name,

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

    / "gru_polyglot_external_metrics.csv"
)


metrics_df.to_csv(

    METRICS_FILE,

    index=False
)


# ============================================================
# 30. SAVE CONFUSION MATRIX
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

    / "gru_polyglot_confusion_matrix.csv"
)


confusion_df.to_csv(
    CONFUSION_FILE
)


# ============================================================
# 31. SAVE CLASSIFICATION REPORT
# ============================================================

REPORT_FILE = (

    REPORT_DIR

    / "gru_polyglot_classification_report.csv"
)


pd.DataFrame(

    report_dict

).transpose().to_csv(

    REPORT_FILE
)


# ============================================================
# 32. FINAL OUTPUT
# ============================================================

print(
    "\n=========================================="
)

print(
    "          GRU EXTERNAL TEST COMPLETE"
)

print(
    "=========================================="
)


print(
    "\nGRU external accuracy:"
)

print(
    f"{accuracy * 100:.2f}%"
)


print(
    "\nGRU external Macro F1:"
)

print(
    f"{macro_f1:.4f}"
)


print(
    "\nGRU external ROC-AUC:"
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
    "PolyglotFakeFacts was used only "
    "for external evaluation."
)

print(
    "Do not retrain or tune the GRU "
    "using this external holdout."
)

print(
    "=========================================="
)