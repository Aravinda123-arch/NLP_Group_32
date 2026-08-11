from pathlib import Path
import html
import re
import time
import unicodedata

import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader

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

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding
)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

# This makes the script work even when executed from src/.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

EXTERNAL_DIR = (
    PROJECT_ROOT
    / "data"
    / "external_validation"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "bert_fake_news_final"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. CONFIGURATION
# ============================================================

# Must match BERT training.
MAX_LENGTH = 256

# Safe for RTX 3050 4 GB.
BATCH_SIZE = 4

SEED = 42

# External records must be from this date or newer.
MIN_DATE = pd.Timestamp(
    "2020-01-01"
)

# Minimum article length after cleaning.
MIN_WORDS = 20

# Number of bootstrap resamples for accuracy CI.
N_BOOTSTRAPS = 1000


# ============================================================
# 3. REPRODUCIBILITY
# ============================================================

np.random.seed(
    SEED
)

torch.manual_seed(
    SEED
)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(
        SEED
    )


# ============================================================
# 4. DEVICE CHECK
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("\n==========================================")
print("       BERT EXTERNAL VALIDATION")
print("==========================================")


print(
    "Device:",
    DEVICE
)


if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    print(
        "PyTorch CUDA:",
        torch.version.cuda
    )

    gpu_memory = (
        torch.cuda
        .get_device_properties(0)
        .total_memory
        / (1024 ** 3)
    )

    print(
        "GPU memory:",
        f"{gpu_memory:.2f} GB"
    )

else:

    print(
        "WARNING: CUDA unavailable."
    )

    print(
        "External testing will use CPU."
    )


# ============================================================
# 5. CHECK REQUIRED DIRECTORIES
# ============================================================

if not EXTERNAL_DIR.exists():

    raise FileNotFoundError(
        f"\nExternal validation directory not found:\n"
        f"{EXTERNAL_DIR}\n\n"
        f"Create the folder and place the PolyglotFakeFacts "
        f"Fake and Real files inside it."
    )


if not MODEL_DIR.exists():

    raise FileNotFoundError(
        f"\nFinal BERT model not found:\n"
        f"{MODEL_DIR}"
    )


if not (
    MODEL_DIR / "config.json"
).exists():

    raise FileNotFoundError(
        f"config.json not found inside:\n"
        f"{MODEL_DIR}"
    )


# ============================================================
# 6. FIND POLYGLOT FAKE / REAL FILES
# ============================================================

SUPPORTED_EXTENSIONS = {
    ".csv",
    ".xlsx",
    ".xls"
}


def find_external_file(
    label_name
):

    label_name = (
        label_name
        .lower()
        .strip()
    )

    candidates = []

    for file_path in (
        EXTERNAL_DIR.iterdir()
    ):

        if not file_path.is_file():
            continue

        if (
            file_path.suffix.lower()
            not in SUPPORTED_EXTENSIONS
        ):
            continue

        # Ignore files created by this script.
        if (
            "bert_external"
            in file_path.stem.lower()
        ):
            continue

        if (
            label_name
            in file_path.stem.lower()
        ):

            candidates.append(
                file_path
            )


    if not candidates:

        raise FileNotFoundError(
            f"\nCould not find a {label_name.upper()} "
            f"external dataset inside:\n"
            f"{EXTERNAL_DIR}\n\n"
            f"Supported file types: CSV, XLSX, XLS"
        )


    # Prefer exact names such as Fake.csv / Real.xlsx.
    exact_matches = [

        file_path

        for file_path in candidates

        if (
            file_path.stem.lower()
            == label_name
        )
    ]


    if exact_matches:

        return sorted(
            exact_matches
        )[0]


    if len(candidates) > 1:

        print(
            f"\nMultiple {label_name.upper()} "
            f"files found:"
        )

        for candidate in candidates:

            print(
                " -",
                candidate.name
            )

        print(
            "\nUsing:",
            sorted(candidates)[0].name
        )


    return sorted(
        candidates
    )[0]


FAKE_FILE = find_external_file(
    "fake"
)

REAL_FILE = find_external_file(
    "real"
)


if FAKE_FILE.resolve() == REAL_FILE.resolve():

    raise RuntimeError(
        "Fake and Real external files "
        "resolved to the same file."
    )


print("\n==========================================")
print("        EXTERNAL DATA FILES")
print("==========================================")


print(
    "Fake file:",
    FAKE_FILE.name
)

print(
    "Real file:",
    REAL_FILE.name
)


# ============================================================
# 7. ROBUST CSV / EXCEL LOADER
# ============================================================

def load_dataset_file(
    file_path
):

    suffix = (
        file_path.suffix.lower()
    )


    # --------------------------------------------------------
    # Excel
    # --------------------------------------------------------

    if suffix in {
        ".xlsx",
        ".xls"
    }:

        try:

            dataframe = pd.read_excel(
                file_path
            )

        except ImportError as error:

            raise ImportError(
                "\nExcel support is missing.\n"
                "For .xlsx files run:\n\n"
                "python -m pip install openpyxl\n"
            ) from error

        return dataframe


    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    encodings = [

        "utf-8-sig",

        "utf-8",

        "cp1252",

        "latin1"
    ]


    last_error = None


    for encoding in encodings:

        try:

            dataframe = pd.read_csv(
                file_path,
                encoding=encoding,
                low_memory=False
            )


            # If everything was incorrectly read into one
            # column, attempt automatic delimiter detection.
            if dataframe.shape[1] == 1:

                dataframe = pd.read_csv(
                    file_path,
                    encoding=encoding,
                    sep=None,
                    engine="python"
                )


            return dataframe


        except Exception as error:

            last_error = error


    raise RuntimeError(
        f"Could not read file: "
        f"{file_path}\n"
        f"Last error: {last_error}"
    )


print(
    "\nLoading external datasets..."
)


fake_df = load_dataset_file(
    FAKE_FILE
)

real_df = load_dataset_file(
    REAL_FILE
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
# 8. NORMALIZE COLUMN NAMES FOR DETECTION
# ============================================================

def normalize_column_name(
    column
):

    column = str(
        column
    )

    column = column.strip().lower()

    column = column.replace(
        "_",
        " "
    )

    column = re.sub(
        r"\s+",
        " ",
        column
    )

    return column


def column_lookup(
    dataframe
):

    return {

        normalize_column_name(column):
            column

        for column in dataframe.columns
    }


# ============================================================
# 9. FIND ENGLISH TRANSLATION COLUMN
# ============================================================

TRANSLATED_TEXT_NAMES = [

    "english translated version",

    "english translation",

    "translated english version",

    "translated english text",

    "english translated text"
]


def find_translated_text_column(
    dataframe,
    dataset_name
):

    lookup = column_lookup(
        dataframe
    )


    for expected_name in (
        TRANSLATED_TEXT_NAMES
    ):

        if expected_name in lookup:

            return lookup[
                expected_name
            ]


    print(
        f"\nAvailable columns in "
        f"{dataset_name}:"
    )

    for column in dataframe.columns:

        print(
            " -",
            column
        )


    raise ValueError(
        f"\nCould not find the English translated "
        f"text column in {dataset_name}.\n\n"
        f"Expected a column such as:\n"
        f"'english translated version'\n\n"
        f"Do NOT automatically use the original "
        f"non-English text with bert-base-uncased."
    )


fake_text_column = (
    find_translated_text_column(
        fake_df,
        "Fake dataset"
    )
)

real_text_column = (
    find_translated_text_column(
        real_df,
        "Real dataset"
    )
)


print("\nText columns:")

print(
    "Fake:",
    fake_text_column
)

print(
    "Real:",
    real_text_column
)


# ============================================================
# 10. FIND DATE COLUMNS
# ============================================================

def find_optional_column(
    dataframe,
    names
):

    lookup = column_lookup(
        dataframe
    )


    for name in names:

        if name in lookup:

            return lookup[
                name
            ]


    return None


def build_external_date(
    dataframe
):

    news_date_column = (
        find_optional_column(

            dataframe,

            [
                "news date",
                "publication date",
                "published date"
            ]
        )
    )


    gathering_date_column = (
        find_optional_column(

            dataframe,

            [
                "gathering date",
                "collection date"
            ]
        )
    )


    external_date = pd.Series(
        pd.NaT,
        index=dataframe.index,
        dtype="datetime64[ns]"
    )


    # Prefer actual news/publication date.
    if news_date_column is not None:

        external_date = pd.to_datetime(
            dataframe[
                news_date_column
            ],
            errors="coerce"
        )


    # Use gathering date only when news date
    # is missing.
    if gathering_date_column is not None:

        gathering_date = pd.to_datetime(
            dataframe[
                gathering_date_column
            ],
            errors="coerce"
        )

        external_date = (
            external_date.fillna(
                gathering_date
            )
        )


    return external_date


# ============================================================
# 11. EXTERNAL TEXT CLEANING
# ============================================================

def clean_external_text(
    text
):

    if pd.isna(text):

        return ""


    text = str(
        text
    )


    # Decode HTML entities.
    text = html.unescape(
        text
    )


    # Normalize Unicode characters.
    text = unicodedata.normalize(
        "NFKC",
        text
    )


    # Remove HTML tags.
    text = re.sub(
        r"<[^>]+>",
        " ",
        text
    )


    # Remove URLs.
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text,
        flags=re.IGNORECASE
    )


    # Remove email addresses.
    text = re.sub(
        r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b",
        " ",
        text
    )


    # Remove common Reuters/AP/AFP dateline
    # boilerplate at the beginning.
    text = re.sub(

        r"^\s*"
        r"[A-Z][A-Z\s\.,'\-]{1,60}"
        r"(?:\([^)]*\))?"
        r"\s*[-–—:]\s*",

        " ",

        text,

        flags=re.IGNORECASE
    )


    # Remove explicit wire-service prefix.
    text = re.sub(

        r"^\s*"
        r"(?:REUTERS|AP|AFP)"
        r"\s*[-–—:]\s*",

        " ",

        text,

        flags=re.IGNORECASE
    )


    # Remove common reporting/editing credits.
    text = re.sub(

        r"\b"
        r"(?:reporting by|"
        r"additional reporting by|"
        r"editing by)"
        r"\b.*$",

        " ",

        text,

        flags=(
            re.IGNORECASE
            |
            re.DOTALL
        )
    )


    # Remove control characters.
    text = re.sub(
        r"[\x00-\x1f\x7f]",
        " ",
        text
    )


    # Match training representation.
    text = text.lower()


    # Normalize whitespace.
    text = re.sub(
        r"\s+",
        " ",
        text
    )


    return text.strip()


# ============================================================
# 12. PREPARE EACH EXTERNAL CLASS
# ============================================================

def prepare_external_class(
    dataframe,
    text_column,
    label,
    source_file
):

    output = pd.DataFrame()


    output[
        "external_text"
    ] = dataframe[
        text_column
    ]


    output[
        "external_date"
    ] = build_external_date(
        dataframe
    )


    output[
        "label"
    ] = int(
        label
    )


    output[
        "source_file"
    ] = (
        source_file.name
    )


    # --------------------------------------------------------
    # Preserve useful metadata if present
    # --------------------------------------------------------

    lookup = column_lookup(
        dataframe
    )


    for normalized_name in [

        "url",

        "domain",

        "language",

        "news headline",

        "keywords"
    ]:

        if normalized_name in lookup:

            output[
                normalized_name.replace(
                    " ",
                    "_"
                )
            ] = dataframe[
                lookup[
                    normalized_name
                ]
            ].values


    # --------------------------------------------------------
    # Clean English-translated text
    # --------------------------------------------------------

    output[
        "fused_text"
    ] = output[
        "external_text"
    ].apply(
        clean_external_text
    )


    return output


fake_external = prepare_external_class(

    dataframe=fake_df,

    text_column=fake_text_column,

    label=0,

    source_file=FAKE_FILE
)


real_external = prepare_external_class(

    dataframe=real_df,

    text_column=real_text_column,

    label=1,

    source_file=REAL_FILE
)


external_df = pd.concat(

    [
        fake_external,
        real_external
    ],

    ignore_index=True
)


print("\n==========================================")
print("       EXTERNAL DATA CLEANING")
print("==========================================")


print(
    "Combined raw records:",
    len(external_df)
)


# ============================================================
# 13. FILTER INVALID / SHORT TEXT
# ============================================================

external_df[
    "word_count"
] = external_df[
    "fused_text"
].str.split().str.len()


before_invalid_filter = len(
    external_df
)


external_df = external_df[

    (external_df["fused_text"] != "")

    &

    (
        external_df["word_count"]
        >= MIN_WORDS
    )

].copy()


print(
    "Removed empty/short records:",
    before_invalid_filter
    - len(external_df)
)


# ============================================================
# 14. DATE FILTER
# ============================================================

# We use:
#
# 1. news date when available
# 2. gathering date as fallback
#
# Records with no usable date are excluded because
# this evaluation intentionally requires recent data.

before_date_filter = len(
    external_df
)


external_df = external_df[

    external_df[
        "external_date"
    ].notna()

    &

    (
        external_df[
            "external_date"
        ]
        >= MIN_DATE
    )

].copy()


print(
    "Removed records with missing/pre-2020 date:",
    before_date_filter
    - len(external_df)
)


# ============================================================
# 15. REMOVE CONFLICTING EXACT TEXTS
# ============================================================

label_counts_per_text = (

    external_df

    .groupby(
        "fused_text"
    )["label"]

    .nunique()
)


conflicting_texts = set(

    label_counts_per_text[
        label_counts_per_text > 1
    ].index
)


print(
    "Conflicting Fake/Real exact texts:",
    len(conflicting_texts)
)


if conflicting_texts:

    external_df = external_df[

        ~external_df[
            "fused_text"
        ].isin(
            conflicting_texts
        )

    ].copy()


# ============================================================
# 16. REMOVE EXACT DUPLICATES
# ============================================================

before_duplicates = len(
    external_df
)


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


print(
    "Exact duplicate records removed:",
    before_duplicates
    - len(external_df)
)


# ============================================================
# 17. OPTIONAL INTERNAL-DATA LEAKAGE CHECK
# ============================================================

#
# External evaluation should be independent.
#
# We check exact cleaned-text overlap against
# the original internal train pool and test set.
#
# If an exact overlap is found, remove it from
# the EXTERNAL set only.
#

INTERNAL_FILES = [

    PROJECT_ROOT
    / "data"
    / "before_feature_engineering_dataset.csv",

    PROJECT_ROOT
    / "data"
    / "test_dataset.csv"
]


internal_texts = set()


for internal_file in INTERNAL_FILES:

    if not internal_file.exists():

        continue


    try:

        internal_df = pd.read_csv(
            internal_file,
            usecols=[
                "fused_text"
            ],
            low_memory=False
        )


        cleaned_internal = (

            internal_df[
                "fused_text"
            ]

            .fillna("")

            .astype(str)

            .apply(
                clean_external_text
            )
        )


        internal_texts.update(

            text

            for text in cleaned_internal

            if text
        )


    except Exception as error:

        print(
            f"WARNING: Could not perform overlap "
            f"check using {internal_file.name}: "
            f"{error}"
        )


if internal_texts:

    overlap_mask = (
        external_df[
            "fused_text"
        ].isin(
            internal_texts
        )
    )


    internal_overlap_count = int(
        overlap_mask.sum()
    )


    print(
        "External/internal exact-text overlap:",
        internal_overlap_count
    )


    if internal_overlap_count > 0:

        external_df = external_df[

            ~overlap_mask

        ].copy()

else:

    print(
        "Internal overlap check skipped."
    )


# ============================================================
# 18. FINAL EXTERNAL DATA VALIDATION
# ============================================================

external_df = external_df.sample(

    frac=1.0,

    random_state=SEED

).reset_index(
    drop=True
)


if external_df.empty:

    raise ValueError(
        "No external records remain after filtering."
    )


if external_df[
    "label"
].nunique() != 2:

    raise ValueError(
        "External evaluation requires both "
        "Fake and Real records."
    )


external_df[
    "external_row_id"
] = np.arange(
    len(external_df)
)


print("\n==========================================")
print("       FINAL EXTERNAL DATASET")
print("==========================================")


print(
    "External records:",
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


print(
    "\nDate range:"
)


print(
    "Earliest:",
    external_df[
        "external_date"
    ].min()
)


print(
    "Latest:",
    external_df[
        "external_date"
    ].max()
)


# ============================================================
# 19. LOAD SAVED TOKENIZER
# ============================================================

print("\n==========================================")
print("        LOADING BERT TOKENIZER")
print("==========================================")


tokenizer = (
    AutoTokenizer
    .from_pretrained(
        MODEL_DIR,
        use_fast=True
    )
)


print(
    "Tokenizer loaded from:"
)

print(
    MODEL_DIR
)


# ============================================================
# 20. LOAD FINAL TRAINED BERT MODEL
# ============================================================

print("\n==========================================")
print("       LOADING FINAL BERT MODEL")
print("==========================================")


model = (

    AutoModelForSequenceClassification

    .from_pretrained(
        MODEL_DIR
    )
)


# ============================================================
# 21. VERIFY LABEL MAPPING
# ============================================================

model_id2label = {

    int(key):
        str(value).upper()

    for key, value
    in model.config.id2label.items()
}


print(
    "Model label mapping:",
    model_id2label
)


EXPECTED_LABELS = {
    0: "FAKE",
    1: "REAL"
}


if model_id2label != EXPECTED_LABELS:

    raise ValueError(
        "\nBERT model label mapping does not match "
        "the external evaluation labels.\n\n"
        f"Expected: {EXPECTED_LABELS}\n"
        f"Found:    {model_id2label}\n\n"
        "Do not continue because this could invert "
        "Fake/Real accuracy."
    )


model.to(
    DEVICE
)


# Turn off dropout/training behaviour.
model.eval()


print(
    "Model loaded successfully."
)


# ============================================================
# 22. EXTERNAL PYTORCH DATASET
# ============================================================

class ExternalNewsDataset(
    Dataset
):

    def __init__(
        self,
        texts,
        labels,
        tokenizer,
        max_length
    ):

        self.texts = texts

        self.labels = labels

        self.tokenizer = tokenizer

        self.max_length = max_length


    def __len__(
        self
    ):

        return len(
            self.labels
        )


    def __getitem__(
        self,
        index
    ):

        encoding = self.tokenizer(

            self.texts[
                index
            ],

            truncation=True,

            max_length=(
                self.max_length
            ),

            padding=False
        )


        encoding[
            "labels"
        ] = int(
            self.labels[
                index
            ]
        )


        return encoding


# ============================================================
# 23. CREATE DATALOADER
# ============================================================

external_dataset = (
    ExternalNewsDataset(

        texts=external_df[
            "fused_text"
        ].tolist(),

        labels=external_df[
            "label"
        ].tolist(),

        tokenizer=tokenizer,

        max_length=MAX_LENGTH
    )
)


data_collator = (
    DataCollatorWithPadding(

        tokenizer=tokenizer,

        return_tensors="pt"
    )
)


external_loader = DataLoader(

    external_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    collate_fn=data_collator,

    num_workers=0,

    pin_memory=(
        DEVICE.type == "cuda"
    )
)


# ============================================================
# 24. TIME FORMATTER
# ============================================================

def format_time(
    seconds
):

    seconds = max(
        0,
        int(seconds)
    )


    hours = (
        seconds
        // 3600
    )

    minutes = (
        (seconds % 3600)
        // 60
    )

    secs = (
        seconds
        % 60
    )


    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{secs:02d}"
    )


# ============================================================
# 25. RUN EXTERNAL INFERENCE
# ============================================================

print("\n==========================================")
print("       EXTERNAL TEST STARTED")
print("==========================================")


all_true_labels = []

all_predictions = []

all_fake_probabilities = []

all_real_probabilities = []


total_batches = len(
    external_loader
)


test_start_time = (
    time.time()
)


with torch.inference_mode():

    for batch_number, batch in enumerate(

        external_loader,

        start=1
    ):

        # ----------------------------------------------------
        # Remove labels before passing data to BERT.
        # ----------------------------------------------------

        labels = batch.pop(
            "labels"
        )


        batch = {

            key:
                value.to(
                    DEVICE,
                    non_blocking=True
                )

            for key, value
            in batch.items()
        }


        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        outputs = model(
            **batch
        )


        logits = (
            outputs.logits
        )


        probabilities = (
            torch.softmax(
                logits,
                dim=1
            )
        )


        predictions = (
            torch.argmax(
                probabilities,
                dim=1
            )
        )


        # ----------------------------------------------------
        # Collect results
        # ----------------------------------------------------

        all_true_labels.extend(

            labels
            .cpu()
            .numpy()
            .tolist()
        )


        all_predictions.extend(

            predictions
            .cpu()
            .numpy()
            .tolist()
        )


        all_fake_probabilities.extend(

            probabilities[
                :,
                0
            ]

            .cpu()
            .numpy()
            .tolist()
        )


        all_real_probabilities.extend(

            probabilities[
                :,
                1
            ]

            .cpu()
            .numpy()
            .tolist()
        )


        # ----------------------------------------------------
        # Live progress / ETA
        # ----------------------------------------------------

        if (

            batch_number == 1

            or

            batch_number % 25 == 0

            or

            batch_number
            == total_batches
        ):

            elapsed = (
                time.time()
                - test_start_time
            )


            progress = (
                batch_number
                / total_batches
            )


            estimated_total = (
                elapsed
                / progress
            )


            remaining = (
                estimated_total
                - elapsed
            )


            percentage = (
                progress
                * 100
            )


            print(

                f"External test: "
                f"{percentage:6.2f}% | "

                f"Batch "
                f"{batch_number}/"
                f"{total_batches} | "

                f"Elapsed "
                f"{format_time(elapsed)} | "

                f"Remaining "
                f"{format_time(remaining)}"
            )


external_test_time = (
    time.time()
    - test_start_time
)


# ============================================================
# 26. CONVERT RESULTS TO NUMPY
# ============================================================

y_true = np.asarray(

    all_true_labels,

    dtype=np.int64
)


y_pred = np.asarray(

    all_predictions,

    dtype=np.int64
)


probability_fake = np.asarray(

    all_fake_probabilities,

    dtype=np.float64
)


probability_real = np.asarray(

    all_real_probabilities,

    dtype=np.float64
)


if len(y_true) != len(
    external_df
):

    raise RuntimeError(
        "Prediction count does not match "
        "external dataset size."
    )


# ============================================================
# 27. EXTERNAL METRICS
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)


balanced_accuracy = (
    balanced_accuracy_score(
        y_true,
        y_pred
    )
)


precision_macro = precision_score(

    y_true,

    y_pred,

    average="macro",

    zero_division=0
)


recall_macro = recall_score(

    y_true,

    y_pred,

    average="macro",

    zero_division=0
)


f1_macro = f1_score(

    y_true,

    y_pred,

    average="macro",

    zero_division=0
)


f1_weighted = f1_score(

    y_true,

    y_pred,

    average="weighted",

    zero_division=0
)


mcc = matthews_corrcoef(
    y_true,
    y_pred
)


try:

    roc_auc = roc_auc_score(
        y_true,
        probability_real
    )

except ValueError:

    roc_auc = float(
        "nan"
    )


# ============================================================
# 28. BOOTSTRAP 95% ACCURACY CONFIDENCE INTERVAL
# ============================================================

print(
    "\nCalculating 95% bootstrap "
    "confidence interval..."
)


rng = np.random.default_rng(
    SEED
)


number_of_samples = len(
    y_true
)


bootstrap_accuracies = []


for _ in range(
    N_BOOTSTRAPS
):

    indices = rng.integers(

        low=0,

        high=number_of_samples,

        size=number_of_samples
    )


    bootstrap_accuracy = (
        accuracy_score(

            y_true[
                indices
            ],

            y_pred[
                indices
            ]
        )
    )


    bootstrap_accuracies.append(
        bootstrap_accuracy
    )


ci_lower = float(
    np.percentile(
        bootstrap_accuracies,
        2.5
    )
)


ci_upper = float(
    np.percentile(
        bootstrap_accuracies,
        97.5
    )
)


# ============================================================
# 29. CLASSIFICATION REPORT
# ============================================================

classification_report_text = (
    classification_report(

        y_true,

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

        y_true,

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
# 30. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(

    y_true,

    y_pred,

    labels=[
        0,
        1
    ]
)


# ============================================================
# 31. PRINT EXTERNAL RESULTS
# ============================================================

print("\n==========================================")
print("     BERT EXTERNAL TEST RESULTS")
print("==========================================")


print(
    f"External records:        "
    f"{len(y_true)}"
)


print(
    f"Accuracy:                "
    f"{accuracy:.4f} "
    f"({accuracy * 100:.2f}%)"
)


print(
    f"95% Accuracy CI:         "
    f"{ci_lower:.4f} - "
    f"{ci_upper:.4f}"
)


print(
    f"Balanced Accuracy:       "
    f"{balanced_accuracy:.4f}"
)


print(
    f"Macro Precision:         "
    f"{precision_macro:.4f}"
)


print(
    f"Macro Recall:            "
    f"{recall_macro:.4f}"
)


print(
    f"Macro F1:                "
    f"{f1_macro:.4f}"
)


print(
    f"Weighted F1:             "
    f"{f1_weighted:.4f}"
)


print(
    f"ROC-AUC:                 "
    f"{roc_auc:.4f}"
)


print(
    f"MCC:                     "
    f"{mcc:.4f}"
)


print(
    f"External test runtime:   "
    f"{format_time(external_test_time)}"
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
    "\nMatrix interpretation:"
)

print(
    "Rows    = Actual [Fake, Real]"
)

print(
    "Columns = Predicted [Fake, Real]"
)


# ============================================================
# 32. SAVE PREDICTIONS
# ============================================================

predictions_df = (
    external_df.copy()
)


predictions_df[
    "true_label"
] = y_true


predictions_df[
    "predicted_label"
] = y_pred


predictions_df[
    "true_class"
] = pd.Series(
    y_true
).map({

    0: "Fake",

    1: "Real"
}).values


predictions_df[
    "predicted_class"
] = pd.Series(
    y_pred
).map({

    0: "Fake",

    1: "Real"
}).values


predictions_df[
    "probability_fake"
] = probability_fake


predictions_df[
    "probability_real"
] = probability_real


predictions_df[
    "correct"
] = (
    y_true
    == y_pred
)


PREDICTIONS_FILE = (

    REPORT_DIR
    / "bert_external_test_predictions.csv"
)


predictions_df.to_csv(

    PREDICTIONS_FILE,

    index=False,

    encoding="utf-8-sig"
)


# ============================================================
# 33. SAVE EXTERNAL METRICS
# ============================================================

metrics_df = pd.DataFrame([{

    "model":
        "BERT",

    "model_path":
        str(MODEL_DIR),

    "external_dataset":
        "PolyglotFakeFacts",

    "fake_source_file":
        FAKE_FILE.name,

    "real_source_file":
        REAL_FILE.name,

    "text_field":
        "english translated version",

    "minimum_date":
        str(MIN_DATE.date()),

    "test_records":
        len(y_true),

    "fake_records":
        int(
            np.sum(
                y_true == 0
            )
        ),

    "real_records":
        int(
            np.sum(
                y_true == 1
            )
        ),

    "accuracy":
        accuracy,

    "accuracy_percent":
        accuracy * 100,

    "accuracy_ci_95_lower":
        ci_lower,

    "accuracy_ci_95_upper":
        ci_upper,

    "balanced_accuracy":
        balanced_accuracy,

    "precision_macro":
        precision_macro,

    "recall_macro":
        recall_macro,

    "f1_macro":
        f1_macro,

    "f1_weighted":
        f1_weighted,

    "roc_auc":
        roc_auc,

    "mcc":
        mcc,

    "max_length":
        MAX_LENGTH,

    "batch_size":
        BATCH_SIZE,

    "cuda_used":
        DEVICE.type == "cuda",

    "device":
        str(DEVICE),

    "test_runtime_seconds":
        external_test_time
}])


METRICS_FILE = (

    REPORT_DIR
    / "bert_external_test_metrics.csv"
)


metrics_df.to_csv(

    METRICS_FILE,

    index=False
)


# ============================================================
# 34. SAVE CONFUSION MATRIX
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
    / "bert_external_test_confusion_matrix.csv"
)


confusion_df.to_csv(
    CONFUSION_FILE
)


# ============================================================
# 35. SAVE CLASSIFICATION REPORT
# ============================================================

CLASSIFICATION_REPORT_FILE = (

    REPORT_DIR
    / "bert_external_classification_report.csv"
)


pd.DataFrame(
    classification_report_dict
).transpose().to_csv(

    CLASSIFICATION_REPORT_FILE
)


# ============================================================
# 36. SAVE CLEANED EXTERNAL EVALUATION SET
# ============================================================

CLEAN_EXTERNAL_FILE = (

    REPORT_DIR
    / "bert_external_evaluation_dataset.csv"
)


external_df.to_csv(

    CLEAN_EXTERNAL_FILE,

    index=False,

    encoding="utf-8-sig"
)


# ============================================================
# 37. FINAL SUMMARY
# ============================================================

print("\n==========================================")
print("     EXTERNAL VALIDATION COMPLETE")
print("==========================================")


print(
    "\nExternal Accuracy:"
)

print(
    f"{accuracy * 100:.2f}%"
)


print(
    "\nExternal Macro F1:"
)

print(
    f"{f1_macro:.4f}"
)


print(
    "\nExternal ROC-AUC:"
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
    CLEAN_EXTERNAL_FILE
)


print(
    "\nIMPORTANT:"
)

print(
    "This external dataset was used for "
    "evaluation only."
)

print(
    "Do NOT fine-tune BERT, select a checkpoint, "
    "change the classification threshold, or tune "
    "hyperparameters based on these results if "
    "PolyglotFakeFacts is your final external holdout."
)


print(
    "\nNext comparison:"
)

print(
    "Random Forest external performance"
)

print(
    "VS"
)

print(
    "BERT external performance"
)


print(
    "=========================================="
)