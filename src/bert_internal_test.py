from pathlib import Path
import time

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
# 1. CONFIGURATION
# ============================================================

MODEL_DIR = Path(
    "models/bert_fake_news_final"
)

TEST_FILE = Path(
    "data/test_dataset.csv"
)

REPORT_DIR = Path(
    "reports"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# Must match BERT training
MAX_LENGTH = 256

# Safe starting point for RTX 3050 4 GB
TEST_BATCH_SIZE = 4

SEED = 42


# ============================================================
# 2. FILE CHECKS
# ============================================================

if not MODEL_DIR.exists():

    raise FileNotFoundError(
        f"Trained BERT model folder not found: "
        f"{MODEL_DIR}"
    )


if not TEST_FILE.exists():

    raise FileNotFoundError(
        f"Test dataset not found: "
        f"{TEST_FILE}"
    )


# Make sure important model files exist
config_file = MODEL_DIR / "config.json"

if not config_file.exists():

    raise FileNotFoundError(
        f"config.json not found inside "
        f"{MODEL_DIR}"
    )


# ============================================================
# 3. HARDWARE CHECK
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("\n==========================================")
print("          BERT FINAL TEST")
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

else:

    print(
        "WARNING: CUDA unavailable."
    )

    print(
        "Testing will use CPU."
    )


# ============================================================
# 4. LOAD UNTOUCHED TEST DATASET
# ============================================================

print("\n==========================================")
print("          LOADING TEST DATA")
print("==========================================")


test_df = pd.read_csv(
    TEST_FILE,
    low_memory=False
)


required_columns = {
    "fused_text",
    "label"
}


missing_columns = (
    required_columns
    - set(test_df.columns)
)


if missing_columns:

    raise ValueError(
        f"Test dataset missing required columns: "
        f"{missing_columns}"
    )


# ============================================================
# 5. VALIDATE TEST DATA
# ============================================================

test_df = test_df.copy()


test_df["fused_text"] = (
    test_df["fused_text"]
    .fillna("")
    .astype(str)
    .str.strip()
)


test_df["label"] = pd.to_numeric(
    test_df["label"],
    errors="coerce"
)


# Remove only invalid records.
# DO NOT perform new preprocessing or fit anything.
test_df = test_df[
    (test_df["fused_text"] != "")
    &
    (test_df["label"].isin([0, 1]))
].copy()


test_df["label"] = (
    test_df["label"]
    .astype(int)
)


test_df = test_df.reset_index(
    drop=True
)


if test_df.empty:

    raise ValueError(
        "No valid test records found."
    )


if test_df["label"].nunique() != 2:

    raise ValueError(
        "Test dataset must contain both "
        "Fake (0) and Real (1)."
    )


print(
    "Test records:",
    len(test_df)
)


print(
    "\nTest class distribution:"
)


print(
    test_df["label"]
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
# 6. LOAD SAVED TOKENIZER
# ============================================================

print("\n==========================================")
print("        LOADING SAVED TOKENIZER")
print("==========================================")


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    use_fast=True
)


print(
    "Tokenizer loaded from:",
    MODEL_DIR
)


# ============================================================
# 7. LOAD TRAINED BERT MODEL
# ============================================================

print("\n==========================================")
print("        LOADING TRAINED BERT")
print("==========================================")


model = (
    AutoModelForSequenceClassification
    .from_pretrained(
        MODEL_DIR
    )
)


model.to(
    DEVICE
)


# IMPORTANT:
# Evaluation mode disables training-specific behaviour
# such as dropout.
model.eval()


print(
    "Model loaded from:",
    MODEL_DIR
)


print(
    "Labels:",
    model.config.id2label
)


# ============================================================
# 8. PYTORCH TEST DATASET
# ============================================================

class BertTestDataset(
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

        encoded = self.tokenizer(

            self.texts[index],

            truncation=True,

            max_length=self.max_length,

            padding=False
        )


        encoded["labels"] = (
            self.labels[index]
        )


        return encoded


# ============================================================
# 9. CREATE TEST DATASET / LOADER
# ============================================================

test_dataset = BertTestDataset(

    texts=test_df[
        "fused_text"
    ].tolist(),

    labels=test_df[
        "label"
    ].tolist(),

    tokenizer=tokenizer,

    max_length=MAX_LENGTH
)


data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    return_tensors="pt"
)


test_loader = DataLoader(

    test_dataset,

    batch_size=TEST_BATCH_SIZE,

    shuffle=False,

    collate_fn=data_collator,

    num_workers=0,

    pin_memory=torch.cuda.is_available()
)


# ============================================================
# 10. RUN FINAL INFERENCE
# ============================================================

print("\n==========================================")
print("        RUNNING FINAL TEST")
print("==========================================")


all_labels = []

all_predictions = []

all_real_probabilities = []


start_time = time.time()


total_batches = len(
    test_loader
)


with torch.inference_mode():

    for batch_number, batch in enumerate(
        test_loader,
        start=1
    ):

        labels = batch.pop(
            "labels"
        )


        batch = {
            key: value.to(
                DEVICE
            )
            for key, value
            in batch.items()
        }


        outputs = model(
            **batch
        )


        logits = (
            outputs.logits
        )


        probabilities = torch.softmax(
            logits,
            dim=1
        )


        predictions = torch.argmax(
            probabilities,
            dim=1
        )


        real_probabilities = (
            probabilities[:, 1]
        )


        all_labels.extend(
            labels.cpu().numpy()
        )


        all_predictions.extend(
            predictions.cpu().numpy()
        )


        all_real_probabilities.extend(
            real_probabilities.cpu().numpy()
        )


        # ----------------------------------------
        # Live test progress
        # ----------------------------------------

        if (
            batch_number == 1
            or batch_number % 50 == 0
            or batch_number == total_batches
        ):

            percentage = (
                batch_number
                / total_batches
                * 100
            )


            print(
                f"Testing: "
                f"{percentage:6.2f}% "
                f"({batch_number}/{total_batches} batches)"
            )


test_time = (
    time.time()
    - start_time
)


# Convert lists to arrays
y_true = np.asarray(
    all_labels
)

y_pred = np.asarray(
    all_predictions
)

y_probability_real = np.asarray(
    all_real_probabilities
)


# ============================================================
# 11. FINAL TEST METRICS
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


mcc = matthews_corrcoef(
    y_true,
    y_pred
)


try:

    roc_auc = roc_auc_score(
        y_true,
        y_probability_real
    )

except ValueError:

    roc_auc = float(
        "nan"
    )


# ============================================================
# 12. BOOTSTRAP 95% ACCURACY CI
# ============================================================

print(
    "\nCalculating 95% bootstrap confidence interval..."
)


rng = np.random.default_rng(
    SEED
)


bootstrap_accuracies = []


number_of_bootstraps = 1000

n_samples = len(
    y_true
)


for _ in range(
    number_of_bootstraps
):

    indices = rng.integers(
        0,
        n_samples,
        size=n_samples
    )


    bootstrap_accuracy = accuracy_score(

        y_true[
            indices
        ],

        y_pred[
            indices
        ]
    )


    bootstrap_accuracies.append(
        bootstrap_accuracy
    )


ci_lower = np.percentile(
    bootstrap_accuracies,
    2.5
)


ci_upper = np.percentile(
    bootstrap_accuracies,
    97.5
)


# ============================================================
# 13. CLASSIFICATION REPORT
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


# ============================================================
# 14. CONFUSION MATRIX
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
# 15. PRINT FINAL RESULTS
# ============================================================

print("\n==========================================")
print("       BERT FINAL TEST RESULTS")
print("==========================================")


print(
    f"Accuracy:             "
    f"{accuracy:.4f} "
    f"({accuracy * 100:.2f}%)"
)


print(
    f"95% Accuracy CI:      "
    f"{ci_lower:.4f} - {ci_upper:.4f}"
)


print(
    f"Balanced Accuracy:    "
    f"{balanced_accuracy:.4f}"
)


print(
    f"Macro Precision:      "
    f"{precision_macro:.4f}"
)


print(
    f"Macro Recall:         "
    f"{recall_macro:.4f}"
)


print(
    f"Macro F1:             "
    f"{f1_macro:.4f}"
)


print(
    f"ROC-AUC:              "
    f"{roc_auc:.4f}"
)


print(
    f"MCC:                  "
    f"{mcc:.4f}"
)


print(
    f"Test runtime:         "
    f"{test_time:.2f} seconds"
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


# ============================================================
# 16. SAVE PREDICTIONS
# ============================================================

prediction_df = pd.DataFrame({

    "true_label":
        y_true,

    "predicted_label":
        y_pred,

    "probability_real":
        y_probability_real
})


prediction_df[
    "true_class"
] = prediction_df[
    "true_label"
].map({

    0: "Fake",

    1: "Real"
})


prediction_df[
    "predicted_class"
] = prediction_df[
    "predicted_label"
].map({

    0: "Fake",

    1: "Real"
})


predictions_file = (

    REPORT_DIR
    / "bert_final_test_predictions.csv"
)


prediction_df.to_csv(

    predictions_file,

    index=False
)


# ============================================================
# 17. SAVE FINAL METRICS
# ============================================================

metrics_df = pd.DataFrame([{

    "model":
        "BERT",

    "model_path":
        str(MODEL_DIR),

    "test_records":
        len(y_true),

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

    "roc_auc":
        roc_auc,

    "mcc":
        mcc,

    "test_runtime_seconds":
        test_time,

    "cuda_used":
        torch.cuda.is_available(),

    "device":
        str(DEVICE)
}])


metrics_file = (

    REPORT_DIR
    / "bert_final_test_metrics.csv"
)


metrics_df.to_csv(

    metrics_file,

    index=False
)


# ============================================================
# 18. SAVE CONFUSION MATRIX
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


confusion_file = (

    REPORT_DIR
    / "bert_final_test_confusion_matrix.csv"
)


confusion_df.to_csv(
    confusion_file
)


# ============================================================
# 19. FINAL MESSAGE
# ============================================================

print("\n==========================================")
print("          TEST COMPLETE")
print("==========================================")


print(
    "Metrics saved:"
)

print(
    metrics_file
)


print(
    "\nPredictions saved:"
)

print(
    predictions_file
)


print(
    "\nConfusion matrix saved:"
)

print(
    confusion_file
)


print(
    "\nIMPORTANT:"
)

print(
    "This is INTERNAL test performance."
)

print(
    "Do not retrain or tune BERT based on "
    "this test dataset."
)

print(
    "Next step: run independent external "
    "validation using PolyglotFakeFacts."
)

print(
    "=========================================="
)