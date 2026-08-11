from pathlib import Path
from importlib.metadata import version, PackageNotFoundError
import math
import sys
import time

import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    EarlyStoppingCallback,
    set_seed
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

MODEL_NAME = "bert-base-uncased"

MAX_LENGTH = 256

NUM_EPOCHS = 4

LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01

TRAIN_BATCH_SIZE = 2
EVAL_BATCH_SIZE = 4

GRADIENT_ACCUMULATION_STEPS = 8

WARMUP_PERCENT = 0.10

SEED = 42

# Update live terminal progress every N optimizer steps.
LIVE_UPDATE_STEPS = 10

# Set True if you do NOT want accidental CPU training.
# Recommended for your RTX 3050 training.
REQUIRE_CUDA = True


TRAIN_FILE = Path(
    "data/bert/bert_train_dataset.csv"
)

VALID_FILE = Path(
    "data/bert/bert_validation_dataset.csv"
)

CHECKPOINT_DIR = Path(
    "models/bert_checkpoints"
)

FINAL_MODEL_DIR = Path(
    "models/bert_fake_news_final"
)

REPORT_DIR = Path(
    "reports"
)


CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FINAL_MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. CHECK REQUIRED PACKAGES
# ============================================================

print("\n==========================================")
print("          DEPENDENCY CHECK")
print("==========================================")


try:
    accelerate_version = version("accelerate")
except PackageNotFoundError:
    raise RuntimeError(
        "\nHugging Face Accelerate is not installed.\n\n"
        "Activate your .venv and run:\n\n"
        'python -m pip install -U "accelerate>=1.1.0"\n'
    )


print(
    "Accelerate version:",
    accelerate_version
)

print(
    "PyTorch version:",
    torch.__version__
)


try:
    transformers_version = version(
        "transformers"
    )

    print(
        "Transformers version:",
        transformers_version
    )

except PackageNotFoundError:
    raise RuntimeError(
        "Transformers is not installed."
    )


# ============================================================
# 3. REPRODUCIBILITY
# ============================================================

set_seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(
        SEED
    )


# ============================================================
# 4. HARDWARE / CUDA CHECK
# ============================================================

USE_CUDA = torch.cuda.is_available()


print("\n==========================================")
print("              HARDWARE")
print("==========================================")

print(
    "CUDA available:",
    USE_CUDA
)


if USE_CUDA:

    GPU_NAME = torch.cuda.get_device_name(
        0
    )

    GPU_MEMORY_GB = (
        torch.cuda.get_device_properties(
            0
        ).total_memory
        / (1024 ** 3)
    )

    print(
        "Training device: CUDA GPU"
    )

    print(
        "GPU:",
        GPU_NAME
    )

    print(
        "GPU memory:",
        f"{GPU_MEMORY_GB:.2f} GB"
    )

    print(
        "PyTorch CUDA:",
        torch.version.cuda
    )

    # Clear unused cached memory before model load.
    torch.cuda.empty_cache()

else:

    print(
        "Training device: CPU"
    )

    print(
        "PyTorch CUDA:",
        torch.version.cuda
    )


    if REQUIRE_CUDA:

        raise RuntimeError(
            "\nCUDA is NOT available.\n"
            "Training has been stopped to prevent "
            "accidental CPU-only BERT training.\n\n"
            "Check with:\n"
            "python -c \"import torch; "
            "print(torch.cuda.is_available())\"\n"
        )

    else:

        print(
            "WARNING: BERT will train on CPU."
        )


# ============================================================
# 5. CHECK INPUT FILES
# ============================================================

for file_path in [
    TRAIN_FILE,
    VALID_FILE
]:

    if not file_path.exists():

        raise FileNotFoundError(
            f"Required dataset not found: "
            f"{file_path}"
        )


# ============================================================
# 6. LOAD TRAIN / VALIDATION DATA
# ============================================================

print("\n==========================================")
print("             LOADING DATA")
print("==========================================")


train_df = pd.read_csv(
    TRAIN_FILE,
    low_memory=False
)

valid_df = pd.read_csv(
    VALID_FILE,
    low_memory=False
)


REQUIRED_COLUMNS = {
    "fused_text",
    "label"
}


for dataset_name, dataframe in [

    ("Training", train_df),

    ("Validation", valid_df)

]:

    missing_columns = (
        REQUIRED_COLUMNS
        - set(dataframe.columns)
    )

    if missing_columns:

        raise ValueError(
            f"{dataset_name} dataset is missing "
            f"required columns: "
            f"{missing_columns}"
        )


# ============================================================
# 7. CLEAN / VALIDATE DATA
# ============================================================

def validate_dataframe(
    dataframe
):

    dataframe = dataframe.copy()

    dataframe["fused_text"] = (
        dataframe["fused_text"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    dataframe["label"] = pd.to_numeric(
        dataframe["label"],
        errors="coerce"
    )

    dataframe = dataframe[
        (dataframe["fused_text"] != "")
        &
        (dataframe["label"].isin([0, 1]))
    ].copy()

    dataframe["label"] = (
        dataframe["label"]
        .astype(int)
    )

    dataframe = dataframe.reset_index(
        drop=True
    )

    return dataframe


train_df = validate_dataframe(
    train_df
)

valid_df = validate_dataframe(
    valid_df
)


if train_df.empty:

    raise ValueError(
        "Training dataset is empty."
    )


if valid_df.empty:

    raise ValueError(
        "Validation dataset is empty."
    )


if train_df["label"].nunique() != 2:

    raise ValueError(
        "Training dataset must contain "
        "Fake (0) and Real (1)."
    )


if valid_df["label"].nunique() != 2:

    raise ValueError(
        "Validation dataset must contain "
        "Fake (0) and Real (1)."
    )


print(
    "Training records:",
    len(train_df)
)

print(
    "Validation records:",
    len(valid_df)
)


print(
    "\nTraining class distribution:"
)

print(
    train_df["label"]
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
    "\nValidation class distribution:"
)

print(
    valid_df["label"]
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
# 8. CHECK TRAIN / VALIDATION TEXT OVERLAP
# ============================================================

train_texts = set(
    train_df["fused_text"]
)

valid_texts = set(
    valid_df["fused_text"]
)

text_overlap = (
    train_texts
    & valid_texts
)


print(
    "\nExact text overlap:",
    len(text_overlap)
)


if text_overlap:

    raise ValueError(
        f"Data leakage detected: "
        f"{len(text_overlap)} exact texts occur in "
        f"both training and validation sets."
    )


# ============================================================
# 9. LOAD TOKENIZER
# ============================================================

print("\n==========================================")
print("          LOADING TOKENIZER")
print("==========================================")


print(
    "Base model:",
    MODEL_NAME
)


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True
)


# ============================================================
# 10. PYTORCH DATASET CLASS
# ============================================================

class NewsDataset(
    Dataset
):

    def __init__(
        self,
        texts,
        labels,
        tokenizer,
        max_length
    ):

        self.encodings = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding=False
        )

        self.labels = labels


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

        item = {

            key: torch.tensor(
                value[index],
                dtype=torch.long
            )

            for key, value
            in self.encodings.items()
        }

        item["labels"] = torch.tensor(
            self.labels[index],
            dtype=torch.long
        )

        return item


# ============================================================
# 11. TOKENIZE DATASETS
# ============================================================

print("\n==========================================")
print("             TOKENIZATION")
print("==========================================")


tokenization_start = (
    time.time()
)


print(
    "Tokenizing training dataset..."
)


train_dataset = NewsDataset(

    texts=train_df[
        "fused_text"
    ].tolist(),

    labels=train_df[
        "label"
    ].tolist(),

    tokenizer=tokenizer,

    max_length=MAX_LENGTH
)


print(
    "Tokenizing validation dataset..."
)


validation_dataset = NewsDataset(

    texts=valid_df[
        "fused_text"
    ].tolist(),

    labels=valid_df[
        "label"
    ].tolist(),

    tokenizer=tokenizer,

    max_length=MAX_LENGTH
)


tokenization_time = (
    time.time()
    - tokenization_start
)


print(
    "Tokenization completed in:",
    f"{tokenization_time:.2f} seconds"
)


# Dynamic padding only up to longest sequence
# within each batch.
data_collator = (
    DataCollatorWithPadding(
        tokenizer=tokenizer
    )
)


# ============================================================
# 12. LOAD BERT CLASSIFICATION MODEL
# ============================================================

print("\n==========================================")
print("            LOADING BERT")
print("==========================================")


ID2LABEL = {
    0: "FAKE",
    1: "REAL"
}

LABEL2ID = {
    "FAKE": 0,
    "REAL": 1
}


model = (
    AutoModelForSequenceClassification
    .from_pretrained(

        MODEL_NAME,

        num_labels=2,

        id2label=ID2LABEL,

        label2id=LABEL2ID
    )
)


# This is expected:
#
# pretrained BERT does not already contain
# your custom Fake/Real classification head.
#
# classifier.weight and classifier.bias are
# newly initialized and will be learned below.

print(
    "\nBERT sequence-classification model loaded."
)

print(
    "Classification labels:"
)

print(
    ID2LABEL
)


# ============================================================
# 13. VALIDATION METRICS
# ============================================================

def compute_metrics(
    eval_prediction
):

    logits = (
        eval_prediction.predictions
    )

    labels = (
        eval_prediction.label_ids
    )


    if isinstance(
        logits,
        tuple
    ):

        logits = logits[0]


    predictions = np.argmax(
        logits,
        axis=1
    )


    # Numerically stable softmax
    shifted_logits = (

        logits
        -
        np.max(
            logits,
            axis=1,
            keepdims=True
        )
    )


    exp_logits = np.exp(
        shifted_logits
    )


    probabilities = (

        exp_logits
        /
        np.sum(
            exp_logits,
            axis=1,
            keepdims=True
        )
    )


    probability_real = (
        probabilities[:, 1]
    )


    accuracy = accuracy_score(
        labels,
        predictions
    )


    balanced_accuracy = (
        balanced_accuracy_score(
            labels,
            predictions
        )
    )


    precision_macro = precision_score(
        labels,
        predictions,
        average="macro",
        zero_division=0
    )


    recall_macro = recall_score(
        labels,
        predictions,
        average="macro",
        zero_division=0
    )


    f1_macro = f1_score(
        labels,
        predictions,
        average="macro",
        zero_division=0
    )


    try:

        roc_auc = roc_auc_score(
            labels,
            probability_real
        )

    except ValueError:

        roc_auc = float(
            "nan"
        )


    return {

        "accuracy":
            accuracy,

        "balanced_accuracy":
            balanced_accuracy,

        "precision_macro":
            precision_macro,

        "recall_macro":
            recall_macro,

        "f1_macro":
            f1_macro,

        "roc_auc":
            roc_auc
    }


# ============================================================
# 14. CALCULATE WARMUP STEPS
# ============================================================

#
# Replaces deprecated:
#
# warmup_ratio = 0.10
#
# We calculate 10% of planned optimizer updates.
#

batches_per_epoch = math.ceil(
    len(train_dataset)
    / TRAIN_BATCH_SIZE
)


optimizer_steps_per_epoch = math.ceil(
    batches_per_epoch
    / GRADIENT_ACCUMULATION_STEPS
)


planned_training_steps = (
    optimizer_steps_per_epoch
    * NUM_EPOCHS
)


WARMUP_STEPS = int(
    planned_training_steps
    * WARMUP_PERCENT
)


print("\n==========================================")
print("          TRAINING PLAN")
print("==========================================")


print(
    "Batches per epoch:",
    batches_per_epoch
)

print(
    "Optimizer steps per epoch:",
    optimizer_steps_per_epoch
)

print(
    "Planned optimizer steps:",
    planned_training_steps
)

print(
    "Warmup steps:",
    WARMUP_STEPS
)


# ============================================================
# 15. TIME FORMAT
# ============================================================

def format_time(
    seconds
):

    seconds = max(
        0,
        int(seconds)
    )

    hours = (
        seconds // 3600
    )

    minutes = (
        (seconds % 3600)
        // 60
    )

    secs = (
        seconds % 60
    )


    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{secs:02d}"
    )


# ============================================================
# 16. LIVE TRAINING + ETA CALLBACK
# ============================================================

class LiveTrainingCallback(
    TrainerCallback
):

    def __init__(
        self,
        update_every=10
    ):

        self.start_time = None

        self.update_every = (
            update_every
        )

        self.last_step = -1


    def on_train_begin(
        self,
        args,
        state,
        control,
        **kwargs
    ):

        self.start_time = (
            time.time()
        )

        self.last_step = -1


        print(
            "\n=========================================="
        )

        print(
            "          LIVE BERT TRAINING"
        )

        print(
            "=========================================="
        )

        print(
            "Planned epochs:",
            NUM_EPOCHS
        )

        print(
            "Total optimizer steps:",
            state.max_steps
        )

        print(
            "Training started..."
        )

        print()


    def on_step_end(
        self,
        args,
        state,
        control,
        **kwargs
    ):

        current_step = (
            state.global_step
        )

        total_steps = (
            state.max_steps
        )


        if (
            self.start_time is None
            or current_step <= 0
            or total_steps <= 0
        ):

            return


        should_update = (

            current_step == 1

            or

            current_step
            % self.update_every
            == 0

            or

            current_step >= total_steps
        )


        if not should_update:

            return


        if current_step == self.last_step:

            return


        self.last_step = (
            current_step
        )


        elapsed = (
            time.time()
            - self.start_time
        )


        progress = (
            current_step
            / total_steps
        )


        progress = min(
            max(
                progress,
                0.0
            ),
            1.0
        )


        average_seconds_per_step = (
            elapsed
            / current_step
        )


        remaining_steps = (
            total_steps
            - current_step
        )


        remaining_time = (
            average_seconds_per_step
            * remaining_steps
        )


        estimated_total_time = (
            elapsed
            + remaining_time
        )


        percentage = (
            progress
            * 100
        )


        # ----------------------------------------
        # Terminal progress bar
        # ----------------------------------------

        bar_length = 30

        completed = int(
            progress
            * bar_length
        )


        completed = min(
            completed,
            bar_length
        )


        bar = (

            "#"
            * completed

            +

            "-"
            * (
                bar_length
                - completed
            )
        )


        epoch = (
            state.epoch
            if state.epoch is not None
            else 0.0
        )


        live_line = (

            f"\r[{bar}] "

            f"{percentage:6.2f}% | "

            f"Epoch "
            f"{epoch:.2f}/{NUM_EPOCHS} | "

            f"Step "
            f"{current_step}/{total_steps} | "

            f"Elapsed "
            f"{format_time(elapsed)} | "

            f"Remaining "
            f"{format_time(remaining_time)} | "

            f"Est.Total "
            f"{format_time(estimated_total_time)}"
        )


        sys.stdout.write(
            live_line
        )

        sys.stdout.flush()


    def on_log(
        self,
        args,
        state,
        control,
        logs=None,
        **kwargs
    ):

        if not logs:

            return


        if "loss" in logs:

            print()

            print(
                f"Training loss: "
                f"{logs['loss']:.4f}"
            )


        if "learning_rate" in logs:

            print(
                "Learning rate:",
                f"{logs['learning_rate']:.8f}"
            )


        if USE_CUDA:

            allocated = (
                torch.cuda.memory_allocated(0)
                / (1024 ** 3)
            )

            reserved = (
                torch.cuda.memory_reserved(0)
                / (1024 ** 3)
            )

            print(
                f"GPU memory allocated: "
                f"{allocated:.2f} GB"
            )

            print(
                f"GPU memory reserved: "
                f"{reserved:.2f} GB"
            )


    def on_evaluate(
        self,
        args,
        state,
        control,
        metrics=None,
        **kwargs
    ):

        print()

        print(
            "\n------------------------------------------"
        )

        print(
            "          VALIDATION RESULTS"
        )

        print(
            "------------------------------------------"
        )


        if state.epoch is not None:

            print(
                f"Epoch: "
                f"{state.epoch:.2f}"
            )


        if metrics:

            metric_names = [

                "eval_loss",

                "eval_accuracy",

                "eval_balanced_accuracy",

                "eval_precision_macro",

                "eval_recall_macro",

                "eval_f1_macro",

                "eval_roc_auc"

            ]


            for metric_name in metric_names:

                if metric_name in metrics:

                    value = metrics[
                        metric_name
                    ]

                    if isinstance(
                        value,
                        (float, int)
                    ):

                        print(
                            f"{metric_name}: "
                            f"{value:.4f}"
                        )


        print(
            "------------------------------------------"
        )


    def on_train_end(
        self,
        args,
        state,
        control,
        **kwargs
    ):

        print()


        if self.start_time is None:

            return


        actual_time = (
            time.time()
            - self.start_time
        )


        print(
            "\n=========================================="
        )

        print(
            "        BERT TRAINING FINISHED"
        )

        print(
            "=========================================="
        )

        print(
            "Completed steps:",
            state.global_step
        )

        print(
            "Final epoch:",
            (
                f"{state.epoch:.2f}"
                if state.epoch is not None
                else "N/A"
            )
        )

        print(
            "Actual training time:",
            format_time(
                actual_time
            )
        )


# ============================================================
# 17. TRAINING ARGUMENTS
# ============================================================

training_args = TrainingArguments(

    output_dir=str(
        CHECKPOINT_DIR
    ),


    # --------------------------------------------------------
    # Main training parameters
    # --------------------------------------------------------

    num_train_epochs=NUM_EPOCHS,

    learning_rate=LEARNING_RATE,

    weight_decay=WEIGHT_DECAY,


    # Replaces deprecated warmup_ratio
    warmup_steps=WARMUP_STEPS,


    # --------------------------------------------------------
    # RTX 3050 4 GB settings
    # --------------------------------------------------------

    per_device_train_batch_size=(
        TRAIN_BATCH_SIZE
    ),

    per_device_eval_batch_size=(
        EVAL_BATCH_SIZE
    ),

    gradient_accumulation_steps=(
        GRADIENT_ACCUMULATION_STEPS
    ),

    gradient_checkpointing=True,


    # --------------------------------------------------------
    # Evaluation / checkpoints
    # --------------------------------------------------------

    eval_strategy="epoch",

    save_strategy="epoch",

    load_best_model_at_end=True,

    metric_for_best_model=(
        "f1_macro"
    ),

    greater_is_better=True,

    save_total_limit=2,


    # --------------------------------------------------------
    # Mixed precision
    # --------------------------------------------------------

    fp16=USE_CUDA,

    bf16=False,


    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------

    logging_strategy="steps",

    logging_steps=50,

    logging_first_step=True,

    # We use custom LiveTrainingCallback.
    disable_tqdm=True,

    report_to="none",


    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optim="adamw_torch",


    # --------------------------------------------------------
    # Windows-safe settings
    # --------------------------------------------------------

    dataloader_num_workers=0,

    dataloader_pin_memory=USE_CUDA,


    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    seed=SEED,

    data_seed=SEED
)


# ============================================================
# 18. CREATE TRAINER
# ============================================================

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=validation_dataset,

    data_collator=data_collator,

    processing_class=tokenizer,

    compute_metrics=compute_metrics,

    callbacks=[

        EarlyStoppingCallback(

            early_stopping_patience=1,

            early_stopping_threshold=0.001
        ),

        LiveTrainingCallback(

            update_every=LIVE_UPDATE_STEPS
        )
    ]
)


# ============================================================
# 19. START BERT FINE-TUNING
# ============================================================

print("\n==========================================")
print("          BERT TRAINING START")
print("==========================================")


print(
    "Model:",
    MODEL_NAME
)

print(
    "Training records:",
    len(train_dataset)
)

print(
    "Validation records:",
    len(validation_dataset)
)

print(
    "Max sequence length:",
    MAX_LENGTH
)

print(
    "Epochs:",
    NUM_EPOCHS
)

print(
    "Train batch size:",
    TRAIN_BATCH_SIZE
)

print(
    "Gradient accumulation:",
    GRADIENT_ACCUMULATION_STEPS
)

print(
    "Effective batch size:",
    TRAIN_BATCH_SIZE
    * GRADIENT_ACCUMULATION_STEPS
)

print(
    "Learning rate:",
    LEARNING_RATE
)

print(
    "Warmup steps:",
    WARMUP_STEPS
)


if USE_CUDA:

    print(
        "Training with GPU:",
        torch.cuda.get_device_name(0)
    )

else:

    print(
        "Training with CPU"
    )


training_start_time = (
    time.time()
)


train_result = trainer.train()


training_time = (
    time.time()
    - training_start_time
)


# ============================================================
# 20. TRAINING RUNTIME
# ============================================================

print(
    "\nTraining process completed."
)

print(
    "Total runtime:",
    format_time(
        training_time
    )
)

print(
    "Total training minutes:",
    round(
        training_time / 60,
        2
    )
)


# ============================================================
# 21. EVALUATE BEST CHECKPOINT
# ============================================================

print("\n==========================================")
print("       FINAL VALIDATION RESULTS")
print("==========================================")


validation_metrics = trainer.evaluate(
    eval_dataset=validation_dataset
)


for metric_name, value in (
    validation_metrics.items()
):

    if isinstance(
        value,
        (float, int)
    ):

        print(
            f"{metric_name}: "
            f"{value:.4f}"
        )


# ============================================================
# 22. SAVE FINAL BERT MODEL
# ============================================================

print("\n==========================================")
print("          SAVING FINAL MODEL")
print("==========================================")


trainer.save_model(
    str(
        FINAL_MODEL_DIR
    )
)


tokenizer.save_pretrained(
    str(
        FINAL_MODEL_DIR
    )
)


print(
    "Final model saved to:"
)

print(
    FINAL_MODEL_DIR
)


# ============================================================
# 23. SAVE VALIDATION REPORT
# ============================================================

validation_report = {

    "model":
        MODEL_NAME,

    "training_records":
        len(train_df),

    "validation_records":
        len(valid_df),

    "max_length":
        MAX_LENGTH,

    "learning_rate":
        LEARNING_RATE,

    "weight_decay":
        WEIGHT_DECAY,

    "warmup_steps":
        WARMUP_STEPS,

    "epochs_requested":
        NUM_EPOCHS,

    "epochs_completed":
        (
            trainer.state.epoch
            if trainer.state.epoch
            is not None
            else np.nan
        ),

    "train_batch_size":
        TRAIN_BATCH_SIZE,

    "eval_batch_size":
        EVAL_BATCH_SIZE,

    "gradient_accumulation_steps":
        GRADIENT_ACCUMULATION_STEPS,

    "effective_batch_size":
        (
            TRAIN_BATCH_SIZE
            * GRADIENT_ACCUMULATION_STEPS
        ),

    "cuda_used":
        USE_CUDA,

    "gpu_name":
        (
            torch.cuda.get_device_name(0)
            if USE_CUDA
            else "CPU"
        ),

    "training_time_seconds":
        training_time,

    "training_time_minutes":
        training_time / 60,

    "best_checkpoint":
        trainer.state.best_model_checkpoint,

    "best_validation_metric":
        trainer.state.best_metric
}


for key, value in (
    validation_metrics.items()
):

    validation_report[
        key
    ] = value


validation_report_file = (

    REPORT_DIR
    / "bert_validation_metrics.csv"
)


pd.DataFrame(
    [validation_report]
).to_csv(

    validation_report_file,

    index=False
)


# ============================================================
# 24. SAVE TRAINING HISTORY
# ============================================================

training_history = pd.DataFrame(
    trainer.state.log_history
)


training_history_file = (

    REPORT_DIR
    / "bert_training_history.csv"
)


training_history.to_csv(

    training_history_file,

    index=False
)


# ============================================================
# 25. SAVE TRAINER METRICS
# ============================================================

trainer_metrics = dict(
    train_result.metrics
)


trainer_metrics[
    "actual_training_time_seconds"
] = training_time


trainer_metrics[
    "actual_training_time_minutes"
] = (
    training_time
    / 60
)


trainer_metrics_file = (

    REPORT_DIR
    / "bert_training_metrics.csv"
)


pd.DataFrame(
    [trainer_metrics]
).to_csv(

    trainer_metrics_file,

    index=False
)


# ============================================================
# 26. FINAL SUMMARY
# ============================================================

print("\n==========================================")
print("       BERT TRAINING COMPLETE")
print("==========================================")


print(
    "Best checkpoint:"
)

print(
    trainer.state.best_model_checkpoint
)


print(
    "\nBest validation Macro F1:"
)


if (
    trainer.state.best_metric
    is not None
):

    print(
        f"{trainer.state.best_metric:.4f}"
    )

else:

    print(
        "N/A"
    )


print(
    "\nActual total training time:"
)

print(
    format_time(
        training_time
    )
)


print(
    "\nCUDA used:",
    USE_CUDA
)


if USE_CUDA:

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


print(
    "\nFinal model:"
)

print(
    FINAL_MODEL_DIR
)


print(
    "\nReports:"
)

print(
    validation_report_file
)

print(
    training_history_file
)

print(
    trainer_metrics_file
)


print(
    "\n=========================================="
)

print(
    "NEXT STEP:"
)

print(
    "Run bert_final_test.py using the "
    "untouched test_dataset.csv."
)

print(
    "=========================================="
)