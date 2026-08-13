from pathlib import Path
import html
import re
import sys
import time
import unicodedata

import joblib
import numpy as np
import pandas as pd
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent


RF_MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "random_forest_model.pkl"
)

TFIDF_FILE = (
    PROJECT_ROOT
    / "models"
    / "tfidf_vectorizer.pkl"
)

BERT_MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "bert_fake_news_final"
)


# Must match BERT training.
BERT_MAX_LENGTH = 256


# Labels
# 0 = Fake
# 1 = Real
LABEL_MAPPING = {
    0: "Fake",
    1: "Real"
}


# Minimum recommended number of words.
MIN_WORDS = 20


# ============================================================
# 2. TEXT CLEANING
# ============================================================

def clean_news_text(text):

    if text is None:
        return ""

    text = str(text)

    # Decode HTML entities.
    text = html.unescape(text)

    # Unicode normalization.
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

    # Remove Reuters / AP / AFP prefixes.
    text = re.sub(
        r"^\s*(?:REUTERS|AP|AFP)\s*[-–—:]\s*",
        " ",
        text,
        flags=re.IGNORECASE
    )

    # Remove reporting/editing credits.
    text = re.sub(
        r"\b(?:reporting by|additional reporting by|editing by)\b.*$",
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

    # Match training text style.
    text = text.lower()

    # Normalize whitespace.
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# 3. INPUT BUILDER
# ============================================================

def build_news_text(
    headline,
    article
):

    headline = (
        headline.strip()
        if headline
        else ""
    )

    article = (
        article.strip()
        if article
        else ""
    )


    combined_text = " ".join(
        part
        for part in [
            headline,
            article
        ]
        if part
    )


    return clean_news_text(
        combined_text
    )


# ============================================================
# 4. CHECK MODEL FILES
# ============================================================

def check_required_files():

    missing = []


    if not RF_MODEL_FILE.exists():

        missing.append(
            RF_MODEL_FILE
        )


    if not TFIDF_FILE.exists():

        missing.append(
            TFIDF_FILE
        )


    if not BERT_MODEL_DIR.exists():

        missing.append(
            BERT_MODEL_DIR
        )


    if missing:

        print(
            "\nERROR: Required model files are missing:"
        )

        for path in missing:

            print(
                " -",
                path
            )

        sys.exit(1)


    if not (
        BERT_MODEL_DIR
        / "config.json"
    ).exists():

        raise FileNotFoundError(
            "BERT config.json is missing."
        )


# ============================================================
# 5. LOAD RANDOM FOREST + TF-IDF
# ============================================================

def load_random_forest():

    print(
        "\nLoading Random Forest..."
    )


    model_package = joblib.load(
        RF_MODEL_FILE
    )


    if isinstance(
        model_package,
        dict
    ):

        if "model" not in model_package:

            raise ValueError(
                "random_forest_model.pkl "
                "does not contain a 'model' key."
            )


        rf_model = (
            model_package["model"]
        )


        saved_feature_names = (
            model_package.get(
                "feature_names"
            )
        )

    else:

        rf_model = (
            model_package
        )

        saved_feature_names = None


    # --------------------------------------------------------
    # Disable Random Forest verbose output.
    #
    # Removes:
    #
    # [Parallel(n_jobs=...)]
    # --------------------------------------------------------

    if hasattr(
        rf_model,
        "verbose"
    ):

        rf_model.verbose = 0


    print(
        "Loading fitted TF-IDF vectorizer..."
    )


    tfidf_vectorizer = joblib.load(
        TFIDF_FILE
    )


    vectorizer_feature_names = (
        tfidf_vectorizer
        .get_feature_names_out()
    )


    number_vectorizer_features = len(
        vectorizer_feature_names
    )


    # --------------------------------------------------------
    # Check feature count compatibility
    # --------------------------------------------------------

    if hasattr(
        rf_model,
        "n_features_in_"
    ):

        number_model_features = (
            rf_model.n_features_in_
        )


        if (
            number_model_features
            != number_vectorizer_features
        ):

            raise ValueError(
                "\nRandom Forest / TF-IDF mismatch!\n"
                f"Random Forest expects: "
                f"{number_model_features} features\n"
                f"TF-IDF contains: "
                f"{number_vectorizer_features} features\n\n"
                "Use the Random Forest and TF-IDF "
                "vectorizer from the same training run."
            )


    # --------------------------------------------------------
    # Exact feature-name compatibility check
    # --------------------------------------------------------

    if saved_feature_names is not None:

        saved_feature_names = np.asarray(
            saved_feature_names,
            dtype=str
        )


        vectorizer_feature_names_array = np.asarray(
            vectorizer_feature_names,
            dtype=str
        )


        if (
            len(saved_feature_names)
            != len(
                vectorizer_feature_names_array
            )
        ):

            raise ValueError(
                "Random Forest and TF-IDF "
                "feature counts do not match."
            )


        if not np.array_equal(
            saved_feature_names,
            vectorizer_feature_names_array
        ):

            raise ValueError(
                "\nRandom Forest feature names "
                "do not match the TF-IDF vectorizer.\n"
                "Use artifacts from the same "
                "feature-engineering run."
            )


    print(
        "Random Forest loaded successfully."
    )

    print(
        "TF-IDF features:",
        number_vectorizer_features
    )


    return (
        rf_model,
        tfidf_vectorizer
    )


# ============================================================
# 6. LOAD BERT MODEL
# ============================================================

def load_bert():

    print(
        "\nLoading BERT tokenizer..."
    )


    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            BERT_MODEL_DIR,
            use_fast=True
        )
    )


    print(
        "Loading trained BERT model..."
    )


    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            BERT_MODEL_DIR
        )
    )


    # --------------------------------------------------------
    # Check label mapping
    # --------------------------------------------------------

    found_mapping = {

        int(key):
            str(value).upper()

        for key, value
        in model.config.id2label.items()
    }


    expected_mapping = {

        0: "FAKE",
        1: "REAL"
    }


    if (
        found_mapping
        != expected_mapping
    ):

        raise ValueError(
            "\nUnexpected BERT label mapping.\n\n"
            f"Expected: {expected_mapping}\n"
            f"Found: {found_mapping}\n\n"
            "Stopping because Fake / Real labels "
            "could otherwise be reversed."
        )


    # --------------------------------------------------------
    # Select GPU / CPU
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    model.to(
        device
    )


    model.eval()


    print(
        "BERT loaded successfully."
    )


    print(
        "BERT device:",
        device
    )


    if (
        device.type
        == "cuda"
    ):

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )


    return (
        tokenizer,
        model,
        device
    )


# ============================================================
# 7. PREPARE RANDOM FOREST FEATURES
# ============================================================

def prepare_rf_features(
    text,
    rf_model,
    tfidf_vectorizer
):

    # IMPORTANT:
    #
    # transform()
    #
    # NOT fit_transform()
    #
    # because the vectorizer is already fitted.

    sparse_features = (
        tfidf_vectorizer
        .transform(
            [text]
        )
    )


    # --------------------------------------------------------
    # Fix sklearn feature-name warning.
    #
    # If Random Forest was trained using named DataFrame
    # columns, create a DataFrame using the same names.
    # --------------------------------------------------------

    if hasattr(
        rf_model,
        "feature_names_in_"
    ):

        feature_names = (
            tfidf_vectorizer
            .get_feature_names_out()
        )


        features = (
            pd.DataFrame.sparse
            .from_spmatrix(
                sparse_features,
                columns=feature_names
            )
        )


        # Use exactly the order expected by RF.
        features = features[
            rf_model.feature_names_in_
        ]


    else:

        features = (
            sparse_features
        )


    return features


# ============================================================
# 8. RANDOM FOREST PREDICTION
# ============================================================

def predict_random_forest(
    text,
    rf_model,
    tfidf_vectorizer
):

    start_time = (
        time.time()
    )


    features = prepare_rf_features(

        text=text,

        rf_model=rf_model,

        tfidf_vectorizer=(
            tfidf_vectorizer
        )
    )


    prediction = int(
        rf_model.predict(
            features
        )[0]
    )


    fake_probability = None
    real_probability = None


    if hasattr(
        rf_model,
        "predict_proba"
    ):

        probabilities = (
            rf_model.predict_proba(
                features
            )[0]
        )


        classes = list(
            rf_model.classes_
        )


        if 0 in classes:

            fake_probability = float(
                probabilities[
                    classes.index(0)
                ]
            )


        if 1 in classes:

            real_probability = float(
                probabilities[
                    classes.index(1)
                ]
            )


    if (
        fake_probability is not None
        and
        real_probability is not None
    ):

        confidence = float(
            max(
                fake_probability,
                real_probability
            )
        )

    else:

        confidence = None


    elapsed = (
        time.time()
        - start_time
    )


    return {

        "model":
            "Random Forest",

        "label":
            prediction,

        "prediction":
            LABEL_MAPPING[
                prediction
            ],

        "fake_probability":
            fake_probability,

        "real_probability":
            real_probability,

        "confidence":
            confidence,

        "time_seconds":
            elapsed
    }


# ============================================================
# 9. BERT PREDICTION
# ============================================================

def predict_bert(
    text,
    tokenizer,
    model,
    device
):

    start_time = (
        time.time()
    )


    encoded = tokenizer(

        text,

        return_tensors="pt",

        truncation=True,

        max_length=BERT_MAX_LENGTH,

        padding=True
    )


    encoded = {

        key:
            value.to(
                device
            )

        for key, value
        in encoded.items()
    }


    # Inference only.
    # No gradients / no model training.
    with torch.inference_mode():

        outputs = model(
            **encoded
        )


        probabilities = (
            torch.softmax(
                outputs.logits,
                dim=1
            )[0]
        )


    probabilities = (

        probabilities

        .detach()

        .cpu()

        .numpy()
    )


    fake_probability = float(
        probabilities[0]
    )


    real_probability = float(
        probabilities[1]
    )


    prediction = int(
        np.argmax(
            probabilities
        )
    )


    confidence = float(
        max(
            fake_probability,
            real_probability
        )
    )


    elapsed = (
        time.time()
        - start_time
    )


    return {

        "model":
            "BERT",

        "label":
            prediction,

        "prediction":
            LABEL_MAPPING[
                prediction
            ],

        "fake_probability":
            fake_probability,

        "real_probability":
            real_probability,

        "confidence":
            confidence,

        "time_seconds":
            elapsed
    }


# ============================================================
# 10. PREDICT NEWS
# ============================================================

def predict_news(
    headline,
    article,
    rf_model,
    tfidf_vectorizer,
    tokenizer,
    bert_model,
    device
):

    cleaned_text = build_news_text(
        headline,
        article
    )


    if not cleaned_text:

        raise ValueError(
            "News text cannot be empty."
        )


    word_count = len(
        cleaned_text.split()
    )


    if (
        word_count
        < MIN_WORDS
    ):

        raise ValueError(
            f"News input contains only "
            f"{word_count} words.\n"
            f"Please enter at least "
            f"{MIN_WORDS} words.\n"
            "A full news article is recommended."
        )


    # ========================================================
    # ANALYZING NEWS
    # ========================================================
    #
    # IMPORTANT:
    #
    # This section remains in the code as requested.
    # Nothing is printed to the terminal.
    #
    # This is where both models analyze the news.
    # ========================================================


    # --------------------------------------------------------
    # ML - Random Forest
    # --------------------------------------------------------

    rf_result = (
        predict_random_forest(

            text=cleaned_text,

            rf_model=rf_model,

            tfidf_vectorizer=(
                tfidf_vectorizer
            )
        )
    )


    # --------------------------------------------------------
    # DL - BERT
    # --------------------------------------------------------

    bert_result = (
        predict_bert(

            text=cleaned_text,

            tokenizer=tokenizer,

            model=bert_model,

            device=device
        )
    )


    return {

        "word_count":
            word_count,

        "random_forest":
            rf_result,

        "bert":
            bert_result
    }


# ============================================================
# 11. CALCULATE FINAL PREDICTION
# ============================================================

def calculate_final_prediction(
    rf_result,
    bert_result
):

    rf_label = (
        rf_result[
            "label"
        ]
    )


    bert_label = (
        bert_result[
            "label"
        ]
    )


    rf_confidence = (
        rf_result[
            "confidence"
        ]
    )


    bert_confidence = (
        bert_result[
            "confidence"
        ]
    )


    # ========================================================
    # CASE 1:
    # RANDOM FOREST AND BERT AGREE
    # ========================================================

    if (
        rf_label
        == bert_label
    ):

        final_label = (
            rf_label
        )


        # If both agree, calculate an average
        # confidence for display.
        if (
            rf_confidence is not None
            and
            bert_confidence is not None
        ):

            final_confidence = (

                rf_confidence
                +
                bert_confidence

            ) / 2


        elif (
            bert_confidence
            is not None
        ):

            final_confidence = (
                bert_confidence
            )


        else:

            final_confidence = (
                rf_confidence
            )


        return {

            "final_label":
                final_label,

            "final_prediction":
                LABEL_MAPPING[
                    final_label
                ],

            "final_confidence":
                final_confidence,

            "agreement":
                True,

            "selected_model":
                "Random Forest + BERT",

            "decision_method":
                "Both models agree"
        }


    # ========================================================
    # CASE 2:
    # MODELS DISAGREE
    #
    # Select the prediction from the model that has
    # the larger prediction confidence.
    # ========================================================


    if (
        rf_confidence is None
        and
        bert_confidence is None
    ):

        raise ValueError(
            "Random Forest and BERT disagree, "
            "but confidence values are unavailable."
        )


    # --------------------------------------------------------
    # Only BERT confidence available
    # --------------------------------------------------------

    if rf_confidence is None:

        selected_result = (
            bert_result
        )


    # --------------------------------------------------------
    # Only RF confidence available
    # --------------------------------------------------------

    elif bert_confidence is None:

        selected_result = (
            rf_result
        )


    # --------------------------------------------------------
    # BERT confidence is greater
    # --------------------------------------------------------

    elif (
        bert_confidence
        >
        rf_confidence
    ):

        selected_result = (
            bert_result
        )


    # --------------------------------------------------------
    # Random Forest confidence is greater
    # --------------------------------------------------------

    elif (
        rf_confidence
        >
        bert_confidence
    ):

        selected_result = (
            rf_result
        )


    # --------------------------------------------------------
    # Exact confidence tie
    #
    # BERT used as deterministic tie-breaker.
    # --------------------------------------------------------

    else:

        selected_result = (
            bert_result
        )


    return {

        "final_label":
            selected_result[
                "label"
            ],

        "final_prediction":
            selected_result[
                "prediction"
            ],

        "final_confidence":
            selected_result[
                "confidence"
            ],

        "agreement":
            False,

        "selected_model":
            selected_result[
                "model"
            ],

        "decision_method":
            (
                "Models disagree - "
                "higher-confidence prediction selected"
            )
    }


# ============================================================
# 12. DISPLAY INDIVIDUAL MODEL RESULT
# ============================================================

def display_model_result(
    result
):

    print(
        "\n------------------------------------------"
    )

    print(
        result[
            "model"
        ]
    )

    print(
        "------------------------------------------"
    )


    print(
        "Prediction:",
        result[
            "prediction"
        ].upper()
    )


    if (
        result[
            "confidence"
        ]
        is not None
    ):

        print(
            "Confidence:",
            f"{result['confidence'] * 100:.2f}%"
        )


    if (
        result[
            "fake_probability"
        ]
        is not None
    ):

        print(
            "Fake probability:",
            f"{result['fake_probability'] * 100:.2f}%"
        )


    if (
        result[
            "real_probability"
        ]
        is not None
    ):

        print(
            "Real probability:",
            f"{result['real_probability'] * 100:.2f}%"
        )


    print(
        "Prediction time:",
        f"{result['time_seconds']:.3f} seconds"
    )


# ============================================================
# 13. DISPLAY FINAL RESULT
# ============================================================

def display_final_result(
    result
):

    rf_result = (
        result[
            "random_forest"
        ]
    )


    bert_result = (
        result[
            "bert"
        ]
    )


    # --------------------------------------------------------
    # Show both model predictions
    # --------------------------------------------------------

    display_model_result(
        rf_result
    )


    display_model_result(
        bert_result
    )


    # --------------------------------------------------------
    # Calculate final decision
    # --------------------------------------------------------

    final_result = (
        calculate_final_prediction(

            rf_result=rf_result,

            bert_result=bert_result
        )
    )


    # --------------------------------------------------------
    # Final comparison
    # --------------------------------------------------------

    print(
        "\n=========================================="
    )

    print(
        "            FINAL RESULT"
    )

    print(
        "=========================================="
    )


    if (
        rf_result[
            "confidence"
        ]
        is not None
    ):

        print(
            "Random Forest:",
            (
                f"{rf_result['prediction'].upper()} "
                f"({rf_result['confidence'] * 100:.2f}%)"
            )
        )

    else:

        print(
            "Random Forest:",
            rf_result[
                "prediction"
            ].upper()
        )


    if (
        bert_result[
            "confidence"
        ]
        is not None
    ):

        print(
            "BERT:",
            (
                f"{bert_result['prediction'].upper()} "
                f"({bert_result['confidence'] * 100:.2f}%)"
            )
        )

    else:

        print(
            "BERT:",
            bert_result[
                "prediction"
            ].upper()
        )


    print()


    # ========================================================
    # SAME PREDICTION
    # ========================================================

    if final_result[
        "agreement"
    ]:

        print(
            "Model agreement: YES"
        )

        print(
            "Both models selected the same class."
        )


    # ========================================================
    # DIFFERENT PREDICTIONS
    # ========================================================

    else:

        print(
            "Model agreement: NO"
        )


        print(
            "Models produced different predictions."
        )


        print(
            "Higher-confidence prediction selected."
        )


    # ========================================================
    # FINAL ANSWER
    # ========================================================

    print(
        "\nFINAL PREDICTION:"
    )


    print(
        ">>>",
        final_result[
            "final_prediction"
        ].upper(),
        "<<<"
    )


    print(
        "\nSelected model:",
        final_result[
            "selected_model"
        ]
    )


    if (
        final_result[
            "final_confidence"
        ]
        is not None
    ):

        print(
            "Final confidence:",
            (
                f"{final_result['final_confidence'] * 100:.2f}%"
            )
        )


    print(
        "Decision method:",
        final_result[
            "decision_method"
        ]
    )


    print(
        "\n=========================================="
    )


    print(
        "NOTE:"
    )


    print(
        "This is a machine-learning prediction "
        "from the trained RF and BERT models."
    )


    print(
        "It is not independent factual "
        "verification of the article."
    )


    print(
        "=========================================="
    )


    # Important for future website API.
    return final_result


# ============================================================
# 14. INTERACTIVE NEWS INPUT
# ============================================================

def get_user_news():

    print(
        "\n=========================================="
    )

    print(
        "       FAKE NEWS DETECTION SYSTEM"
    )

    print(
        "=========================================="
    )


    print(
        "\nEnter an English news article."
    )


    print(
        "A full article is recommended."
    )


    headline = input(
        "\nNews headline "
        "(optional): "
    ).strip()


    print(
        "\nPaste news article body."
    )


    print(
        "When finished, type END "
        "on a new line."
    )


    print()


    article_lines = []


    while True:

        try:

            line = input()

        except EOFError:

            break


        if (
            line.strip().upper()
            == "END"
        ):

            break


        article_lines.append(
            line
        )


    article = "\n".join(
        article_lines
    )


    return (
        headline,
        article
    )


# ============================================================
# 15. MAIN
# ============================================================

def main():

    print(
        "\n=========================================="
    )

    print(
        "       INITIALIZING MODELS"
    )

    print(
        "=========================================="
    )


    # --------------------------------------------------------
    # Check model files
    # --------------------------------------------------------

    check_required_files()


    # --------------------------------------------------------
    # Load Random Forest + TF-IDF
    # --------------------------------------------------------

    (
        rf_model,
        tfidf_vectorizer
    ) = load_random_forest()


    # --------------------------------------------------------
    # Load BERT
    # --------------------------------------------------------

    (
        tokenizer,
        bert_model,
        device
    ) = load_bert()


    print(
        "\n=========================================="
    )

    print(
        "       ALL MODELS READY"
    )

    print(
        "=========================================="
    )


    # --------------------------------------------------------
    # Keep models loaded and allow multiple predictions.
    # --------------------------------------------------------

    while True:

        try:

            headline, article = (
                get_user_news()
            )


            result = predict_news(

                headline=headline,

                article=article,

                rf_model=rf_model,

                tfidf_vectorizer=(
                    tfidf_vectorizer
                ),

                tokenizer=tokenizer,

                bert_model=bert_model,

                device=device
            )


            display_final_result(
                result
            )


        except ValueError as error:

            print(
                "\nINPUT ERROR:"
            )

            print(
                error
            )


        except KeyboardInterrupt:

            print(
                "\n\nPrediction system stopped."
            )

            break


        except Exception as error:

            print(
                "\nPREDICTION ERROR:"
            )

            print(
                type(error).__name__,
                ":",
                error
            )


        # ----------------------------------------------------
        # Test another article
        # ----------------------------------------------------

        print(
            "\nCheck another news article?"
        )


        choice = input(
            "Enter Y for Yes or N for No: "
        ).strip().lower()


        if choice not in {
            "y",
            "yes"
        }:

            print(
                "\nPrediction system closed."
            )

            break


# ============================================================
# 16. RUN
# ============================================================

if __name__ == "__main__":

    main()