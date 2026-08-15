from pathlib import Path

import torch

from backend.predictors import (
    RandomForestPredictor,
    BertPredictor,
)


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)


MODEL_DIR = (
    PROJECT_ROOT
    / "models"
)


RF_MODEL_FILE = (
    MODEL_DIR
    / "random_forest_model.pkl"
)


TFIDF_FILE = (
    MODEL_DIR
    / "tfidf_vectorizer.pkl"
)


BERT_MODEL_DIR = (
    MODEL_DIR
    / "bert_fake_news_final"
)


# ============================================================
# MODEL MANAGER
# ============================================================

class ModelManager:

    def __init__(
        self
    ):

        self.random_forest = None

        self.bert = None

        self.loaded = False


    # ========================================================
    # LOAD ALL MODELS
    # ========================================================

    def load_models(
        self
    ) -> None:

        if self.loaded:

            return


        print(
            "\n=========================================="
        )

        print(
            "          LOADING MODELS"
        )

        print(
            "=========================================="
        )


        # ----------------------------------------------------
        # Random Forest
        # ----------------------------------------------------

        self.random_forest = (
            RandomForestPredictor(

                model_file=(
                    RF_MODEL_FILE
                ),

                vectorizer_file=(
                    TFIDF_FILE
                ),
            )
        )


        self.random_forest.load()


        # ----------------------------------------------------
        # BERT
        # ----------------------------------------------------

        self.bert = (
            BertPredictor(

                model_directory=(
                    BERT_MODEL_DIR
                ),

                max_length=256,
            )
        )


        try:
            self.bert.load()
        except Exception as error:
            print(f"Warning: BERT failed to load: {error}. Application will operate with Random Forest.")

        self.loaded = True


        print(
            "\n=========================================="
        )

        print(
            "          MODELS READY"
        )

        print(
            "=========================================="
        )


    # ========================================================
    # UNLOAD
    # ========================================================

    def unload_models(
        self
    ) -> None:

        self.random_forest = None

        self.bert = None

        self.loaded = False


        if torch.cuda.is_available():

            torch.cuda.empty_cache()


    # ========================================================
    # STATUS
    # ========================================================

    def get_status(
        self
    ) -> dict:

        rf_loaded = (
            self.random_forest
            is not None
            and
            self.random_forest.loaded
        )


        bert_loaded = (
            self.bert
            is not None
            and
            self.bert.loaded
        )


        bert_device = None


        if bert_loaded:

            bert_device = str(
                self.bert.device
            )


        return {

            "status":
                (
                    "ready"
                    if self.loaded
                    else "not_ready"
                ),

            "models_loaded":
                self.loaded,

            "random_forest_loaded":
                rf_loaded,

            "bert_loaded":
                bert_loaded,

            "bert_device":
                bert_device,
        }


# ============================================================
# GLOBAL MODEL MANAGER
# ============================================================

model_manager = (
    ModelManager()
)