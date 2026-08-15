from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd

from backend.schemas import (
    ModelPrediction,
)


# ============================================================
# RANDOM FOREST PREDICTOR
# ============================================================

class RandomForestPredictor:

    MODEL_NAME = "Random Forest"


    def __init__(
        self,
        model_file: Path,
        vectorizer_file: Path,
    ):

        self.model_file = Path(
            model_file
        )

        self.vectorizer_file = Path(
            vectorizer_file
        )

        self.model = None

        self.vectorizer = None

        self.loaded = False


    # ========================================================
    # LOAD MODEL
    # ========================================================

    def load(
        self
    ) -> None:

        if self.loaded:
            return


        if not self.model_file.exists():

            raise FileNotFoundError(
                f"Random Forest model not found: "
                f"{self.model_file}"
            )


        if not self.vectorizer_file.exists():

            raise FileNotFoundError(
                f"TF-IDF vectorizer not found: "
                f"{self.vectorizer_file}"
            )


        print(
            "Loading Random Forest..."
        )


        model_package = joblib.load(
            self.model_file
        )


        # ----------------------------------------------------
        # Support model package created during your training
        # ----------------------------------------------------

        if isinstance(
            model_package,
            dict
        ):

            if "model" not in model_package:

                raise ValueError(
                    "Random Forest package does not "
                    "contain the 'model' key."
                )


            self.model = (
                model_package[
                    "model"
                ]
            )


            saved_feature_names = (
                model_package.get(
                    "feature_names"
                )
            )


        else:

            self.model = (
                model_package
            )

            saved_feature_names = None


        # ----------------------------------------------------
        # Stop Random Forest [Parallel(...)] terminal output
        # ----------------------------------------------------

        if hasattr(
            self.model,
            "verbose"
        ):

            self.model.verbose = 0


        print(
            "Loading TF-IDF vectorizer..."
        )


        self.vectorizer = joblib.load(
            self.vectorizer_file
        )


        # ----------------------------------------------------
        # Validate TF-IDF compatibility
        # ----------------------------------------------------

        vectorizer_names = (
            self.vectorizer
            .get_feature_names_out()
        )


        vectorizer_feature_count = len(
            vectorizer_names
        )


        if hasattr(
            self.model,
            "n_features_in_"
        ):

            if (
                self.model.n_features_in_
                !=
                vectorizer_feature_count
            ):

                raise ValueError(
                    "Random Forest and TF-IDF "
                    "vectorizer do not match.\n"
                    f"RF features: "
                    f"{self.model.n_features_in_}\n"
                    f"TF-IDF features: "
                    f"{vectorizer_feature_count}"
                )


        # ----------------------------------------------------
        # Strong exact feature-name validation
        # ----------------------------------------------------

        if saved_feature_names is not None:

            saved_feature_names = (
                np.asarray(
                    saved_feature_names,
                    dtype=str,
                )
            )


            vectorizer_names_array = (
                np.asarray(
                    vectorizer_names,
                    dtype=str,
                )
            )


            if not np.array_equal(
                saved_feature_names,
                vectorizer_names_array,
            ):

                raise ValueError(
                    "Random Forest feature names "
                    "do not match the saved "
                    "TF-IDF vectorizer."
                )


        self.loaded = True


        print(
            "Random Forest ready."
        )


    # ========================================================
    # PREPARE TF-IDF FEATURES
    # ========================================================

    def _prepare_features(
        self,
        text: str,
    ):

        sparse_features = (
            self.vectorizer
            .transform(
                [text]
            )
        )


        # ----------------------------------------------------
        # If RF was trained with DataFrame feature names,
        # provide exactly the same feature names/order.
        # ----------------------------------------------------

        if hasattr(
            self.model,
            "feature_names_in_",
        ):

            feature_names = (
                self.vectorizer
                .get_feature_names_out()
            )


            features = (
                pd.DataFrame.sparse
                .from_spmatrix(
                    sparse_features,
                    columns=feature_names,
                )
            )


            features = features[
                self.model.feature_names_in_
            ]


            return features


        return sparse_features


    # ========================================================
    # PREDICT
    # ========================================================

    def predict(
        self,
        text: str,
    ) -> ModelPrediction:

        if not self.loaded:

            raise RuntimeError(
                "Random Forest has not been loaded."
            )


        start_time = (
            time.perf_counter()
        )


        features = (
            self._prepare_features(
                text
            )
        )


        prediction = int(
            self.model.predict(
                features
            )[0]
        )


        # ----------------------------------------------------
        # Probability support required for final comparison
        # ----------------------------------------------------

        if not hasattr(
            self.model,
            "predict_proba",
        ):

            raise RuntimeError(
                "Random Forest model does not "
                "support predict_proba()."
            )


        probabilities = (
            self.model.predict_proba(
                features
            )[0]
        )


        classes = list(
            self.model.classes_
        )


        if (
            0 not in classes
            or
            1 not in classes
        ):

            raise ValueError(
                "Random Forest classes must contain "
                "0 = Fake and 1 = Real."
            )


        fake_probability = float(
            probabilities[
                classes.index(0)
            ]
        )


        real_probability = float(
            probabilities[
                classes.index(1)
            ]
        )


        confidence = float(
            max(
                fake_probability,
                real_probability,
            )
        )


        elapsed = (
            time.perf_counter()
            - start_time
        )


        prediction_name = (
            "Fake"
            if prediction == 0
            else "Real"
        )


        return ModelPrediction(

            model=self.MODEL_NAME,

            label=prediction,

            prediction=prediction_name,

            confidence=confidence,

            fake_probability=(
                fake_probability
            ),

            real_probability=(
                real_probability
            ),

            time_seconds=float(
                elapsed
            ),
        )