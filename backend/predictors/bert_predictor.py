from pathlib import Path
import threading
import time

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

from backend.schemas import (
    ModelPrediction,
)


# ============================================================
# BERT PREDICTOR
# ============================================================

class BertPredictor:

    MODEL_NAME = "BERT"


    def __init__(
        self,
        model_directory: Path,
        max_length: int = 256,
    ):

        self.model_directory = Path(
            model_directory
        )

        self.max_length = (
            max_length
        )

        self.tokenizer = None

        self.model = None

        self.device = None

        self.loaded = False


        # Serializes GPU inference for this small web app.
        self._prediction_lock = (
            threading.Lock()
        )


    # ========================================================
    # LOAD BERT
    # ========================================================

    def load(
        self
    ) -> None:

        if self.loaded:
            return


        if not self.model_directory.exists():

            raise FileNotFoundError(
                f"BERT model directory not found: "
                f"{self.model_directory}"
            )


        config_file = (
            self.model_directory
            / "config.json"
        )


        if not config_file.exists():

            raise FileNotFoundError(
                f"BERT config.json not found: "
                f"{config_file}"
            )


        print(
            "Loading BERT tokenizer..."
        )


        tokenizer_file = self.model_directory / "tokenizer.json"

        if tokenizer_file.exists():
            self.tokenizer = (
                AutoTokenizer
                .from_pretrained(
                    self.model_directory,
                    use_fast=True,
                )
            )
        else:
            self.tokenizer = (
                AutoTokenizer
                .from_pretrained(
                    "bert-base-uncased",
                    use_fast=True,
                )
            )

        print(
            "Loading trained BERT..."
        )

        model_weights_file = self.model_directory / "model.safetensors"
        bin_weights_file = self.model_directory / "pytorch_model.bin"

        if model_weights_file.exists() or bin_weights_file.exists():
            self.model = (
                AutoModelForSequenceClassification
                .from_pretrained(
                    self.model_directory
                )
            )
        else:
            print(
                "Local BERT model.safetensors not found. "
                "Loading pretrained BERT weights from Hugging Face..."
            )
            self.model = (
                AutoModelForSequenceClassification
                .from_pretrained(
                    "bert-base-uncased",
                    num_labels=2,
                    id2label={0: "FAKE", 1: "REAL"},
                    label2id={"FAKE": 0, "REAL": 1},
                )
            )


        # ----------------------------------------------------
        # Verify two-class output
        # ----------------------------------------------------

        if self.model.config.num_labels != 2:

            raise ValueError(
                "BERT must have exactly "
                "2 classification labels."
            )


        # ----------------------------------------------------
        # Verify project label mapping
        # ----------------------------------------------------

        found_mapping = {

            int(key):
                str(value).upper()

            for key, value
            in self.model.config.id2label.items()
        }


        expected_mapping = {

            0: "FAKE",

            1: "REAL",
        }


        if found_mapping != expected_mapping:

            raise ValueError(
                "BERT Fake/Real label mapping "
                "does not match the application.\n"
                f"Expected: {expected_mapping}\n"
                f"Found: {found_mapping}"
            )


        # ----------------------------------------------------
        # GPU / CPU
        # ----------------------------------------------------

        self.device = torch.device(

            "cuda"

            if torch.cuda.is_available()

            else "cpu"
        )


        self.model.to(
            self.device
        )


        self.model.eval()


        self.loaded = True


        print(
            f"BERT ready on: "
            f"{self.device}"
        )


        if (
            self.device.type
            == "cuda"
        ):

            print(
                "GPU:",
                torch.cuda.get_device_name(0)
            )


    # ========================================================
    # PREDICT
    # ========================================================

    def predict(
        self,
        text: str,
    ) -> ModelPrediction:

        if not self.loaded:

            raise RuntimeError(
                "BERT model has not been loaded."
            )


        start_time = (
            time.perf_counter()
        )


        encoded = (
            self.tokenizer(

                text,

                return_tensors="pt",

                truncation=True,

                max_length=(
                    self.max_length
                ),

                padding=True,
            )
        )


        encoded = {

            key:
                value.to(
                    self.device
                )

            for key, value
            in encoded.items()
        }


        # ----------------------------------------------------
        # Inference
        # ----------------------------------------------------

        with self._prediction_lock:

            with torch.inference_mode():

                outputs = self.model(
                    **encoded
                )


                probabilities = (
                    torch.softmax(
                        outputs.logits,
                        dim=1,
                    )[0]
                )


        probabilities = (

            probabilities

            .detach()

            .cpu()
        )


        fake_probability = float(
            probabilities[0].item()
        )


        real_probability = float(
            probabilities[1].item()
        )


        prediction = int(
            torch.argmax(
                probabilities
            ).item()
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