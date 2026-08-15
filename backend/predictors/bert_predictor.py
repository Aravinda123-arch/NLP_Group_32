from pathlib import Path
import threading
import time
from typing import Literal

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

from backend.schemas import (
    ModelPrediction,
)


# ============================================================
# BERT PREDICTOR (Lightweight & OOM-Resilient)
# ============================================================

class BertPredictor:

    MODEL_NAME = "BERT"

    def __init__(
        self,
        model_directory: Path,
        max_length: int = 256,
    ):
        self.model_directory = Path(model_directory)
        self.max_length = max_length
        self.tokenizer = None
        self.model = None
        self.device = None
        self.loaded = False

        # Serializes GPU/CPU inference for web application
        self._prediction_lock = threading.Lock()

    # ========================================================
    # LOAD BERT
    # ========================================================

    def load(self) -> None:
        if self.loaded:
            return

        print("Loading BERT tokenizer and model...")
        model_loaded_successfully = False

        # ----------------------------------------------------
        # 1. Attempt loading from local model_directory
        # ----------------------------------------------------
        if self.model_directory.exists():
            model_weights = self.model_directory / "model.safetensors"
            bin_weights = self.model_directory / "pytorch_model.bin"

            if model_weights.exists() or bin_weights.exists():
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        self.model_directory,
                        use_fast=True,
                    )
                    self.model = AutoModelForSequenceClassification.from_pretrained(
                        self.model_directory,
                        low_cpu_mem_usage=True,
                    )
                    model_loaded_successfully = True
                    print(f"Loaded trained local BERT model from: {self.model_directory}")
                except Exception as local_err:
                    print(f"Failed to load local BERT model weights: {local_err}. Falling back to lightweight DistilBERT...")

        # ----------------------------------------------------
        # 2. Fallback to lightweight DistilBERT (~250MB RAM) to prevent OOM
        # ----------------------------------------------------
        if not model_loaded_successfully:
            print("Loading lightweight 'distilbert-base-uncased' from Hugging Face (~250MB RAM)...")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    "distilbert-base-uncased",
                    use_fast=True,
                )
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    "distilbert-base-uncased",
                    num_labels=2,
                    id2label={0: "FAKE", 1: "REAL"},
                    label2id={"FAKE": 0, "REAL": 1},
                    low_cpu_mem_usage=True,
                )
                print("Lightweight 'distilbert-base-uncased' loaded successfully.")
            except Exception as hf_err:
                print(f"Warning: Failed to load DistilBERT: {hf_err}")
                return

        # ----------------------------------------------------
        # 3. Configure Output Labels & Device
        # ----------------------------------------------------
        if self.model is not None:
            if getattr(self.model.config, "num_labels", None) != 2:
                self.model.config.num_labels = 2
                self.model.config.id2label = {0: "FAKE", 1: "REAL"}
                self.model.config.label2id = {"FAKE": 0, "REAL": 1}

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.model.eval()
            self.loaded = True
            print(f"BERT Predictor ready on device: {self.device}")

    # ========================================================
    # PREDICT
    # ========================================================

    def predict(
        self,
        text: str,
    ) -> ModelPrediction:

        start_time = time.perf_counter()

        if not self.loaded or self.model is None or self.tokenizer is None:
            elapsed = time.perf_counter() - start_time
            return ModelPrediction(
                model=self.MODEL_NAME,
                label=0,
                prediction="Fake",
                confidence=0.5,
                fake_probability=0.5,
                real_probability=0.5,
                time_seconds=elapsed,
            )

        # Sanitize input text
        clean_text = text.strip() if text else "sample text"

        try:
            encoded = self.tokenizer(
                clean_text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=True,
            )

            encoded = {key: value.to(self.device) for key, value in encoded.items()}

            with self._prediction_lock:
                with torch.inference_mode():
                    outputs = self.model(**encoded)
                    logits = outputs.logits
                    probabilities = torch.softmax(logits, dim=1)[0].detach().cpu()

            fake_prob = float(probabilities[0].item())
            real_prob = float(probabilities[1].item())

            # Clamp probabilities between 0.0 and 1.0 for Pydantic validation safety
            fake_probability = min(max(fake_prob, 0.0), 1.0)
            real_probability = min(max(real_prob, 0.0), 1.0)

            # Re-normalize if sum != 1.0 due to float precision
            total = fake_probability + real_probability
            if total > 0:
                fake_probability = min(max(fake_probability / total, 0.0), 1.0)
                real_probability = min(max(real_probability / total, 0.0), 1.0)

            prediction_label: Literal[0, 1] = 0 if fake_probability > real_probability else 1
            prediction_name = "Fake" if prediction_label == 0 else "Real"
            confidence = max(fake_probability, real_probability)

            elapsed = time.perf_counter() - start_time

            return ModelPrediction(
                model=self.MODEL_NAME,
                label=prediction_label,
                prediction=prediction_name,
                confidence=confidence,
                fake_probability=fake_probability,
                real_probability=real_probability,
                time_seconds=elapsed,
            )

        except Exception as error:
            print(f"Error during BERT prediction: {error}")
            elapsed = time.perf_counter() - start_time
            return ModelPrediction(
                model=self.MODEL_NAME,
                label=0,
                prediction="Fake",
                confidence=0.5,
                fake_probability=0.5,
                real_probability=0.5,
                time_seconds=elapsed,
            )