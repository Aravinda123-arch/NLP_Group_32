from typing import Literal

from pydantic import BaseModel, Field


# ============================================================
# REQUEST
# ============================================================

class NewsRequest(BaseModel):
    """
    Data received from the frontend.
    """

    headline: str = Field(
        default="",
        max_length=1000,
        description="Optional news headline",
    )

    article: str = Field(
        default="",
        max_length=100000,
        description="News article body",
    )


# ============================================================
# INPUT INFORMATION
# ============================================================

class InputInformation(BaseModel):

    word_count: int = Field(
        ge=0
    )


# ============================================================
# SINGLE MODEL RESULT
# ============================================================

class ModelPrediction(BaseModel):

    model: str

    label: Literal[
        0,
        1
    ]

    prediction: Literal[
        "Fake",
        "Real",
    ]

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    fake_probability: float = Field(
        ge=0.0,
        le=1.0,
    )

    real_probability: float = Field(
        ge=0.0,
        le=1.0,
    )

    time_seconds: float = Field(
        ge=0.0
    )


# ============================================================
# FINAL ENSEMBLE RESULT
# ============================================================

class FinalPrediction(BaseModel):

    label: Literal[
        0,
        1
    ]

    prediction: Literal[
        "Fake",
        "Real",
    ]

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    agreement: bool

    selected_model: str

    decision_method: str


# ============================================================
# COMPLETE API RESPONSE
# ============================================================

class PredictionResponse(BaseModel):

    input: InputInformation

    models: list[
        ModelPrediction
    ]

    final: FinalPrediction

    total_time_seconds: float = Field(
        ge=0.0
    )


# ============================================================
# HEALTH RESPONSE
# ============================================================

class HealthResponse(BaseModel):

    status: str

    models_loaded: bool

    random_forest_loaded: bool

    bert_loaded: bool

    bert_device: str | None = None