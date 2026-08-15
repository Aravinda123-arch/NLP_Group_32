import time

from backend.preprocessing import (
    build_news_text,
    validate_news_text,
)

from backend.decision_service import (
    calculate_final_prediction,
)

from backend.model_manager import (
    model_manager,
)

from backend.schemas import (
    InputInformation,
    PredictionResponse,
)


# ============================================================
# COMPLETE PREDICTION PROCESS
# ============================================================

def predict_news(
    headline: str,
    article: str,
) -> PredictionResponse:

    total_start = (
        time.perf_counter()
    )


    # ========================================================
    # CHECK MODEL MANAGER
    # ========================================================

    if not model_manager.loaded:

        raise RuntimeError(
            "Prediction models are not loaded."
        )


    if (
        model_manager.random_forest
        is None
    ):

        raise RuntimeError(
            "Random Forest is unavailable."
        )


    if (
        model_manager.bert
        is None
    ):

        raise RuntimeError(
            "BERT is unavailable."
        )


    # ========================================================
    # BUILD / CLEAN USER INPUT
    # ========================================================

    cleaned_text = build_news_text(

        headline=headline,

        article=article,
    )


    word_count = (
        validate_news_text(
            cleaned_text
        )
    )


    # ========================================================
    # ANALYZING NEWS
    # ========================================================
    #
    # This is the web-app equivalent of your old:
    #
    #     ANALYZING NEWS
    #
    # terminal section.
    #
    # No terminal output is required here.
    # ========================================================


    # --------------------------------------------------------
    # Random Forest ML
    # --------------------------------------------------------

    rf_result = (
        model_manager
        .random_forest
        .predict(
            cleaned_text
        )
    )


    # --------------------------------------------------------
    # BERT DL
    # --------------------------------------------------------

    bert_result = (
        model_manager
        .bert
        .predict(
            cleaned_text
        )
    )


    # --------------------------------------------------------
    # Final decision
    # --------------------------------------------------------

    final_result = (
        calculate_final_prediction(

            random_forest_result=(
                rf_result
            ),

            bert_result=(
                bert_result
            ),
        )
    )


    total_time = (
        time.perf_counter()
        - total_start
    )


    # ========================================================
    # STANDARD RESPONSE
    # ========================================================

    return PredictionResponse(

        input=InputInformation(
            word_count=word_count
        ),

        models=[
            rf_result,
            bert_result,
        ],

        final=final_result,

        total_time_seconds=float(
            total_time
        ),
    )