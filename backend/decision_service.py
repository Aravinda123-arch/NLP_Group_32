from backend.schemas import (
    ModelPrediction,
    FinalPrediction,
)


# ============================================================
# FINAL MODEL DECISION
# ============================================================

def calculate_final_prediction(
    random_forest_result: ModelPrediction,
    bert_result: ModelPrediction,
) -> FinalPrediction:


    # ========================================================
    # CASE 1:
    # BOTH MODELS AGREE
    # ========================================================

    if (
        random_forest_result.label
        ==
        bert_result.label
    ):

        final_label = (
            random_forest_result.label
        )


        final_prediction = (
            random_forest_result.prediction
        )


        # Average confidence only for display
        # when both models predict the same class.
        final_confidence = (

            random_forest_result.confidence
            +
            bert_result.confidence

        ) / 2.0


        return FinalPrediction(

            label=final_label,

            prediction=final_prediction,

            confidence=float(
                final_confidence
            ),

            agreement=True,

            selected_model=(
                "Random Forest + BERT"
            ),

            decision_method=(
                "Both models agree"
            ),
        )


    # ========================================================
    # CASE 2:
    # MODELS DISAGREE
    # ========================================================

    if (
        bert_result.confidence
        >
        random_forest_result.confidence
    ):

        selected = (
            bert_result
        )


    elif (
        random_forest_result.confidence
        >
        bert_result.confidence
    ):

        selected = (
            random_forest_result
        )


    else:

        # Exact probability tie.
        # Use BERT consistently as tie breaker.
        selected = (
            bert_result
        )


    return FinalPrediction(

        label=selected.label,

        prediction=selected.prediction,

        confidence=selected.confidence,

        agreement=False,

        selected_model=selected.model,

        decision_method=(
            "Models disagree - "
            "higher-confidence prediction selected"
        ),
    )