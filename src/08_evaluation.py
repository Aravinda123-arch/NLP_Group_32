import os
import pickle
import logging


import numpy as np
import pandas as pd


import matplotlib.pyplot as plt



from sklearn.metrics import (

    accuracy_score,

    precision_score,

    recall_score,

    f1_score,

    confusion_matrix,

    classification_report,

    roc_curve,

    auc

)



from tensorflow.keras.models import load_model





# ============================================================
# PATH CONFIGURATION
# ============================================================


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)



FEATURE_FOLDER = os.path.join(

    BASE_DIR,

    "data",

    "features"

)



MODEL_FOLDER = os.path.join(

    BASE_DIR,

    "models"

)



RESULT_FOLDER = os.path.join(

    BASE_DIR,

    "results"

)



LOG_FOLDER = os.path.join(

    BASE_DIR,

    "logs"

)



os.makedirs(

    RESULT_FOLDER,

    exist_ok=True

)





# ============================================================
# LOGGING
# ============================================================


logging.basicConfig(

    filename=os.path.join(

        LOG_FOLDER,

        "evaluation.log"

    ),

    level=logging.INFO,

    format=

    "%(asctime)s | %(levelname)s | %(message)s"

)


logging.info(
    "Evaluation started"
)





# ============================================================
# LOAD SVM DATA
# ============================================================


def load_svm_data():


    try:


        with open(

            os.path.join(

                FEATURE_FOLDER,

                "X_test_tfidf.pkl"

            ),

            "rb"

        ) as file:


            X_test = pickle.load(file)



        with open(

            os.path.join(

                FEATURE_FOLDER,

                "y_test.pkl"

            ),

            "rb"

        ) as file:


            y_test = pickle.load(file)



        with open(

            os.path.join(

                MODEL_FOLDER,

                "svm_model.pkl"

            ),

            "rb"

        ) as file:


            model = pickle.load(file)



        return (

            model,

            X_test,

            y_test

        )



    except Exception as e:


        logging.error(

            f"SVM loading error: {e}"

        )


        raise e







# ============================================================
# LOAD GRU DATA
# ============================================================


def load_gru_data():


    try:


        X_test = np.load(

            os.path.join(

                FEATURE_FOLDER,

                "X_test_sequences.npy"

            )

        )



        with open(

            os.path.join(

                FEATURE_FOLDER,

                "y_test.pkl"

            ),

            "rb"

        ) as file:


            y_test = pickle.load(file)




        model = load_model(

            os.path.join(

                MODEL_FOLDER,

                "gru_model.h5"

            )

        )



        return (

            model,

            X_test,

            y_test

        )



    except Exception as e:


        logging.error(

            f"GRU loading error: {e}"

        )


        raise e






# ============================================================
# METRIC CALCULATION
# ============================================================


def calculate_metrics(

        y_true,

        y_pred

):


    return {


        "Accuracy":

        accuracy_score(

            y_true,

            y_pred

        ),


        "Precision":

        precision_score(

            y_true,

            y_pred,

            zero_division=0

        ),



        "Recall":

        recall_score(

            y_true,

            y_pred,

            zero_division=0

        ),



        "F1 Score":

        f1_score(

            y_true,

            y_pred,

            zero_division=0

        )

    }






# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================


def save_confusion_matrix(

        y_true,

        y_pred,

        filename

):


    try:


        matrix = confusion_matrix(

            y_true,

            y_pred

        )



        plt.figure(

            figsize=(5,5)

        )



        plt.imshow(

            matrix

        )



        plt.title(

            "Confusion Matrix"

        )



        plt.xlabel(

            "Predicted"

        )



        plt.ylabel(

            "Actual"

        )



        for i in range(

            len(matrix)

        ):


            for j in range(

                len(matrix[i])

            ):


                plt.text(

                    j,

                    i,

                    matrix[i][j],

                    ha="center",

                    va="center"

                )



        plt.colorbar()



        plt.savefig(

            os.path.join(

                RESULT_FOLDER,

                filename

            )

        )



        plt.close()



    except Exception as e:


        logging.error(

            f"Confusion matrix error: {e}"

        )








# ============================================================
# SVM EVALUATION
# ============================================================


def evaluate_svm():


    model, X_test, y_test = load_svm_data()



    predictions = model.predict(

        X_test

    )



    metrics = calculate_metrics(

        y_test,

        predictions

    )



    save_confusion_matrix(

        y_test,

        predictions,

        "confusion_matrix_svm.png"

    )



    report = classification_report(

        y_test,

        predictions

    )



    with open(

        os.path.join(

            RESULT_FOLDER,

            "svm_evaluation.txt"

        ),

        "w"

    ) as file:


        file.write(

            str(metrics)

        )


        file.write(

            "\n\n"

        )


        file.write(

            report

        )



    return metrics






# ============================================================
# GRU EVALUATION
# ============================================================


def evaluate_gru():


    model, X_test, y_test = load_gru_data()



    probabilities = model.predict(

        X_test

    )



    predictions = (

        probabilities > 0.5

    ).astype(int)



    predictions = predictions.flatten()



    metrics = calculate_metrics(

        y_test,

        predictions

    )



    save_confusion_matrix(

        y_test,

        predictions,

        "confusion_matrix_gru.png"

    )



    report = classification_report(

        y_test,

        predictions

    )



    with open(

        os.path.join(

            RESULT_FOLDER,

            "gru_evaluation.txt"

        ),

        "w"

    ) as file:


        file.write(

            str(metrics)

        )


        file.write(

            "\n\n"

        )


        file.write(

            report

        )



    return (

        metrics,

        probabilities.flatten()

    )







# ============================================================
# ROC CURVE
# ============================================================


def plot_roc(

        y_test,

        svm_model,

        gru_probability

):


    try:


        svm_scores = svm_model.decision_function(

            load_svm_data()[1]

        )



        fpr_svm, tpr_svm, _ = roc_curve(

            y_test,

            svm_scores

        )



        fpr_gru, tpr_gru, _ = roc_curve(

            y_test,

            gru_probability

        )



        plt.figure(

            figsize=(8,6)

        )



        plt.plot(

            fpr_svm,

            tpr_svm,

            label="SVM"

        )


        plt.plot(

            fpr_gru,

            tpr_gru,

            label="GRU"

        )



        plt.xlabel(

            "False Positive Rate"

        )


        plt.ylabel(

            "True Positive Rate"

        )


        plt.title(

            "ROC Curve Comparison"

        )


        plt.legend()



        plt.savefig(

            os.path.join(

                RESULT_FOLDER,

                "roc_curve_comparison.png"

            )

        )


        plt.close()



    except Exception as e:


        logging.error(

            f"ROC error: {e}"

        )







# ============================================================
# MAIN
# ============================================================


def main():


    try:


        print(
            "Evaluating SVM..."
        )


        svm_metrics = evaluate_svm()



        print(
            "Evaluating GRU..."
        )


        gru_metrics, gru_prob = evaluate_gru()



        comparison = pd.DataFrame(

            [

                svm_metrics,

                gru_metrics

            ],

            index=[

                "SVM",

                "GRU"

            ]

        )



        comparison.to_csv(

            os.path.join(

                RESULT_FOLDER,

                "model_comparison.csv"

            )

        )



        y_test = load_gru_data()[2]



        plot_roc(

            y_test,

            load_svm_data()[0],

            gru_prob

        )



        print(

            "\nMODEL COMPARISON"

        )

        print(

            comparison

        )



        print(

            "\nEvaluation completed!"

        )



        logging.info(

            "Evaluation completed successfully"

        )



    except Exception as e:


        logging.error(

            f"Execution failed: {e}"

        )


        print(

            f"Error: {e}"

        )






if __name__ == "__main__":

    main()