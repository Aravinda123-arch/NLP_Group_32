import os
import pickle
import logging


import numpy as np


from sklearn.svm import LinearSVC


from sklearn.metrics import (

    accuracy_score,

    precision_score,

    recall_score,

    f1_score,

    classification_report

)



from sklearn.model_selection import GridSearchCV






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

    MODEL_FOLDER,

    exist_ok=True

)



os.makedirs(

    RESULT_FOLDER,

    exist_ok=True

)



os.makedirs(

    LOG_FOLDER,

    exist_ok=True

)







# ============================================================
# LOGGING
# ============================================================


logging.basicConfig(

    filename=os.path.join(

        LOG_FOLDER,

        "svm_training.log"

    ),

    level=logging.INFO,

    format=

    "%(asctime)s | %(levelname)s | %(message)s"

)



logging.info(
    "SVM training started"
)






# ============================================================
# LOAD FEATURES
# ============================================================


def load_features():


    try:


        files = {}


        filenames = [

            "X_train_tfidf.pkl",

            "X_test_tfidf.pkl",

            "y_train.pkl",

            "y_test.pkl"

        ]



        for file_name in filenames:


            with open(

                os.path.join(

                    FEATURE_FOLDER,

                    file_name

                ),

                "rb"

            ) as file:


                files[file_name] = pickle.load(

                    file

                )



        logging.info(
            "Features loaded successfully"
        )


        return (

            files["X_train_tfidf.pkl"],

            files["X_test_tfidf.pkl"],

            files["y_train.pkl"],

            files["y_test.pkl"]

        )



    except Exception as e:


        logging.error(

            f"Feature loading error: {e}"

        )


        raise e







# ============================================================
# CREATE SVM MODEL
# ============================================================


def create_model():


    try:


        svm = LinearSVC()


        parameters = {


            "C":

            [

                0.1,

                1,

                10

            ],


            "loss":

            [

                "hinge",

                "squared_hinge"

            ]

        }



        grid_search = GridSearchCV(

            estimator=svm,

            param_grid=parameters,

            cv=5,

            scoring="accuracy",

            n_jobs=-1

        )



        return grid_search



    except Exception as e:


        logging.error(

            f"Model creation error: {e}"

        )


        raise e







# ============================================================
# TRAIN MODEL
# ============================================================


def train_model(

        model,

        X_train,

        y_train

):


    try:


        model.fit(

            X_train,

            y_train

        )


        logging.info(

            "SVM model trained successfully"

        )


        return model



    except Exception as e:


        logging.error(

            f"Training error: {e}"

        )


        raise e






# ============================================================
# EVALUATION
# ============================================================


def evaluate_model(

        model,

        X_test,

        y_test

):


    try:


        predictions = model.predict(

            X_test

        )



        accuracy = accuracy_score(

            y_test,

            predictions

        )



        precision = precision_score(

            y_test,

            predictions,

            average="weighted"

        )



        recall = recall_score(

            y_test,

            predictions,

            average="weighted"

        )



        f1 = f1_score(

            y_test,

            predictions,

            average="weighted"

        )



        report = classification_report(

            y_test,

            predictions

        )



        results = f"""

SVM MODEL RESULTS
==========================


Accuracy  : {accuracy:.4f}

Precision : {precision:.4f}

Recall    : {recall:.4f}

F1 Score  : {f1:.4f}



Classification Report:

{report}


Best Parameters:

{model.best_params_}

"""


        return results



    except Exception as e:


        logging.error(

            f"Evaluation error: {e}"

        )


        raise e







# ============================================================
# SAVE MODEL
# ============================================================


def save_model(model):


    try:


        path = os.path.join(

            MODEL_FOLDER,

            "svm_model.pkl"

        )



        with open(

            path,

            "wb"

        ) as file:


            pickle.dump(

                model,

                file

            )



        logging.info(

            "SVM model saved"

        )


        return path



    except Exception as e:


        logging.error(

            f"Model saving error: {e}"

        )


        raise e







# ============================================================
# SAVE RESULTS
# ============================================================


def save_results(results):


    try:


        file_path = os.path.join(

            RESULT_FOLDER,

            "svm_results.txt"

        )



        with open(

            file_path,

            "w",

            encoding="utf-8"

        ) as file:


            file.write(

                results

            )



        logging.info(

            "Results saved"

        )



    except Exception as e:


        logging.error(

            f"Result saving error: {e}"

        )






# ============================================================
# MAIN
# ============================================================


def main():


    try:


        print(
            "Loading features..."
        )


        (

            X_train,

            X_test,

            y_train,

            y_test

        ) = load_features()



        print(
            "Creating SVM model..."
        )


        model = create_model()



        print(
            "Training SVM..."
        )


        model = train_model(

            model,

            X_train,

            y_train

        )



        print(
            "Evaluating model..."
        )


        results = evaluate_model(

            model,

            X_test,

            y_test

        )



        print(results)



        save_model(

            model

        )


        save_results(

            results

        )



        print(

            "\nSVM training completed!"

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