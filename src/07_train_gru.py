import os
import pickle
import logging


import numpy as np


import matplotlib.pyplot as plt



from tensorflow.keras.models import Sequential


from tensorflow.keras.layers import (

    Embedding,

    GRU,

    Bidirectional,

    Dense,

    Dropout

)


from tensorflow.keras.callbacks import (

    EarlyStopping,

    ModelCheckpoint

)


from tensorflow.keras.optimizers import Adam






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

        "gru_training.log"

    ),

    level=logging.INFO,

    format=

    "%(asctime)s | %(levelname)s | %(message)s"

)



logging.info(
    "GRU training started"
)






# ============================================================
# PARAMETERS
# ============================================================


VOCAB_SIZE = 20000


EMBEDDING_DIM = 128


GRU_UNITS = 64


DROPOUT_RATE = 0.3


EPOCHS = 20


BATCH_SIZE = 64







# ============================================================
# LOAD DATA
# ============================================================


def load_data():


    try:


        X_train = np.load(

            os.path.join(

                FEATURE_FOLDER,

                "X_train_sequences.npy"

            )

        )



        X_test = np.load(

            os.path.join(

                FEATURE_FOLDER,

                "X_test_sequences.npy"

            )

        )



        with open(

            os.path.join(

                FEATURE_FOLDER,

                "y_train.pkl"

            ),

            "rb"

        ) as file:


            y_train = pickle.load(file)




        with open(

            os.path.join(

                FEATURE_FOLDER,

                "y_test.pkl"

            ),

            "rb"

        ) as file:


            y_test = pickle.load(file)




        logging.info(

            "GRU data loaded successfully"

        )



        return (

            X_train,

            X_test,

            y_train,

            y_test

        )



    except Exception as e:


        logging.error(

            f"Data loading error: {e}"

        )


        raise e







# ============================================================
# BUILD GRU MODEL
# ============================================================


def build_model(input_length):


    try:


        model = Sequential()



        model.add(

            Embedding(

                input_dim=VOCAB_SIZE,

                output_dim=EMBEDDING_DIM,

                input_length=input_length

            )

        )



        model.add(

            Bidirectional(

                GRU(

                    GRU_UNITS,

                    return_sequences=False

                )

            )

        )



        model.add(

            Dropout(

                DROPOUT_RATE

            )

        )



        model.add(

            Dense(

                64,

                activation="relu"

            )

        )



        model.add(

            Dropout(

                DROPOUT_RATE

            )

        )



        model.add(

            Dense(

                1,

                activation="sigmoid"

            )

        )




        model.compile(

            optimizer=Adam(

                learning_rate=0.001

            ),

            loss="binary_crossentropy",

            metrics=[

                "accuracy"

            ]

        )



        logging.info(

            "GRU model created"

        )



        return model



    except Exception as e:


        logging.error(

            f"Model building error: {e}"

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



        checkpoint_path = os.path.join(

            MODEL_FOLDER,

            "best_gru_model.h5"

        )



        callbacks = [



            EarlyStopping(

                monitor="val_loss",

                patience=3,

                restore_best_weights=True

            ),




            ModelCheckpoint(

                filepath=checkpoint_path,

                monitor="val_accuracy",

                save_best_only=True

            )

        ]





        history = model.fit(

            X_train,

            y_train,

            validation_split=0.2,

            epochs=EPOCHS,

            batch_size=BATCH_SIZE,

            callbacks=callbacks

        )




        logging.info(

            "GRU training completed"

        )


        return history



    except Exception as e:


        logging.error(

            f"Training error: {e}"

        )


        raise e







# ============================================================
# SAVE MODEL
# ============================================================


def save_model(model):


    try:


        path = os.path.join(

            MODEL_FOLDER,

            "gru_model.h5"

        )



        model.save(

            path

        )



        logging.info(

            "GRU model saved"

        )



    except Exception as e:


        logging.error(

            f"Model saving error: {e}"

        )









# ============================================================
# SAVE HISTORY
# ============================================================


def save_history(history):


    try:


        with open(

            os.path.join(

                RESULT_FOLDER,

                "gru_history.pkl"

            ),

            "wb"

        ) as file:


            pickle.dump(

                history.history,

                file

            )



    except Exception as e:


        logging.error(

            f"History saving error: {e}"

        )







# ============================================================
# TRAINING GRAPH
# ============================================================


def plot_history(history):


    try:


        plt.figure(

            figsize=(8,5)

        )


        plt.plot(

            history.history["accuracy"]

        )


        plt.plot(

            history.history["val_accuracy"]

        )


        plt.title(

            "GRU Accuracy"

        )


        plt.xlabel(

            "Epoch"

        )


        plt.ylabel(

            "Accuracy"

        )


        plt.legend(

            [

                "Training",

                "Validation"

            ]

        )


        plt.savefig(

            os.path.join(

                RESULT_FOLDER,

                "gru_training_plot.png"

            )

        )


        plt.close()



    except Exception as e:


        logging.error(

            f"Plot error: {e}"

        )








# ============================================================
# MAIN
# ============================================================


def main():


    try:


        print(
            "Loading GRU data..."
        )


        (

            X_train,

            X_test,

            y_train,

            y_test

        ) = load_data()



        print(
            "Building GRU model..."
        )



        model = build_model(

            X_train.shape[1]

        )



        model.summary()



        print(
            "Training GRU model..."
        )



        history = train_model(

            model,

            X_train,

            y_train

        )



        save_model(

            model

        )



        save_history(

            history

        )



        plot_history(

            history

        )



        print(

            "\nGRU training completed!"

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