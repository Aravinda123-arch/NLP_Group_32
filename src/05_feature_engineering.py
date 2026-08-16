import os
import logging
import pickle

import pandas as pd
import numpy as np


from tqdm import tqdm


from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.preprocessing import LabelEncoder


# pyrefly: ignore [missing-import]
from tensorflow.keras.preprocessing.text import Tokenizer

# pyrefly: ignore [missing-import]
from tensorflow.keras.preprocessing.sequence import pad_sequences




# ============================================================
# PATH CONFIGURATION
# ============================================================


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)



INPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "advanced_nlp_features.csv"
)



FEATURE_FOLDER = os.path.join(
    BASE_DIR,
    "data",
    "features"
)



LOG_FOLDER = os.path.join(
    BASE_DIR,
    "logs"
)



os.makedirs(
    FEATURE_FOLDER,
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
        "feature_engineering.log"
    ),

    level=logging.INFO,

    format=
    "%(asctime)s | %(levelname)s | %(message)s"

)



logging.info(
    "Feature engineering started"
)






# ============================================================
# CONFIGURATION
# ============================================================


TEST_SIZE = 0.2


RANDOM_STATE = 42



MAX_WORDS = 20000


MAX_SEQUENCE_LENGTH = 300





# ============================================================
# LOAD DATA
# ============================================================


def load_dataset():


    try:


        df = pd.read_csv(
            INPUT_FILE
        )


        logging.info(

            f"Dataset loaded {df.shape}"

        )


        return df



    except Exception as e:


        logging.error(
            f"Loading error: {e}"
        )


        raise e






# ============================================================
# PREPARE TEXT AND LABEL
# ============================================================


def prepare_data(df):


    try:


        if "clean_text" not in df.columns:


            raise Exception(
                "clean_text column missing"
            )



        if "label" not in df.columns:


            raise Exception(
                "label column missing"
            )



        X = df[
            "clean_text"
        ]



        y = df[
            "label"
        ]



        encoder = LabelEncoder()


        y = encoder.fit_transform(
            y
        )



        return X, y



    except Exception as e:


        logging.error(
            f"Data preparation error: {e}"
        )


        raise e






# ============================================================
# TRAIN TEST SPLIT
# ============================================================


def split_dataset(X, y):


    try:


        return train_test_split(

            X,

            y,

            test_size=TEST_SIZE,

            random_state=RANDOM_STATE,

            stratify=y

        )



    except Exception as e:


        logging.error(
            f"Split error: {e}"
        )


        raise e






# ============================================================
# TF-IDF FEATURE EXTRACTION
# ============================================================


def create_tfidf_features(
        X_train,
        X_test
):


    try:


        vectorizer = TfidfVectorizer(

            max_features=10000,

            ngram_range=(1,2),

            min_df=2,

            max_df=0.95

        )



        X_train_tfidf = vectorizer.fit_transform(
            X_train
        )



        X_test_tfidf = vectorizer.transform(
            X_test
        )



        with open(

            os.path.join(

                FEATURE_FOLDER,

                "tfidf_vectorizer.pkl"

            ),

            "wb"

        ) as file:


            pickle.dump(

                vectorizer,

                file

            )



        return (

            X_train_tfidf,

            X_test_tfidf

        )



    except Exception as e:


        logging.error(
            f"TF-IDF error: {e}"
        )


        raise e






# ============================================================
# SAVE MACHINE LEARNING FEATURES
# ============================================================


def save_ml_features(

        X_train,

        X_test,

        y_train,

        y_test

):


    try:


        files = {


            "X_train_tfidf.pkl":
            X_train,


            "X_test_tfidf.pkl":
            X_test,


            "y_train.pkl":
            y_train,


            "y_test.pkl":
            y_test

        }



        for filename, data in files.items():


            with open(

                os.path.join(

                    FEATURE_FOLDER,

                    filename

                ),

                "wb"

            ) as file:


                pickle.dump(

                    data,

                    file

                )



        logging.info(
            "ML features saved"
        )



    except Exception as e:


        logging.error(
            f"Saving ML feature error: {e}"
        )







# ============================================================
# GRU TOKENIZATION
# ============================================================


def create_sequences(

        X_train,

        X_test

):


    try:



        tokenizer = Tokenizer(

            num_words=MAX_WORDS,

            oov_token="<OOV>"

        )



        tokenizer.fit_on_texts(
            X_train
        )



        train_sequences = tokenizer.texts_to_sequences(

            X_train

        )



        test_sequences = tokenizer.texts_to_sequences(

            X_test

        )



        X_train_pad = pad_sequences(

            train_sequences,

            maxlen=MAX_SEQUENCE_LENGTH,

            padding="post"

        )



        X_test_pad = pad_sequences(

            test_sequences,

            maxlen=MAX_SEQUENCE_LENGTH,

            padding="post"

        )




        with open(

            os.path.join(

                FEATURE_FOLDER,

                "tokenizer.pkl"

            ),

            "wb"

        ) as file:


            pickle.dump(

                tokenizer,

                file

            )



        return (

            X_train_pad,

            X_test_pad

        )



    except Exception as e:


        logging.error(
            f"Sequence creation error: {e}"
        )


        raise e






# ============================================================
# SAVE DEEP LEARNING FEATURES
# ============================================================


def save_dl_features(

        X_train,

        X_test

):


    try:


        np.save(

            os.path.join(

                FEATURE_FOLDER,

                "X_train_sequences.npy"

            ),

            X_train

        )



        np.save(

            os.path.join(

                FEATURE_FOLDER,

                "X_test_sequences.npy"

            ),

            X_test

        )



        logging.info(
            "DL features saved"
        )



    except Exception as e:


        logging.error(
            f"DL saving error: {e}"
        )







# ============================================================
# MAIN
# ============================================================


def main():


    try:


        print(
            "Loading dataset..."
        )


        df = load_dataset()



        print(
            "Preparing data..."
        )


        X, y = prepare_data(
            df
        )



        print(
            "Splitting dataset..."
        )


        (

            X_train,

            X_test,

            y_train,

            y_test

        ) = split_dataset(

            X,

            y

        )




        print(
            "Creating TF-IDF features..."
        )



        (

            X_train_tfidf,

            X_test_tfidf

        ) = create_tfidf_features(

            X_train,

            X_test

        )




        save_ml_features(

            X_train_tfidf,

            X_test_tfidf,

            y_train,

            y_test

        )




        print(
            "Creating GRU sequences..."
        )



        (

            X_train_seq,

            X_test_seq

        ) = create_sequences(

            X_train,

            X_test

        )



        save_dl_features(

            X_train_seq,

            X_test_seq

        )




        print(
            "\nFeature engineering completed!"
        )



        logging.info(
            "Feature engineering completed successfully"
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