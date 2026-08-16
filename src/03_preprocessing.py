"""
03_preprocessing.py

Fake News Detection Project
Natural Language Processing Module

Purpose:
    Perform complete NLP preprocessing pipeline:
    - Text normalization
    - Contraction expansion
    - Language detection
    - Tokenization
    - Stop word removal
    - Stemming
    - Lemmatization
    - Logging
    - Error handling
    - CSV generation

Input:
    data/cleaned_news.csv

Output:
    data/preprocessed_news.csv
"""

import os
import re
import logging
import pandas as pd
import nltk  # pyrefly: ignore [missing-import]

from tqdm import tqdm
from langdetect import detect, DetectorFactory  # pyrefly: ignore [missing-import]

from nltk.tokenize import word_tokenize  # pyrefly: ignore [missing-import]
from nltk.corpus import stopwords  # pyrefly: ignore [missing-import]
from nltk.stem import PorterStemmer, WordNetLemmatizer  # pyrefly: ignore [missing-import]


# ============================================================
# NLTK DOWNLOADS
# ============================================================

def download_nltk_resources():

    resources = [
        "punkt",
        "stopwords",
        "wordnet",
        "omw-1.4"
    ]

    for resource in resources:
        try:
            nltk.download(resource, quiet=True)

        except Exception as e:
            print(
                f"Failed downloading {resource}: {e}"
            )


download_nltk_resources()


# ============================================================
# CONFIGURATION
# ============================================================

DetectorFactory.seed = 0


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


INPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "cleaned_news.csv"
)


OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "preprocessed_news.csv"
)


LOG_FOLDER = os.path.join(
    BASE_DIR,
    "logs"
)


os.makedirs(
    LOG_FOLDER,
    exist_ok=True
)


# ============================================================
# LOGGING CONFIGURATION
# ============================================================

logging.basicConfig(
    filename=os.path.join(
        LOG_FOLDER,
        "preprocessing.log"
    ),

    level=logging.INFO,

    format=
    "%(asctime)s | %(levelname)s | %(message)s"
)


logging.info(
    "Preprocessing started"
)


# ============================================================
# NLP OBJECTS
# ============================================================


stop_words = set(
    stopwords.words("english")
)


stemmer = PorterStemmer()


lemmatizer = WordNetLemmatizer()



# ============================================================
# CONTRACTION DICTIONARY
# ============================================================


CONTRACTIONS = {

    "can't": "cannot",
    "won't": "will not",
    "n't": " not",

    "'re": " are",
    "'s": " is",
    "'d": " would",
    "'ll": " will",
    "'t": " not",
    "'ve": " have",
    "'m": " am"

}



# ============================================================
# TEXT NORMALIZATION
# ============================================================


def normalize_text(text):

    """
    Convert text into normalized format
    """

    try:

        text = str(text)


        # lowercase

        text = text.lower()



        # remove html tags

        text = re.sub(
            r"<.*?>",
            "",
            text
        )



        # remove urls

        text = re.sub(
            r"http\S+|www\S+",
            "",
            text
        )



        # expand contractions

        for contraction, expansion in CONTRACTIONS.items():

            text = text.replace(
                contraction,
                expansion
            )



        # remove special characters

        text = re.sub(
            r"[^a-zA-Z\s]",
            " ",
            text
        )



        # remove extra spaces

        text = re.sub(
            r"\s+",
            " ",
            text
        )


        return text.strip()


    except Exception as e:

        logging.error(
            f"Normalization error: {e}"
        )

        return ""




# ============================================================
# LANGUAGE DETECTION
# ============================================================


def detect_language(text):

    try:

        if len(text.strip()) < 3:

            return "unknown"


        return detect(text)


    except Exception:

        return "unknown"




# ============================================================
# TOKENIZATION
# ============================================================


def tokenize_text(text):

    try:

        tokens = word_tokenize(text)

        return tokens


    except Exception as e:

        logging.error(
            f"Tokenization error: {e}"
        )

        return []




# ============================================================
# STOP WORD REMOVAL
# ============================================================


def remove_stopwords(tokens):

    try:

        filtered_words = []


        for word in tokens:

            if word not in stop_words:

                filtered_words.append(word)


        return filtered_words


    except Exception as e:

        logging.error(
            f"Stop word removal error: {e}"
        )

        return []




# ============================================================
# STEMMING
# ============================================================


def stem_words(tokens):

    try:

        return [

            stemmer.stem(word)

            for word in tokens

        ]


    except Exception as e:

        logging.error(
            f"Stemming error: {e}"
        )

        return []




# ============================================================
# LEMMATIZATION
# ============================================================


def lemmatize_words(tokens):

    try:

        return [

            lemmatizer.lemmatize(word)

            for word in tokens

        ]


    except Exception as e:

        logging.error(
            f"Lemmatization error: {e}"
        )

        return []




# ============================================================
# COMPLETE PREPROCESSING PIPELINE
# ============================================================


def preprocess_pipeline(text):

    try:


        # Normalize

        cleaned_text = normalize_text(
            text
        )


        # Language detection

        language = detect_language(
            cleaned_text
        )



        # Tokenization

        tokens = tokenize_text(
            cleaned_text
        )



        # Stop word removal

        tokens = remove_stopwords(
            tokens
        )



        # Stemming

        stemmed_tokens = stem_words(
            tokens
        )



        # Lemmatization

        lemma_tokens = lemmatize_words(
            tokens
        )



        return pd.Series(

            [

                cleaned_text,

                language,

                tokens,

                stemmed_tokens,

                lemma_tokens

            ]

        )



    except Exception as e:

        logging.error(
            f"Pipeline error: {e}"
        )


        return pd.Series(

            [

                "",

                "unknown",

                [],

                [],

                []

            ]

        )




# ============================================================
# LOAD DATA
# ============================================================


def load_dataset():

    try:

        df = pd.read_csv(
            INPUT_FILE
        )


        logging.info(
            f"Dataset loaded successfully: {df.shape}"
        )


        return df



    except Exception as e:

        logging.error(
            f"Dataset loading failed: {e}"
        )


        raise e




# ============================================================
# MAIN FUNCTION
# ============================================================


def main():


    try:


        print(
            "Loading dataset..."
        )


        df = load_dataset()



        if "text" not in df.columns:

            raise Exception(
                "Column 'text' not found"
            )



        print(
            "Running preprocessing..."
        )



        tqdm.pandas()



        results = df["text"].progress_apply(
            preprocess_pipeline
        )



        results.columns = [

            "clean_text",

            "language",

            "tokens",

            "stemmed_tokens",

            "lemmatized_tokens"

        ]



        df = pd.concat(

            [

                df,

                results

            ],

            axis=1

        )



        # Save output


        df.to_csv(

            OUTPUT_FILE,

            index=False

        )



        logging.info(

            "Preprocessing completed successfully"

        )



        print(
            "\nPreprocessing completed!"
        )


        print(

            f"Saved file: {OUTPUT_FILE}"

        )



    except Exception as e:


        logging.error(

            f"Main execution failed: {e}"

        )


        print(
            f"Error: {e}"
        )





# ============================================================
# EXECUTION
# ============================================================


if __name__ == "__main__":

    main()