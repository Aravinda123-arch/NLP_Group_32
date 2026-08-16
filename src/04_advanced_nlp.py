import os
import re
import logging
import pandas as pd
import nltk  # pyrefly: ignore [missing-import]
import sys


from tqdm import tqdm
from collections import Counter


import spacy  # pyrefly: ignore [missing-import]


# ============================================================
# NLTK DOWNLOADS
# ============================================================


def download_resources():

    resources = [
        "punkt",
        "averaged_perceptron_tagger",
        "maxent_ne_chunker",
        "words"
    ]


    for resource in resources:

        try:

            nltk.download(
                resource,
                quiet=True
            )

        except Exception as e:

            print(
                f"Download error {resource}: {e}"
            )



download_resources()



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
    "preprocessed_news.csv"
)



OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "advanced_nlp_features.csv"
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
# LOGGING
# ============================================================


logging.basicConfig(

    filename=os.path.join(
        LOG_FOLDER,
        "advanced_nlp.log"
    ),

    level=logging.INFO,

    format=
    "%(asctime)s | %(levelname)s | %(message)s"

)


logging.info(
    "Advanced NLP processing started"
)



# ============================================================
# LOAD SPACY MODEL
# ============================================================


try:

    nlp = spacy.load(
        "en_core_web_sm"
    )


except Exception:


    logging.warning(
        "Downloading spaCy model"
    )


    os.system(
        f'"{sys.executable}" -m spacy download en_core_web_sm'
    )


    nlp = spacy.load(
        "en_core_web_sm"
    )





# ============================================================
# TEXT STATISTICS
# ============================================================


def text_statistics(text):

    try:


        text = str(text)


        words = text.split()


        sentences = re.split(
            r"[.!?]",
            text
        )


        sentences = [

            s for s in sentences

            if s.strip()

        ]


        return pd.Series(

            [

                len(text),

                len(words),

                len(sentences),

                sum(
                    len(word)
                    for word in words
                )
                /
                max(
                    len(words),
                    1
                )

            ]

        )


    except Exception as e:


        logging.error(
            f"Statistics error: {e}"
        )


        return pd.Series(
            [
                0,
                0,
                0,
                0
            ]
        )





# ============================================================
# POS TAGGING
# ============================================================


def pos_tagging(text):


    try:


        tokens = nltk.word_tokenize(
            text
        )


        tagged_words = nltk.pos_tag(
            tokens
        )


        pos_count = Counter()


        for word, tag in tagged_words:

            pos_count[tag] += 1



        return dict(
            pos_count
        )



    except Exception as e:


        logging.error(
            f"POS tagging error: {e}"
        )


        return {}






# ============================================================
# NAMED ENTITY RECOGNITION
# ============================================================


def extract_entities(text):


    try:


        document = nlp(
            text
        )


        entities = []


        for entity in document.ents:


            entities.append(

                {

                    "text":
                    entity.text,

                    "label":
                    entity.label_

                }

            )



        return entities



    except Exception as e:


        logging.error(
            f"NER error: {e}"
        )


        return []






# ============================================================
# ENTITY COUNTS
# ============================================================


def entity_statistics(entities):


    try:


        entity_counter = Counter()



        for entity in entities:


            entity_counter[
                entity["label"]
            ] += 1



        return dict(
            entity_counter
        )



    except Exception as e:


        logging.error(
            f"Entity count error: {e}"
        )


        return {}







# ============================================================
# ADVANCED NLP PIPELINE
# ============================================================


def process_text(text):


    try:


        statistics = text_statistics(
            text
        )


        statistics.index = [

            "character_count",

            "word_count",

            "sentence_count",

            "average_word_length"

        ]



        pos_features = pos_tagging(
            text
        )


        entities = extract_entities(
            text
        )


        entity_features = entity_statistics(
            entities
        )



        return pd.Series(

            [

                statistics.to_dict(),

                pos_features,

                entities,

                entity_features

            ]

        )



    except Exception as e:


        logging.error(
            f"Advanced pipeline error: {e}"
        )


        return pd.Series(
            [
                {},
                {},
                [],
                {}
            ]
        )







# ============================================================
# LOAD DATA
# ============================================================


def load_data():


    try:


        df = pd.read_csv(
            INPUT_FILE
        )


        logging.info(

            f"Loaded dataset {df.shape}"

        )


        return df



    except Exception as e:


        logging.error(
            f"Loading failed: {e}"
        )


        raise e







# ============================================================
# MAIN FUNCTION
# ============================================================


def main():


    try:


        print(
            "Loading preprocessed dataset..."
        )


        df = load_data()



        if "clean_text" not in df.columns:


            raise Exception(

                "clean_text column missing"

            )



        print(
            "Extracting advanced NLP features..."
        )


        tqdm.pandas()



        features = df[
            "clean_text"
        ].progress_apply(

            process_text

        )



        features.columns = [

            "text_statistics",

            "pos_features",

            "named_entities",

            "entity_features"

        ]



        df = pd.concat(

            [

                df,

                features

            ],

            axis=1

        )



        df.to_csv(

            OUTPUT_FILE,

            index=False

        )



        logging.info(

            "Advanced NLP completed successfully"

        )


        print(
            "\nAdvanced NLP completed!"
        )


        print(
            f"Saved: {OUTPUT_FILE}"
        )



    except Exception as e:


        logging.error(
            f"Execution error: {e}"
        )


        print(
            f"Error: {e}"
        )






# ============================================================
# RUN
# ============================================================


if __name__ == "__main__":

    main()