import os
import re
import pickle
import logging


# pyrefly: ignore [missing-import]
import numpy as np
import streamlit as st



from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer



from tensorflow.keras.models import load_model


# pyrefly: ignore [missing-import]
from tensorflow.keras.preprocessing.sequence import pad_sequences







# ============================================================
# PATH CONFIGURATION
# ============================================================


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)



MODEL_FOLDER = os.path.join(

    BASE_DIR,

    "models"

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

    LOG_FOLDER,

    exist_ok=True

)







# ============================================================
# LOGGING
# ============================================================


logging.basicConfig(

    filename=os.path.join(

        LOG_FOLDER,

        "app.log"

    ),

    level=logging.INFO,

    format=

    "%(asctime)s | %(levelname)s | %(message)s"

)






# ============================================================
# LOAD NLP OBJECTS
# ============================================================


stop_words = set(

    stopwords.words(

        "english"

    )

)



lemmatizer = WordNetLemmatizer()







# ============================================================
# LOAD MODELS
# ============================================================


@st.cache_resource

def load_models():


    try:


        with open(

            os.path.join(

                MODEL_FOLDER,

                "svm_model.pkl"

            ),

            "rb"

        ) as file:


            svm_model = pickle.load(

                file

            )




        gru_model = load_model(

            os.path.join(

                MODEL_FOLDER,

                "gru_model.h5"

            )

        )




        with open(

            os.path.join(

                FEATURE_FOLDER,

                "tfidf_vectorizer.pkl"

            ),

            "rb"

        ) as file:


            vectorizer = pickle.load(

                file

            )




        with open(

            os.path.join(

                FEATURE_FOLDER,

                "tokenizer.pkl"

            ),

            "rb"

        ) as file:


            tokenizer = pickle.load(

                file

            )



        return (

            svm_model,

            gru_model,

            vectorizer,

            tokenizer

        )



    except Exception as e:


        logging.error(

            f"Model loading error: {e}"

        )


        st.error(

            "Models could not be loaded"

        )


        return None







# ============================================================
# TEXT PREPROCESSING
# ============================================================


def preprocess_text(text):


    try:


        text = text.lower()



        text = re.sub(

            r"http\S+",

            "",

            text

        )



        text = re.sub(

            r"[^a-zA-Z\s]",

            "",

            text

        )



        tokens = word_tokenize(

            text

        )



        tokens = [

            word

            for word in tokens

            if word not in stop_words

        ]



        tokens = [

            lemmatizer.lemmatize(

                word

            )

            for word in tokens

        ]



        return " ".join(

            tokens

        )



    except Exception as e:


        logging.error(

            f"Preprocessing error: {e}"

        )


        return ""









# ============================================================
# SVM PREDICTION
# ============================================================


def svm_prediction(

        model,

        vectorizer,

        text

):


    try:


        vector = vectorizer.transform(

            [text]

        )



        prediction = model.predict(

            vector

        )[0]



        confidence = model.decision_function(

            vector

        )[0]



        confidence = abs(

            confidence

        )



        return (

            prediction,

            confidence

        )



    except Exception as e:


        logging.error(

            f"SVM prediction error: {e}"

        )


        return (

            None,

            0

        )








# ============================================================
# GRU PREDICTION
# ============================================================


def gru_prediction(

        model,

        tokenizer,

        text

):


    try:


        sequence = tokenizer.texts_to_sequences(

            [text]

        )



        padded = pad_sequences(

            sequence,

            maxlen=300,

            padding="post"

        )



        probability = model.predict(

            padded

        )[0][0]



        prediction = (

            1

            if probability >= 0.5

            else 0

        )



        return (

            prediction,

            probability

        )



    except Exception as e:


        logging.error(

            f"GRU prediction error: {e}"

        )


        return (

            None,

            0

        )








# ============================================================
# STREAMLIT UI
# ============================================================


st.set_page_config(

    page_title=

    "Fake News Detection",

    page_icon=

    "📰",

    layout=

    "centered"

)





st.title(

    "📰 Fake News Detection System"

)



st.write(

    """
Enter a news article below.

The system will analyze the text using:

- SVM Machine Learning Model
- GRU Deep Learning Model

and predict whether it is Fake or Real.
"""

)






models = load_models()



if models:


    svm_model, gru_model, vectorizer, tokenizer = models



    news_text = st.text_area(

        "Enter News Content",

        height=250

    )



    if st.button(

        "Predict"

    ):



        if news_text.strip() == "":


            st.warning(

                "Please enter news text"

            )



        else:



            with st.spinner(

                "Analyzing..."

            ):



                cleaned_text = preprocess_text(

                    news_text

                )



                svm_result, svm_score = svm_prediction(

                    svm_model,

                    vectorizer,

                    cleaned_text

                )



                gru_result, gru_score = gru_prediction(

                    gru_model,

                    tokenizer,

                    cleaned_text

                )






            st.subheader(

                "Prediction Results"

            )



            col1, col2 = st.columns(2)



            with col1:


                st.write(

                    "### SVM Result"

                )



                if svm_result == 1:

                    st.success(

                        "REAL NEWS"

                    )

                else:

                    st.error(

                        "FAKE NEWS"

                    )



                st.write(

                    f"Confidence: {svm_score:.2f}"

                )





            with col2:


                st.write(

                    "### GRU Result"

                )



                if gru_result == 1:

                    st.success(

                        "REAL NEWS"

                    )

                else:

                    st.error(

                        "FAKE NEWS"

                    )



                st.write(

                    f"Confidence: {gru_score:.2f}"

                )






            st.divider()



            if svm_result == gru_result:


                if svm_result == 1:


                    st.success(

                        "Final Prediction: REAL NEWS"

                    )


                else:


                    st.error(

                        "Final Prediction: FAKE NEWS"

                    )


            else:


                st.warning(

                    "Models disagree. Manual verification recommended."

                )