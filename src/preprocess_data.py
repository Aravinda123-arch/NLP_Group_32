import pandas as pd

import nltk

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer



# 1. Ensure NLTK resources are available locally

nltk_resources = [

    ('stopwords', 'corpora/stopwords'),

    ('punkt', 'tokenizers/punkt'),

    ('punkt_tab', 'tokenizers/punkt_tab'),

    ('wordnet', 'corpora/wordnet'),

    ('omw-1.4', 'corpora/omw-1.4')

]



for resource, path in nltk_resources:

    try:

        nltk.data.find(path)

    except LookupError:

        nltk.download(resource, quiet=True)



# 2. Load cleaned data from Part 2

df = pd.read_csv(r"C:\Users\User\OneDrive\Desktop\Fake News Detection\data\cleaned_news.csv")



# 3. Initialize NLP Preprocessing Tools

stop_words = set(stopwords.words("english"))

lemmatizer = WordNetLemmatizer()



def preprocess_text(text):

    # Tokenization

    tokens = word_tokenize(str(text))

    # Stop Word Removal & Lemmatization

    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 1]

    return " ".join(tokens)



print("Applying Tokenization, Stop Word Removal, and Lemmatization...")

df["processed_text"] = df["clean_text"].fillna("").apply(preprocess_text)



# 4. Save preprocessed dataset

df.to_csv(r"C:\Users\User\OneDrive\Desktop\Fake News Detection\data\preprocessed_news.csv", index=False)



# 5. Extract TF-IDF Features (using Unigrams and Bigrams for maximum accuracy)

print("Extracting TF-IDF Features...")

tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))

X_tfidf = tfidf.fit_transform(df["processed_text"])



print("\nPart 3 Success: Data Preprocessed and Features Extracted.")

print(f"TF-IDF Matrix Shape: {X_tfidf.shape}")

print("\n--- Top 5 Preprocessed Dataset Preview ---")

print(df[["processed_text", "label"]].head())