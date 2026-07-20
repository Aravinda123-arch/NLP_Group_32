import pandas as pd
import ast
import nltk

from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

# -------------------------
# Load Dataset
# -------------------------

df = pd.read_csv("data/stopwords_removed.csv")

# Convert string to list

df["filtered_tokens"] = df["filtered_tokens"].apply(ast.literal_eval)

# -------------------------
# Stemming
# -------------------------

def stem_words(tokens):

    return [stemmer.stem(word) for word in tokens]

df["stemmed_tokens"] = df["filtered_tokens"].apply(stem_words)

# Convert tokens into sentence

df["processed_text"] = df["stemmed_tokens"].apply(lambda x: " ".join(x))

# Save

df.to_csv("data/stemmed_news.csv", index=False)

print("Stemming Completed Successfully")
print(df[["processed_text"]].head())

