import pandas as pd
import ast
import nltk

from nltk.corpus import stopwords

# Download stopwords
nltk.download("stopwords")

# ----------------------------
# Load tokenized dataset
# ----------------------------

df = pd.read_csv("data/tokenized_news.csv")

# Convert string representation back to list
df["tokens"] = df["tokens"].apply(ast.literal_eval)

stop_words = set(stopwords.words("english"))

# ----------------------------
# Remove Stop Words
# ----------------------------

def remove_stopwords(tokens):

    return [word for word in tokens if word not in stop_words]

df["filtered_tokens"] = df["tokens"].apply(remove_stopwords)

# Save

df.to_csv("data/stopwords_removed.csv", index=False)

print("Stopword Removal Completed")
print(df[["filtered_tokens"]].head())