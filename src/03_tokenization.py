import pandas as pd
import nltk
from nltk.tokenize import word_tokenize

# Download tokenizer (first time only)
nltk.download("punkt")

# -------------------------
# Load cleaned dataset
# -------------------------

df = pd.read_csv("data/cleaned_news.csv")

# -------------------------
# Tokenization Function
# -------------------------

def tokenize_text(text):

    if pd.isna(text):
        return []

    return word_tokenize(text)

# Apply tokenization

df["tokens"] = df["clean_text"].apply(tokenize_text)

# Save

df.to_csv("data/tokenized_news.csv", index=False)

print("Tokenization Completed Successfully")
print(df[["tokens"]].head())