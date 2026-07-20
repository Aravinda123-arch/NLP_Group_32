import pandas as pd
import re

# Load merged dataset
df = pd.read_csv(r"D:\SLTC\Third Year\First Sem\Natural Language Processing\Assignment\CA\NLP_Group_32\data\merged_news.csv")

# -----------------------------
# Combine title + text
# -----------------------------

df["content"] = df["title"] + " " + df["text"]

# -----------------------------
# Cleaning Function
# -----------------------------

def clean_text(text):

    if pd.isna(text):
        return ""

    # lowercase
    text = text.lower()

    # remove html
    text = re.sub(r'<.*?>', ' ', text)

    # remove urls
    text = re.sub(r'http\S+|www\S+', ' ', text)

    # remove emails
    text = re.sub(r'\S+@\S+', ' ', text)

    # remove numbers
    text = re.sub(r'\d+', ' ', text)

    # remove punctuation
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    # remove multiple spaces
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

# Apply cleaning

df["clean_text"] = df["content"].apply(clean_text)

# Remove empty rows

df = df[df["clean_text"] != ""]

# Save dataset

df.to_csv("data/cleaned_news.csv", index=False)

print("Cleaning Completed Successfully")

print()

print(df[["clean_text"]].head())