import os
import pandas as pd
import ast
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

nltk.download("wordnet", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
INPUT_FILE = os.path.join(DATA_DIR, "stopwords_removed.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "lemmatized_news.csv")

lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def lemmatize_tokens(tokens):
    if not tokens or not isinstance(tokens, list):
        return []
    pos_tags = nltk.pos_tag(tokens)
    return [lemmatizer.lemmatize(word, get_wordnet_pos(pos)) for word, pos in pos_tags]

# -------------------------
# Load Dataset
# -------------------------
print("Loading dataset for lemmatization...")
df = pd.read_csv(INPUT_FILE)

# Convert string representation back to list
df["filtered_tokens"] = df["filtered_tokens"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])

# -------------------------
# Lemmatization
# -------------------------
print("Lemmatizing tokens using WordNet with POS tags...")
df["lemmatized_tokens"] = df["filtered_tokens"].apply(lemmatize_tokens)

# Rejoin tokens into clean sentence string for vectorizers
df["processed_text"] = df["lemmatized_tokens"].apply(lambda x: " ".join(x))

# Save
df.to_csv(OUTPUT_FILE, index=False)

print("Lemmatization Completed Successfully")
print(f"Saved lemmatized dataset to: {OUTPUT_FILE}")
print(df[["processed_text"]].head())
