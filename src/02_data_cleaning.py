from pathlib import Path
import pandas as pd
import re
import unicodedata


# ==========================================================
# Project Paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

INPUT_FILE = DATA_DIR / "merged_news.csv"

OUTPUT_FILE = DATA_DIR / "cleaned_news.csv"


# ==========================================================
# Load Dataset
# ==========================================================

print("=" * 70)
print("Loading merged dataset...")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

print(f"Rows Loaded : {len(df):,}")


# ==========================================================
# Cleaning Functions
# ==========================================================

def remove_html(text):

    return re.sub(r"<.*?>", " ", text)


def remove_urls(text):

    return re.sub(r"http\S+|www\S+|https\S+", " ", text)


def remove_emails(text):

    return re.sub(r"\S+@\S+", " ", text)


def remove_mentions(text):

    return re.sub(r"@\w+", " ", text)


def remove_hashtags(text):

    return re.sub(r"#\w+", " ", text)


def remove_numbers(text):

    return re.sub(r"\d+", " ", text)


def remove_punctuation(text):

    return re.sub(r"[^\w\s]", " ", text)


def remove_non_ascii(text):

    text = unicodedata.normalize("NFKD", text)

    return text.encode("ascii", "ignore").decode("utf-8", "ignore")


def remove_extra_spaces(text):

    return re.sub(r"\s+", " ", text).strip()


# ==========================================================
# Main Cleaning Function
# ==========================================================

def clean_text(text):

    if pd.isna(text):

        return ""

    text = str(text)

    text = text.lower()

    text = remove_html(text)

    text = remove_urls(text)

    text = remove_emails(text)

    text = remove_mentions(text)

    text = remove_hashtags(text)

    text = remove_non_ascii(text)

    text = remove_numbers(text)

    text = remove_punctuation(text)

    text = remove_extra_spaces(text)

    return text


# ==========================================================
# Clean Title
# ==========================================================

print("\nCleaning titles...")

df["title"] = df["title"].apply(clean_text)


# ==========================================================
# Clean Article
# ==========================================================

print("Cleaning articles...")

df["text"] = df["text"].apply(clean_text)


# ==========================================================
# Remove Empty Records
# ==========================================================

print("Removing empty rows...")

df = df[df["title"].str.strip() != ""]

df = df[df["text"].str.strip() != ""]


# ==========================================================
# Remove Duplicate News
# ==========================================================

print("Removing duplicate records...")

df.drop_duplicates(

    subset=["title", "text"],

    inplace=True

)


# ==========================================================
# Reset Index
# ==========================================================

df.reset_index(

    drop=True,

    inplace=True

)


# ==========================================================
# Save Dataset
# ==========================================================

df.to_csv(

    OUTPUT_FILE,

    index=False,

    encoding="utf-8"

)


# ==========================================================
# Display Information
# ==========================================================

print("\nCleaning Completed Successfully")

print("-" * 60)

print(f"Final Records : {len(df):,}")

print("\nClass Distribution")

print(

    df["label"]

    .value_counts()

    .rename({

        0: "Fake",

        1: "Real"

    })

)

print("\nPreview")

print(df.head())

print("\nSaved To")

print(OUTPUT_FILE)

print("=" * 70)