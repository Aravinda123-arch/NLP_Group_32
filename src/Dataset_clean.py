import pandas as pd
import re
import html
import hashlib
import unicodedata
from pathlib import Path

try:
    from ftfy import fix_text
except ImportError:
    fix_text = None


INPUT = Path("data/merged_dataset.csv")
OUTPUT = Path("data/cleaned_dataset.csv")

BAD_ENCODING = re.compile(
    r"â€™|â€˜|â€œ|â€|â€“|â€”|â€¦|Ã.|Â|�"
)


def repair_encoding(value):
    """Repair UTF-8/Windows encoding problems."""

    if pd.isna(value):
        return ""

    text = html.unescape(str(value).strip())

    # Best repair method
    if fix_text:
        text = fix_text(text)

    # Additional repair when ftfy is unavailable
    for encoding in ("latin1", "cp1252"):
        try:
            candidate = text.encode(encoding).decode("utf-8")

            if len(BAD_ENCODING.findall(candidate)) < len(
                BAD_ENCODING.findall(text)
            ):
                text = candidate
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass

    replacements = {
        "â€™": "'",
        "â€˜": "'",
        "â€œ": '"',
        "â€": '"',
        "â€“": "-",
        "â€”": "-",
        "â€¦": "...",
        "Â": "",
        "�": ""
    }

    for bad, correct in replacements.items():
        text = text.replace(bad, correct)

    return unicodedata.normalize("NFKC", text)


def clean_text(value):
    text = repair_encoding(value)

    # Remove HTML
    text = re.sub(r"<[^>]+>", " ", text)

    # Replace URLs and emails with neutral tokens
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " URL ",
        text,
        flags=re.I
    )

    text = re.sub(
        r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b",
        " EMAIL ",
        text
    )

    # Remove Reuters datelines
    # Example: WASHINGTON (Reuters) -
    text = re.sub(
        r"^\s*[A-Z][A-Z\s,./'-]{1,100}"
        r"\s+\(Reuters\)\s*[-–—:]?\s*",
        " ",
        text,
        flags=re.I
    )

    # Remove remaining Reuters markers
    text = re.sub(
        r"\(?\bReuters\b\)?",
        " ",
        text,
        flags=re.I
    )

    # Remove journalist-credit boilerplate
    text = re.sub(
        r"\b(?:reporting|writing|editing|additional reporting)"
        r"\s+by\s+[^.;]{1,150}[.;]?",
        " ",
        text,
        flags=re.I
    )

    # Remove invisible/control characters
    text = re.sub(
        r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]",
        " ",
        text
    )

    # Normalize whitespace
    return re.sub(r"\s+", " ", text).strip()


def normalized_key(text):
    """Create normalized text for duplicate detection."""

    text = unicodedata.normalize(
        "NFKC",
        str(text)
    ).casefold()

    return re.sub(r"[^a-z0-9]+", "", text)


def hash_value(text):
    return hashlib.sha1(
        text.encode("utf-8", errors="ignore")
    ).hexdigest()


# --------------------------------------------------
# Load dataset
# --------------------------------------------------
if not INPUT.exists():
    raise FileNotFoundError(f"File not found: {INPUT}")

df = pd.read_csv(
    INPUT,
    encoding="utf-8",
    encoding_errors="replace",
    on_bad_lines="skip",
    low_memory=False
)

df.columns = df.columns.str.strip().str.lower()

required = {"title", "text", "label"}
missing = required - set(df.columns)

if missing:
    raise ValueError(f"Missing columns: {missing}")

initial_count = len(df)


def remove_rows(mask):
    global df
    df = df.loc[~mask].copy()


# --------------------------------------------------
# Validate labels
# --------------------------------------------------
df["label"] = (
    df["label"]
    .astype(str)
    .str.strip()
    .str.lower()
    .replace({
        "fake": "0",
        "false": "0",
        "real": "1",
        "true": "1"
    })
)

df["label"] = pd.to_numeric(
    df["label"],
    errors="coerce"
)

remove_rows(~df["label"].isin([0, 1]))

df["label"] = df["label"].astype(int)


# --------------------------------------------------
# Clean title and article text
# --------------------------------------------------
df["title"] = df["title"].apply(clean_text)
df["text"] = df["text"].apply(clean_text)


# --------------------------------------------------
# Parse dates
# --------------------------------------------------
if "date" in df.columns:
    try:
        df["date"] = pd.to_datetime(
            df["date"],
            errors="coerce",
            format="mixed"
        )
    except (TypeError, ValueError):
        df["date"] = pd.to_datetime(
            df["date"],
            errors="coerce"
        )

    df["date"] = df["date"].dt.strftime("%Y-%m-%d")


# --------------------------------------------------
# Remove corrupted or incomplete records
# --------------------------------------------------
remaining_encoding_error = (
    df["title"].str.contains(BAD_ENCODING, na=False)
    | df["text"].str.contains(BAD_ENCODING, na=False)
)

remove_rows(remaining_encoding_error)

title_words = df["title"].str.split().str.len()
text_words = df["text"].str.split().str.len()

remove_rows((title_words < 3) | (text_words < 30))


# --------------------------------------------------
# Create model input
# --------------------------------------------------
df["model_text"] = (
    df["title"] + " " + df["text"]
).str.replace(
    r"\s+",
    " ",
    regex=True
).str.strip()


# --------------------------------------------------
# Remove duplicates and label conflicts
# --------------------------------------------------
duplicate_source = df.apply(
    lambda row: (
        row["text"]
        if len(row["text"]) >= 100
        else row["model_text"]
    ),
    axis=1
)

df["_duplicate_key"] = duplicate_source.apply(
    lambda value: hash_value(normalized_key(value))
)

label_counts = (
    df.groupby("_duplicate_key")["label"]
    .nunique()
)

conflicting_keys = label_counts[
    label_counts > 1
].index

remove_rows(df["_duplicate_key"].isin(conflicting_keys))

duplicate_mask = df.duplicated(
    subset="_duplicate_key",
    keep="first"
)

remove_rows(duplicate_mask)


# --------------------------------------------------
# Create group ID for safe train-test splitting
# --------------------------------------------------
def create_group_id(row):
    title_key = normalized_key(row["title"])

    if len(title_key) >= 15:
        source = title_key
    else:
        source = normalized_key(row["text"][:500])

    return hash_value(source)


df["group_id"] = df.apply(
    create_group_id,
    axis=1
)


# --------------------------------------------------
# Keep useful columns only
# --------------------------------------------------
columns = [
    column for column in [
        "title",
        "text",
        "date",
        "label",
        "model_text",
        "group_id"
    ]
    if column in df.columns
]

df = df[columns]

# Shuffle
df = df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# --------------------------------------------------
# Save using Excel-compatible UTF-8 encoding
# --------------------------------------------------
df.to_csv(
    OUTPUT,
    index=False,
    encoding="utf-8-sig"
)


# --------------------------------------------------
# Final validation
# --------------------------------------------------
# Final validation report
# --------------------------------------------------
bad_pattern = r"â|Ã|Â|\ufffd"
check_cols = [col for col in ["title", "text", "model_text", "processed_text", "label", "group_id"] if col in df.columns]
text_cols = [col for col in ["title", "text", "model_text", "processed_text"] if col in df.columns]

print("========== FINAL DATA CHECK ==========")
print("Initial records:", initial_count)
print("Final records:", len(df))
print("Removed records:", initial_count - len(df))

print("\nShape:")
print(df.shape)

print("\nMissing values:")
print(df[check_cols].isna().sum())

print("\nEmpty values:")
for column in text_cols:
    print(f"{column} : {df[column].astype(str).str.strip().eq('').sum()}")

print("\nEncoding problems:")
for column in text_cols:
    count = df[column].astype(str).apply(
        lambda x: bool(re.search(bad_pattern, x))
    ).sum()
    print(f"{column} : {count}")

print("\nDuplicates:")
print("Duplicate model_text:", df.duplicated("model_text").sum())
if "processed_text" in df.columns:
    print("Duplicate processed_text:", df.duplicated("processed_text").sum())

print("\nInvalid labels:")
print((~df["label"].isin([0, 1])).sum())

print("\nClass distribution:")
print(df["label"].value_counts().sort_index().rename(index={0: "Fake (0)", 1: "Real (1)"}))
print(df["label"].value_counts(normalize=True).sort_index().round(4).rename(index={0: "Fake (0)", 1: "Real (1)"}))

if "processed_text" in df.columns:
    print("\nVery short processed records:")
    print((df["processed_text"].str.split().str.len() < 10).sum())

print("\nReuters remaining:")
print(
    df["model_text"].str.contains(
        r"\breuters\b",
        case=False,
        regex=True,
        na=False
    ).sum()
)

print("\nClean dataset saved:", OUTPUT)