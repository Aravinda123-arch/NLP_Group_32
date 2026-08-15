import html
import re
import unicodedata


# ============================================================
# SETTINGS
# ============================================================

MIN_WORDS = 20


# ============================================================
# BASIC NEWS CLEANING
# ============================================================

def clean_news_text(
    text: str
) -> str:
    """
    Basic text cleaning compatible with the
    trained fake-news models.

    IMPORTANT:
    - No TF-IDF fitting
    - No model training
    - No stemming
    - No stop-word removal
    """

    if text is None:
        return ""

    text = str(
        text
    )


    # --------------------------------------------------------
    # Decode HTML entities
    # --------------------------------------------------------

    text = html.unescape(
        text
    )


    # --------------------------------------------------------
    # Unicode normalization
    # --------------------------------------------------------

    text = unicodedata.normalize(
        "NFKC",
        text
    )


    # --------------------------------------------------------
    # Remove HTML tags
    # --------------------------------------------------------

    text = re.sub(
        r"<[^>]+>",
        " ",
        text,
    )


    # --------------------------------------------------------
    # Remove URLs
    # --------------------------------------------------------

    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text,
        flags=re.IGNORECASE,
    )


    # --------------------------------------------------------
    # Remove email addresses
    # --------------------------------------------------------

    text = re.sub(
        r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b",
        " ",
        text,
    )


    # --------------------------------------------------------
    # Remove Reuters/AP/AFP prefix
    # --------------------------------------------------------

    text = re.sub(
        r"^\s*(?:REUTERS|AP|AFP)\s*[-–—:]\s*",
        " ",
        text,
        flags=re.IGNORECASE,
    )


    # --------------------------------------------------------
    # Remove common reporting/editing credits
    # --------------------------------------------------------

    text = re.sub(
        r"\b(?:reporting by|additional reporting by|editing by)\b.*$",
        " ",
        text,
        flags=(
            re.IGNORECASE
            |
            re.DOTALL
        ),
    )


    # --------------------------------------------------------
    # Remove control characters
    # --------------------------------------------------------

    text = re.sub(
        r"[\x00-\x1f\x7f]",
        " ",
        text,
    )


    # --------------------------------------------------------
    # Match training representation
    # --------------------------------------------------------

    text = text.lower()


    # --------------------------------------------------------
    # Normalize whitespace
    # --------------------------------------------------------

    text = re.sub(
        r"\s+",
        " ",
        text,
    )


    return text.strip()


# ============================================================
# BUILD COMPLETE NEWS INPUT
# ============================================================

def build_news_text(
    headline: str = "",
    article: str = "",
) -> str:

    headline = (
        headline.strip()
        if headline
        else ""
    )

    article = (
        article.strip()
        if article
        else ""
    )


    combined_text = " ".join(
        part
        for part in [
            headline,
            article,
        ]
        if part
    )


    return clean_news_text(
        combined_text
    )


# ============================================================
# VALIDATE USER INPUT
# ============================================================

def validate_news_text(
    text: str
) -> int:
    """
    Validate news text and return word count.
    """

    if not text:

        raise ValueError(
            "News text cannot be empty."
        )


    word_count = len(
        text.split()
    )


    if word_count < MIN_WORDS:

        raise ValueError(
            f"News contains only "
            f"{word_count} words. "
            f"Please enter at least "
            f"{MIN_WORDS} words. "
            f"A complete article is recommended."
        )


    return word_count