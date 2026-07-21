import pandas as pd
import re

def clean_news_text(text):
    """
    Applies all essential NLP cleaning steps for fake news detection:
    1. Removes publisher datelines (e.g., 'WASHINGTON (Reuters) -') to eliminate data leakage.
    2. Removes URLs and web addresses.
    3. Removes leftover HTML tags.
    4. Converts text to lowercase (for uncased BERT and TF-IDF alignment).
    5. Normalizes whitespace, removing tabs and line breaks.
    """
    if not isinstance(text, str):
        return ""

    # 1. Remove Reuters datelines at the beginning of the text (Prevents Data Leakage)
    text = re.sub(r'^.*?\(Reuters\)\s*-\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^.*?\bReuters\b\s*-\s*', '', text, flags=re.IGNORECASE)

    # 2. Remove URLs and links (http, https, www)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # 3. Remove HTML tags (e.g., <br>, <a>)
    text = re.sub(r'<.*?>', '', text)

    # 4. Convert to lowercase
    text = text.lower()

    # 5. Normalize whitespace (convert newlines/tabs to single spaces and trim)
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def main():
    # Load your merged dataset
    input_path = "data/merged_dataset.csv"
    output_path = "data/cleaned_dataset.csv"

    print("Loading merged dataset...")
    df = pd.read_csv(input_path)

    print("Applying full cleaning pipeline... (This may take a minute)")
    df['cleaned_text'] = df['text'].apply(clean_news_text)

    # Filter out any blank rows resulting from empty text fields
    df = df[df['cleaned_text'].str.len() > 0].reset_index(drop=True)

    # Drop duplicate articles
    print(f"Total rows before duplicate removal: {len(df)}")
    df.drop_duplicates(subset=['cleaned_text'], keep='first', inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"Total rows after duplicate removal: {len(df)}")

    # Save the cleaned dataset
    df.to_csv(output_path, index=False)

    print(f"\n✅ Success! Cleaned dataset saved to: {output_path}")
    
    # Display sample comparison
    print("\n--- BEFORE CLEANING ---")
    print(df['text'].iloc[0][:150])
    print("\n--- AFTER CLEANING ---")
    print(df['cleaned_text'].iloc[0][:150])


if __name__ == "__main__":
    main()