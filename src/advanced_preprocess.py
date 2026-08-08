import pandas as pd
import re
import unicodedata
import hashlib

def remove_publisher_leakage(text):
    """
    Removes the 'Reuters' dateline to prevent data leakage 
    and forces models to learn actual linguistic context.
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Removes the exact pattern "(reuters) -" from the middle of the text
    text = re.sub(r'\(reuters\)\s*-\s*', ' ', text, flags=re.IGNORECASE)
    
    # 2. As an absolute safeguard, removes standalone "reuters" anywhere else
    text = re.sub(r'\breuters\b', '', text, flags=re.IGNORECASE)
    
    return text

def fix_mojibake_encoding(text):
    """
    Fixes Windows-1252 / UTF-8 Mojibake artifacts (e.g., â€™, â€œ, â€“)
    and maps them to clean ASCII characters.
    """
    if not isinstance(text, str):
        return ""

    mojibake_dict = {
        'â€™': "'", 'â€˜': "'", 'â€œ': '"', 'â€': '"',
        'â€“': '-', 'â€”': '-', 'â€¦': '...', 'Ã©': 'e', 'Â': ''
    }
    
    # Replace known Mojibake sequences
    for bad_char, good_char in mojibake_dict.items():
        text = text.replace(bad_char, good_char)

    # Normalize standard Unicode smart quotes
    text = unicodedata.normalize('NFKD', text)
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')

    # Strip remaining non-ASCII characters cleanly
    text = text.encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text

def expand_true_contractions(text):
    """
    Expands ONLY clear contractions (e.g., don't -> do not).
    Leaves possessive nouns (e.g., Kaepernick's) strictly untouched.
    """
    if not isinstance(text, str):
        return ""
        
    contractions = {
        r"\bcan't\b": "cannot", r"\bwon't\b": "will not", r"\bdon't\b": "do not",
        r"\bdoesn't\b": "does not", r"\bdidn't\b": "did not", r"\bisn't\b": "is not",
        r"\baren't\b": "are not", r"\bwasn't\b": "was not", r"\bweren't\b": "were not",
        r"\bhaven't\b": "have not", r"\bhasn't\b": "has not", r"\bhadn't\b": "had not",
        r"\bwouldn't\b": "would not", r"\bcouldn't\b": "could not", r"\bshouldn't\b": "should not",
        r"\bit's\b": "it is", r"\bhe's\b": "he is", r"\bshe's\b": "she is",
        r"\bthat's\b": "that is", r"\bwhat's\b": "what is"
    }

    for pattern, replacement in contractions.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
    return text

def main():
    input_path = "data/preprocessed_dataset.csv"
    output_path = "data/advanced_preprocessed_dataset.csv"

    print("Loading dataset...")
    df = pd.read_csv(input_path)
    print(f"Original shape: {df.shape}")

    # ==========================================
    # STEP 1: Title + Text Fusion
    # ==========================================
    print("\n1. Fusing Title + Text...")
    title_col = df['title'] if 'title' in df.columns else ""
    text_col = df['text'] if 'text' in df.columns else df['cleaned_text']
    df['fused_text'] = title_col.fillna('') + " " + text_col.fillna('')

    # ==========================================
    # STEP 2: Advanced Text Cleaning
    # ==========================================
    print("2. Scrubbing 'Reuters' data leakage to build real-world intelligence...")
    df['fused_text'] = df['fused_text'].apply(remove_publisher_leakage)

    print("3. Fixing Mojibake & Encoding Artifacts...")
    df['fused_text'] = df['fused_text'].apply(fix_mojibake_encoding)

    print("4. Expanding True Contractions...")
    df['fused_text'] = df['fused_text'].apply(expand_true_contractions)

    # ==========================================
    # STEP 3: Lowercasing & Missing Values
    # ==========================================
    print("5. Applying Lowercasing & Cleaning Whitespace...")
    df['fused_text'] = df['fused_text'].str.lower()
    df['fused_text'] = df['fused_text'].str.replace(r'\s+', ' ', regex=True).str.strip()
    
    print("6. Dropping Missing Values...")
    df = df.dropna(subset=['fused_text', 'label'])
    df = df[df['fused_text'].astype(str).str.len() > 0]

    if 'group_id' not in df.columns:
        df['group_id'] = df['fused_text'].apply(
            lambda text: hashlib.sha1(str(text)[:500].encode('utf-8')).hexdigest()[:16]
        )

    # ==========================================
    # STEP 4: Handle Class Imbalance
    # ==========================================
    print("\n7. Handling Class Imbalance...")
    class_counts = df['label'].value_counts()
    
    min_class_size = class_counts.min()
    
    # Downsample and shuffle
    df_balanced = df.groupby('label').sample(n=min_class_size, random_state=42)
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Class distribution AFTER balancing:\n{df_balanced['label'].value_counts()}")

    # ==========================================
    # STEP 5: Save Final Master Dataset
    # ==========================================
    final_cols = ['fused_text', 'label', 'group_id']
    df_balanced[final_cols].to_csv(output_path, index=False)
    
    print(f"\n SUCCESS! Master dataset (100% Leak-Proof) saved to: {output_path}")

if __name__ == "__main__":
    main()