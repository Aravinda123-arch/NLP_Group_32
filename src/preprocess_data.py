import pandas as pd

def apply_preprocessing():
    # 1. Load the dataset
    input_path = "data/cleaned_dataset.csv"  # Using the file you cleaned in the previous step
    output_path = "data/preprocessed_dataset.csv"
    
    print("Loading dataset...")
    df = pd.read_csv(input_path)
    print(f"Original shape: {df.shape}")

    # ==========================================
    # METHOD 1: Handling Missing Values
    # ==========================================
    print("\nHandling missing values...")
    # Drop rows where 'text' or 'label' is NaN (empty)
    df = df.dropna(subset=['text', 'label'])
    # Drop rows where the text is just empty whitespace
    df = df[df['text'].astype(str).str.strip() != '']
    print(f"Shape after removing missing values: {df.shape}")

    # ==========================================
    # METHOD 2: Lowercasing
    # ==========================================
    print("Applying lowercasing...")
    # Convert all text to lowercase to standardize the vocabulary
    df['text'] = df['text'].astype(str).str.lower()

    # ==========================================
    # METHOD 3: Handling Class Imbalance
    # ==========================================
    print("Handling class imbalance...")
    class_counts = df['label'].value_counts()
    print(f"Class distribution before balancing:\n{class_counts}")
    
    # Find the size of the smaller class
    min_class_size = class_counts.min()
    
    # Downsample both classes so they have the exact same number of rows
    # random_state=42 ensures the random selection is the same every time
    df_balanced = df.groupby('label').sample(n=min_class_size, random_state=42)
    
    # Shuffle the newly balanced dataset and reset the index
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"Class distribution after balancing:\n{df_balanced['label'].value_counts()}")

    # ==========================================
    # Save the Final Dataset
    # ==========================================
    df_balanced.to_csv(output_path, index=False)
    print(f"\n Success! Final preprocessed dataset saved to: {output_path}")

if __name__ == "__main__":
    apply_preprocessing()