import pandas as pd
import re
import string

# 1. Load combined dataset
df = pd.read_csv(r"C:\Users\User\OneDrive\Desktop\Fake News Detection\data\combined_news.csv")

# 2. Text cleaning function
def clean_text(text):
    text = str(text).lower()                                         # Convert to lower case
    text = re.sub(r'http\S+|www\S+', '', text)                       # Remove web URLs
    text = re.sub(r'<.*?>', '', text)                                # Remove HTML tags
    text = text.translate(str.maketrans('', '', string.punctuation)) # Remove punctuation
    text = re.sub(r'\d+', '', text)                                  # Remove numbers/digits
    text = re.sub(r'\s+', ' ', text).strip()                          # Collapse extra whitespace
    return text

# 3. Apply cleaning across all articles
df["clean_text"] = df["content"].apply(clean_text)

# 4. Save cleaned output
df.to_csv(r"C:\Users\User\OneDrive\Desktop\Fake News Detection\data\cleaned_news.csv", index=False)

print("Part 2 Success: Data Cleaned.")
print("\n--- Top 5 Cleaned Dataset Preview ---")
print(df[["clean_text", "label"]].head())