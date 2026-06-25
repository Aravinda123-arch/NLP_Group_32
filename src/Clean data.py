import pandas as pd
import re
import string

# Load datasets
fake = pd.read_csv(r"C:\Users\User\OneDrive\Desktop\Fake News Detection\data\Fake.csv")
true = pd.read_csv(r"C:\Users\User\OneDrive\Desktop\Fake News Detection\data\True.csv")

print(fake.head())
print(true.head())

# Add labels
fake["label"] = 0
true["label"] = 1

# Combine datasets
df = pd.concat([fake, true], axis=0)

# Reset index
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Combine title and text
df["content"] = df["title"] + " " + df["text"]

def clean_text(text):

    text = str(text)

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Remove punctuation
    text = text.translate(
        str.maketrans('', '', string.punctuation)
    )

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text

df["clean_text"] = df["content"].apply(clean_text)

print(df[["clean_text", "label"]].head())

# Save cleaned dataset
df.to_csv(r"C:\Users\User\OneDrive\Desktop\Fake News Detection\data\cleaned_news.csv", index=False)

print("---------------Dataset cleaned successfully!---------------")
print(fake.head())
print(true.head())

