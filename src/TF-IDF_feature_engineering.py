import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import sys

# 1. Define your exact file paths
input_filename = 'data/before_Feature_engineering_dataset.csv'
output_filename = 'data/TF-IDF_converted_dataset.csv'

print(f"Loading text data from {input_filename}...")
df = pd.read_csv(input_filename)

# --- DIAGNOSTIC TOOL ---
# This will print the actual column names in your dataset so you know what to type below.
print("\n--- AVAILABLE COLUMNS IN YOUR DATASET ---")
print(df.columns.tolist())
print("-----------------------------------------\n")

# --- IMPORTANT SETUP ---
# Look at the printout above, and change these variables to match your exact column names!
text_column = 'fused_text' # <--- Change 'text' to your actual text column name from the list above
label_column = 'label' # <--- Change 'label' to your actual True/Fake answer column name

# Safety check to prevent the KeyError from crashing the whole script
if text_column not in df.columns:
    print(f"ERROR: The column '{text_column}' was not found.")
    print("Please update the 'text_column' variable on line 20 with one of the column names printed above.")
    sys.exit()

# Clean up any accidental blank rows to prevent errors
df[text_column] = df[text_column].fillna('')

# 2. Initialize the TF-IDF Vectorizer
print(f"Translating '{text_column}' to TF-IDF math scores...")
tfidf = TfidfVectorizer(max_features=1000)

# 3. Fit and Transform the text
tfidf_matrix = tfidf.fit_transform(df[text_column])

# 4. Convert the matrix into a readable DataFrame
print("Formatting the new dataset...")
tfidf_df = pd.DataFrame(
    tfidf_matrix.toarray(), 
    columns=tfidf.get_feature_names_out()
)

# 5. Re-attach your answer labels so the model knows which row is Fake or True
if label_column in df.columns:
    tfidf_df['target_label'] = df[label_column].values
else:
    print(f"WARNING: The label column '{label_column}' was not found. Labels were not added to the output.")

# 6. Save the new dataset
print(f"Saving converted dataset to {output_filename}...")
tfidf_df.to_csv(output_filename, index=False)

print("-" * 30)
print("TF-IDF Conversion Successful!")
print(f"File saved at: {output_filename}")
print(f"Final Dataset Shape: {tfidf_df.shape[0]} rows and {tfidf_df.shape[1]} columns.")