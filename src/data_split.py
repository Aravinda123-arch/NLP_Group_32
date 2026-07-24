import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Load your latest preprocessed dataset from the data folder
input_filename = 'data/advanced_preprocessed_dataset.csv'
print(f"Loading data from {input_filename}...")
df = pd.read_csv(input_filename)

# 2. Split the dataset (80% for training/before feature engineering, 20% for testing)
# We use random_state=42 so that the random shuffle is exactly the same every time you run it.
train_df, test_df = train_test_split(df, test_size=0.20, random_state=42)

# 3. Define your new file names with the 'data/' folder path included
train_filename = 'data/before_Feature_engineering_dataset.csv'
test_filename = 'data/test_dataset.csv'

# 4. Save the split datasets to new CSV files in the data folder
# index=False prevents pandas from writing the row numbers into the CSV
print("Saving split datasets to the data folder...")
train_df.to_csv(train_filename, index=False)
test_df.to_csv(test_filename, index=False)

# 5. Output a success message with row counts to verify
print("-" * 30)
print("Split Successful!")
print(f"Total original rows: {len(df)}")
print(f"80% Training Data saved as: '{train_filename}' (Rows: {len(train_df)})")
print(f"20% Test Data saved as: '{test_filename}' (Rows: {len(test_df)})")