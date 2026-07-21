import pandas as pd

print("Loading datasets...")

# 1. Corrected paths and added headers since the raw CSV files do not contain a header row
cols = ['title', 'text', 'subject', 'date']
fake_df = pd.read_csv("data/Fake.csv", header=None, names=cols)
true_df = pd.read_csv("data/True.csv", header=None, names=cols)

# 2. Add the classification labels
fake_df['label'] = 0  # 0 represents Fake News
true_df['label'] = 1  # 1 represents True News

# 3. Merge the two datasets vertically
merged_df = pd.concat([fake_df, true_df], ignore_index=True)

# 4. Shuffle the data thoroughly and reset the index
merged_df = merged_df.sample(frac=1, random_state=42).reset_index(drop=True)

# 5. Clean up unnecessary columns
# Dropping 'date' and 'subject' as they usually don't help text classification
if 'date' in merged_df.columns and 'subject' in merged_df.columns:
    merged_df = merged_df.drop(columns=['date', 'subject'])

# 6. Save the final merged dataset using the corrected path
merged_df.to_csv("data/merged_dataset.csv", index=False)

print(f"Merge successful! Total records: {len(merged_df)}")
print("File saved as 'data/merged_dataset.csv'")