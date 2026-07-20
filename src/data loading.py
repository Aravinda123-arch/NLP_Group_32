import pandas as pd

print("Loading datasets... please wait.")

# 1. Use the exact paths from where your VS Code terminal is running
true_df = pd.read_csv("Fake-News-Detection/data/True.csv")
fake_df = pd.read_csv("Fake-News-Detection/data/Fake.csv")

# 2. Add labels (1 = Real, 0 = Fake)
true_df["label"] = 1
fake_df["label"] = 0

# 3. Combine and shuffle
print("Merging and shuffling data...")
merged_df = pd.concat([true_df, fake_df], ignore_index=True)
merged_df = merged_df.sample(frac=1, random_state=42).reset_index(drop=True)

# 4. Save the merged file
merged_df.to_csv("merged_fake_real_news.csv.csv", index=False)
print("Done! Merged dataset saved.")