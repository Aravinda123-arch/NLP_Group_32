import pandas as pd

# 1. Load original datasets
fake_path = r"C:\Users\User\OneDrive\Desktop\Fake News Detection\data\Fake.csv"
true_path = r"C:\Users\User\OneDrive\Desktop\Fake News Detection\data\True.csv"

fake = pd.read_csv(fake_path)
true = pd.read_csv(true_path)

# 2. Assign binary labels (Fake = 0, True = 1)
fake["label"] = 0
true["label"] = 1

# 3. Combine without dropping or adding any extra data
df = pd.concat([fake, true], axis=0)

# 4. Shuffle rows to mix fake and real articles randomly
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# 5. Merge title and text into a single content feature
df["content"] = df["title"].fillna("") + " " + df["text"].fillna("")

# 6. Save combined dataset
df.to_csv(r"C:\Users\User\OneDrive\Desktop\Fake News Detection\data\combined_news.csv", index=False)

print("Part 1 Success: Data Loaded and Combined.")
print(f"Total Original Rows Retained: {len(df)}")

