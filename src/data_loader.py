import pandas as pd
from sklearn.utils import shuffle
import os

def load_datasets(fake_path, true_path):
    """
    Load Fake and True datasets.

    Parameters
    ----------
    fake_path : str
        Path to Fake.csv

    true_path : str
        Path to True.csv

    Returns
    -------
    fake_df : DataFrame

    true_df : DataFrame
    """

    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)

    return fake_df, true_df

def inspect_dataset(df, name):

    print("=" * 60)
    print(f"{name} Dataset")
    print("=" * 60)

    print("\nShape")
    print(df.shape)

    print("\nColumns")
    print(df.columns)

    print("\nFirst Five Records")
    print(df.head())

    print("\nMissing Values")
    print(df.isnull().sum())

    print("\nDuplicate Records")
    print(df.duplicated().sum())

def add_labels(fake_df, true_df):

    fake_df["label"] = 0

    true_df["label"] = 1

    return fake_df, true_df

def merge_dataset(fake_df, true_df):

    dataset = pd.concat(
        [fake_df, true_df],
        ignore_index=True
    )

    return dataset 

def shuffle_dataset(dataset):

    dataset = shuffle(
        dataset,
        random_state=42
    )

    dataset.reset_index(
        drop=True,
        inplace=True
    )

    return dataset

def save_dataset(dataset, output_path):

    dataset.to_csv(
        output_path,
        index=False
    )

    print("\nDataset saved successfully!")    

def main():

    fake_path = "../data/Fake.csv"

    true_path = "../data/True.csv"

    output_path = "../data/merged_news.csv"

    fake_df, true_df = load_datasets(
        fake_path,
        true_path
    )

    inspect_dataset(fake_df, "Fake")

    inspect_dataset(true_df, "True")

    fake_df, true_df = add_labels(
        fake_df,
        true_df
    )

    dataset = merge_dataset(
        fake_df,
        true_df
    )

    dataset = shuffle_dataset(dataset)

    save_dataset(
        dataset,
        output_path
    )

    print("\nFinal Dataset Shape")

    print(dataset.shape)

    print("\nClass Distribution")

    print(dataset["label"].value_counts())


if __name__ == "__main__":

    main()