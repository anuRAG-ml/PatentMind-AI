
import pandas as pd
import os


# 1. Define file paths
INPUT_FILE = "data/raw/bigpatent_balanced.csv"
OUTPUT_FILE = "data/processed/bigpatent_cleaned.csv"


# 2. Load the dataset
def load_data(path):
    print("Loading dataset...")
    df = pd.read_csv(path)
    print("Original shape:", df.shape)
    return df


# 3. Clean text
def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text)
    text = " ".join(text.split())

    return text


# 4. Preprocess the dataset
def preprocess_data(df):

    print("Cleaning text...")

    df["abstract"] = df["abstract"].apply(clean_text)
    df["description"] = df["description"].apply(clean_text)

    # Remove empty records
    df = df[
        (df["abstract"] != "") &
        (df["description"] != "")
    ]

    # Remove duplicate records
    df = df.drop_duplicates()

    return df


# 5. Main execution
if __name__ == "__main__":

    df = load_data(INPUT_FILE)

    df = preprocess_data(df)

    os.makedirs("data/processed", exist_ok=True)

    df.to_csv(OUTPUT_FILE, index=False)

    print("\nPreprocessing completed!")
    print("Final shape:", df.shape)
    print("Saved to:", OUTPUT_FILE)
