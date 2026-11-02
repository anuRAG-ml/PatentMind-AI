
import argparse
import html
import re
import unicodedata
from pathlib import Path

import pandas as pd


def clean_text(text):
    """Conservatively normalize patent text without removing technical terms."""
    if pd.isna(text):
        return ""

    text = str(text)
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)

    # Normalize line endings and whitespace.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def preprocess_dataset(input_path, output_path, chunksize=250):
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Dataset not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    required_columns = {"category", "abstract", "description"}
    first_chunk = True
    total_rows = 0

    for chunk in pd.read_csv(input_path, chunksize=chunksize):
        missing = required_columns - set(chunk.columns)

        if missing:
            raise ValueError(f"Missing columns: {missing}")

        processed = pd.DataFrame({
            "category": chunk["category"].astype(str).str.strip(),
            "abstract_clean": chunk["abstract"].map(clean_text),
            "description_clean": chunk["description"].map(clean_text),
        })

        processed.to_csv(
            output_path,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False,
            encoding="utf-8"
        )

        total_rows += len(chunk)
        first_chunk = False

        print(f"Processed {total_rows} records")

    print("\nPreprocessing completed!")
    print(f"Total records: {total_rows}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BIGPATENT text preprocessing"
    )

    parser.add_argument(
        "--input",
        default="data/raw/bigpatent_balanced.csv"
    )

    parser.add_argument(
        "--output",
        default="data/processed/bigpatent_preprocessed.csv"
    )

    args = parser.parse_args()

    preprocess_dataset(args.input, args.output)
