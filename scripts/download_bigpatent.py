from datasets import load_dataset
import csv
import os

# Number of patents to collect from each category
SAMPLES_PER_CATEGORY = 1000

# BIGPATENT CPC categories
CATEGORIES = ["a", "b", "c", "d", "e", "f", "g", "h", "y"]

OUTPUT_FILE = "data/raw/bigpatent_balanced.csv"

os.makedirs("data/raw", exist_ok=True)

with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["category", "abstract", "description"]
    )
    writer.writeheader()

    for category in CATEGORIES:
        print(f"\nCollecting category: {category.upper()}")

        dataset = load_dataset(
            "NortheasternUniversity/big_patent",
            category,
            split="train",
            streaming=True
        )

        count = 0

        for record in dataset:
            writer.writerow({
                "category": category.upper(),
                "abstract": record["abstract"],
                "description": record["description"]
            })

            count += 1

            if count % 100 == 0:
                print(f"  Collected {count}/{SAMPLES_PER_CATEGORY}")

            if count >= SAMPLES_PER_CATEGORY:
                break

        print(f"Finished category {category.upper()}: {count} samples")

print("\n✅ Dataset collection complete!")
print(f"Saved to: {OUTPUT_FILE}")