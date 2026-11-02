"""
Test script for PatentMind-AI: Evaluates the trained classification pipeline
on a sample patent abstract.
"""

import os
import sys
import joblib
import numpy as np

CPC_MAPPING = {
    "A": "Human Necessities",
    "B": "Performing Operations; Transporting",
    "C": "Chemistry; Metallurgy",
    "D": "Textiles; Paper",
    "E": "Fixed Constructions",
    "F": "Mechanical Engineering; Lighting; Heating; Weapons",
    "G": "Physics",
    "H": "Electricity",
    "Y": "General / Emerging Cross-Sectional Technologies",
}

MODEL_PATH = "models/patent_classifier_pipeline.joblib"


def main():
    print(f"[INFO] Checking pipeline file at: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model pipeline not found at {MODEL_PATH}")

    print("[INFO] Loading saved pipeline...")
    try:
        pipeline = joblib.load(MODEL_PATH)
        print("[INFO] Pipeline loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to load pipeline: {e}", file=sys.stderr)
        raise

    # Realistic electrical engineering patent abstract (CPC Category H: Electricity)
    sample_abstract = (
        "A high-voltage power distribution and switching system comprising a step-down transformer, "
        "a solid-state power semiconductor converter circuit with insulated-gate bipolar transistors (IGBT), "
        "and a digital controller configured to monitor AC phase voltage, regulate reactive power flow, "
        "and interrupt electrical fault currents across transmission lines."
    )

    print("\n" + "=" * 65)
    print("                    SAMPLE PATENT TEST")
    print("=" * 65)
    print(f"Input Abstract:\n\"{sample_abstract}\"\n")

    # Predict class
    predicted_category = pipeline.predict([sample_abstract])[0]
    category_name = CPC_MAPPING.get(predicted_category, "Unknown Category")

    print("-" * 65)
    print(f"Predicted CPC Category: {predicted_category}")
    print(f"Category Description:   {category_name}")
    print("-" * 65)

    # Predict probabilities across all categories
    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba([sample_abstract])[0]
        classes = pipeline.classes_

        print("\nPrediction Probabilities across CPC Categories:")
        # Sort descending by probability
        sorted_indices = np.argsort(probs)[::-1]
        for idx in sorted_indices:
            cat = classes[idx]
            desc = CPC_MAPPING.get(cat, "Unknown")
            prob_pct = probs[idx] * 100
            print(f"  [{cat}] {desc:<48} : {prob_pct:6.2f}%")

    print("=" * 65)
    print("[SUCCESS] Test execution complete.")


if __name__ == "__main__":
    main()

