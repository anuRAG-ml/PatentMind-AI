"""
PatentMind-AI: Baseline Patent Category Classifier
Trains a TF-IDF + Logistic Regression model on bigpatent_balanced.csv
to predict CPC classification categories from patent abstracts.
"""

import json
import os
import sys
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def main():
    # 1. Configuration & Paths
    data_path = "data/raw/bigpatent_balanced.csv"
    model_dir = "models"
    eval_dir = "evaluation"
    text_column = "abstract"
    label_column = "category"
    random_seed = 42
    test_size = 0.20

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    # 2. Load Dataset
    print(f"[INFO] Loading dataset from: {data_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    df = pd.read_csv(data_path)
    total_raw_rows = len(df)
    print(f"[INFO] Raw dataset loaded: {total_raw_rows} rows, {df.shape[1]} columns")

    # 3. Handle Missing Values
    initial_count = len(df)
    df = df.dropna(subset=[text_column, label_column])
    df[text_column] = df[text_column].astype(str).str.strip()
    df = df[df[text_column] != ""]
    cleaned_count = len(df)
    dropped = initial_count - cleaned_count
    print(f"[INFO] Missing/empty rows dropped: {dropped}. Usable rows: {cleaned_count}")

    X = df[text_column]
    y = df[label_column]

    # 4. Stratified Train/Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_seed,
        stratify=y,
    )
    print(f"[INFO] Training samples: {len(X_train)} (800 per category)")
    print(f"[INFO] Test samples:     {len(X_test)} (200 per category)")

    # 5. Feature Extraction (TF-IDF fitted ONLY on training data)
    print("\n[INFO] Fitting TF-IDF Vectorizer on training data...")
    vectorizer = TfidfVectorizer(
        max_features=15000,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True,
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print(f"[INFO] Vocabulary size: {len(vectorizer.vocabulary_)} features")

    # 6. Train Logistic Regression Classifier
    print("\n[INFO] Training Logistic Regression model...")
    classifier = LogisticRegression(
        max_iter=1000,
        random_state=random_seed,
        solver="lbfgs",
    )
    classifier.fit(X_train_tfidf, y_train)
    print("[INFO] Model training complete.")

    # 7. Evaluation on Held-out Test Set
    print("\n" + "=" * 50)
    print("             EVALUATION ON TEST SET")
    print("=" * 50)

    y_pred = classifier.predict(X_test_tfidf)

    acc = accuracy_score(y_test, y_pred)
    prec_macro = precision_score(y_test, y_pred, average="macro")
    rec_macro = recall_score(y_test, y_pred, average="macro")
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")

    print(f"Accuracy:          {acc * 100:.2f}%")
    print(f"Macro Precision:   {prec_macro * 100:.2f}%")
    print(f"Macro Recall:      {rec_macro * 100:.2f}%")
    print(f"Macro F1-Score:    {f1_macro * 100:.2f}%")
    print(f"Weighted F1-Score: {f1_weighted * 100:.2f}%")

    labels = sorted(y.unique())
    clf_rep_dict = classification_report(y_test, y_pred, output_dict=True)
    clf_rep_str = classification_report(y_test, y_pred)

    print("\n--- Detailed Classification Report ---")
    print(clf_rep_str)

    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_df = pd.DataFrame(
        cm,
        index=[f"True_{l}" for l in labels],
        columns=[f"Pred_{l}" for l in labels],
    )
    print("--- Confusion Matrix ---")
    print(cm_df)

    # 8. Save Model Artifacts
    print("\n" + "=" * 50)
    print("[INFO] Saving model artifacts and reports...")
    vec_path = os.path.join(model_dir, "tfidf_vectorizer.joblib")
    clf_path = os.path.join(model_dir, "patent_classifier.joblib")
    pipe_path = os.path.join(model_dir, "patent_classifier_pipeline.joblib")

    joblib.dump(vectorizer, vec_path)
    joblib.dump(classifier, clf_path)

    # Unified pipeline for inference
    pipeline = Pipeline([
        ("tfidf", vectorizer),
        ("classifier", classifier),
    ])
    joblib.dump(pipeline, pipe_path)

    # Save evaluation reports
    report_txt_path = os.path.join(eval_dir, "classification_report.txt")
    report_json_path = os.path.join(eval_dir, "classification_metrics.json")

    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write("=== PatentMind-AI: TF-IDF + Logistic Regression Evaluation ===\n\n")
        f.write(f"Accuracy:          {acc * 100:.2f}%\n")
        f.write(f"Macro Precision:   {prec_macro * 100:.2f}%\n")
        f.write(f"Macro Recall:      {rec_macro * 100:.2f}%\n")
        f.write(f"Macro F1-Score:    {f1_macro * 100:.2f}%\n")
        f.write(f"Weighted F1-Score: {f1_weighted * 100:.2f}%\n\n")
        f.write("Classification Report:\n")
        f.write(clf_rep_str)
        f.write("\nConfusion Matrix:\n")
        f.write(cm_df.to_string())

    metrics_payload = {
        "model": "TF-IDF + LogisticRegression",
        "dataset_rows": total_raw_rows,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "accuracy": acc,
        "macro_precision": prec_macro,
        "macro_recall": rec_macro,
        "macro_f1": f1_macro,
        "weighted_f1": f1_weighted,
        "classification_report": clf_rep_dict,
        "confusion_matrix": cm.tolist(),
        "classes": labels,
    }
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    print(f"[OK] Vectorizer saved to: {vec_path}")
    print(f"[OK] Classifier saved to: {clf_path}")
    print(f"[OK] Pipeline saved to:   {pipe_path}")
    print(f"[OK] Report text saved:   {report_txt_path}")
    print(f"[OK] Metrics JSON saved:  {report_json_path}")
    print("[SUCCESS] All artifacts and evaluation reports successfully generated!")


if __name__ == "__main__":
    main()

