
"""
PatentMind-AI: Patent Category Classification
Model: TF-IDF + Logistic Regression
Dataset: BIGPATENT balanced 9,000-record subset
Experiment tracking: MLflow 3.x + SQLite

Features:
- Reproducible stratified train/test split
- Dataset SHA256 fingerprint
- MLflow parameter and metric tracking
- Per-class evaluation metrics
- Confusion matrix
- MLflow model logging
- Local Joblib model artifacts
- JSON and text evaluation reports
"""

import os
import json
import hashlib
import logging
from datetime import datetime, timezone

import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_PATH = "data/raw/bigpatent_balanced.csv"

MODEL_DIR = "models"
EVAL_DIR = "evaluation"

TEXT_COLUMN = "abstract"
LABEL_COLUMN = "category"

RANDOM_SEED = 42
TEST_SIZE = 0.20

EXPERIMENT_NAME = "PatentMind-Classification"
TRACKING_URI = "sqlite:///mlflow.db"

TFIDF_CONFIG = {
    "max_features": 15000,
    "ngram_range": (1, 2),
    "stop_words": "english",
    "sublinear_tf": True,
}

CLASSIFIER_CONFIG = {
    "max_iter": 1000,
    "random_state": RANDOM_SEED,
    "solver": "lbfgs",
}


# ============================================================
# 2. LOGGING CONFIGURATION
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("PatentMind-AI")


# ============================================================
# 3. DATASET FINGERPRINT
# ============================================================

def calculate_file_hash(file_path):
    """Calculate SHA256 hash of the dataset."""

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024), b""
        ):
            sha256.update(chunk)

    return sha256.hexdigest()


# ============================================================
# 4. LOAD AND CLEAN DATASET
# ============================================================

def load_dataset():

    logger.info("Loading dataset: %s", DATA_PATH)

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}. "
            "Run dvc pull if the dataset is managed by DVC."
        )

    dataset_hash = calculate_file_hash(DATA_PATH)

    df = pd.read_csv(DATA_PATH)

    required_columns = [
        TEXT_COLUMN,
        LABEL_COLUMN,
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    raw_rows = len(df)

    df = df.dropna(
        subset=required_columns
    ).copy()

    df[TEXT_COLUMN] = (
        df[TEXT_COLUMN].astype(str).str.strip()
    )

    df[LABEL_COLUMN] = (
        df[LABEL_COLUMN].astype(str).str.strip()
    )

    df = df[
        (df[TEXT_COLUMN] != "") &
        (df[LABEL_COLUMN] != "")
    ].copy()

    if df.empty:
        raise ValueError("No usable records in dataset.")

    if df[LABEL_COLUMN].nunique() < 2:
        raise ValueError(
            "Classification requires at least two categories."
        )

    logger.info("Raw records: %d", raw_rows)
    logger.info("Usable records: %d", len(df))
    logger.info(
        "Removed records: %d",
        raw_rows - len(df)
    )
    logger.info(
        "Categories: %d",
        df[LABEL_COLUMN].nunique()
    )

    return df, raw_rows, dataset_hash


# ============================================================
# 5. SAVE EVALUATION REPORTS
# ============================================================

def save_evaluation_reports(
    metrics,
    report_dict,
    report_text,
    confusion_df,
    labels,
    raw_rows,
    used_rows,
    train_rows,
    test_rows,
    dataset_hash,
    run_id,
):

    os.makedirs(EVAL_DIR, exist_ok=True)

    report_txt_path = os.path.join(
        EVAL_DIR,
        "classification_report.txt",
    )

    report_json_path = os.path.join(
        EVAL_DIR,
        "classification_metrics.json",
    )

    confusion_path = os.path.join(
        EVAL_DIR,
        "confusion_matrix.csv",
    )

    config_path = os.path.join(
        EVAL_DIR,
        "training_configuration.json",
    )

    # Save text report
    with open(
        report_txt_path,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "PatentMind-AI Classification Evaluation\n\n"
        )

        for name, value in metrics.items():
            file.write(
                f"{name}: {value:.6f}\n"
            )

        file.write("\nClassification Report:\n")
        file.write(report_text)

        file.write("\nConfusion Matrix:\n")
        file.write(confusion_df.to_string())

    # Save JSON report
    metrics_payload = {
        "model": "TF-IDF + LogisticRegression",
        "dataset_rows": raw_rows,
        "used_rows": used_rows,
        "train_rows": train_rows,
        "test_rows": test_rows,
        "dataset_sha256": dataset_hash,
        "metrics": metrics,
        "classification_report": report_dict,
        "confusion_matrix": confusion_df.values.tolist(),
        "classes": labels,
        "mlflow_run_id": run_id,
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    with open(
        report_json_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metrics_payload,
            file,
            indent=4,
        )

    # Save confusion matrix
    confusion_df.to_csv(confusion_path)

    # Save training configuration
    config = {
        "random_seed": RANDOM_SEED,
        "test_size": TEST_SIZE,
        "tfidf": TFIDF_CONFIG,
        "classifier": CLASSIFIER_CONFIG,
        "dataset_sha256": dataset_hash,
    }

    with open(
        config_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            config,
            file,
            indent=4,
        )

    return [
        report_txt_path,
        report_json_path,
        confusion_path,
        config_path,
    ]


# ============================================================
# 6. MAIN TRAINING FUNCTION
# ============================================================

def train_and_evaluate():

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(EVAL_DIR, exist_ok=True)

    # --------------------------------------------------------
    # Configure MLflow
    # --------------------------------------------------------

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    logger.info(
        "MLflow tracking URI: %s",
        mlflow.get_tracking_uri()
    )

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    df, raw_rows, dataset_hash = load_dataset()

    X = df[TEXT_COLUMN]
    y = df[LABEL_COLUMN]

    # --------------------------------------------------------
    # Stratified train/test split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    logger.info("Training records: %d", len(X_train))
    logger.info("Testing records: %d", len(X_test))

    # --------------------------------------------------------
    # Start MLflow run
    # --------------------------------------------------------

    with mlflow.start_run(
        run_name="TFIDF_LogisticRegression"
    ) as run:

        logger.info(
            "MLflow Run ID: %s",
            run.info.run_id
        )

        # ----------------------------------------------------
        # 6.1 Log parameters
        # ----------------------------------------------------

        params = {
            "model_type": "TF-IDF + Logistic Regression",
            "dataset": "BIGPATENT balanced subset",
            "dataset_raw_rows": raw_rows,
            "dataset_used_rows": len(df),
            "dataset_sha256": dataset_hash,
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "test_size": TEST_SIZE,
            "random_seed": RANDOM_SEED,
            "split_strategy": "stratified",
            "text_column": TEXT_COLUMN,
            "target_column": LABEL_COLUMN,
            "tfidf_max_features": 15000,
            "tfidf_ngram_range": "1,2",
            "tfidf_stop_words": "english",
            "tfidf_sublinear_tf": True,
            "classifier_max_iter": 1000,
            "classifier_solver": "lbfgs",
        }

        mlflow.log_params(params)

        # ----------------------------------------------------
        # 6.2 TF-IDF feature extraction
        # ----------------------------------------------------

        logger.info("Fitting TF-IDF vectorizer...")

        vectorizer = TfidfVectorizer(
            **TFIDF_CONFIG
        )

        X_train_tfidf = vectorizer.fit_transform(
            X_train
        )

        X_test_tfidf = vectorizer.transform(
            X_test
        )

        vocabulary_size = len(
            vectorizer.vocabulary_
        )

        logger.info(
            "Vocabulary size: %d",
            vocabulary_size
        )

        mlflow.log_metric(
            "vocabulary_size",
            vocabulary_size
        )

        # ----------------------------------------------------
        # 6.3 Train Logistic Regression
        # ----------------------------------------------------

        logger.info("Training Logistic Regression...")

        classifier = LogisticRegression(
            **CLASSIFIER_CONFIG
        )

        classifier.fit(
            X_train_tfidf,
            y_train
        )

        logger.info("Training completed.")

        # ----------------------------------------------------
        # 6.4 Predictions
        # ----------------------------------------------------

        y_pred = classifier.predict(
            X_test_tfidf
        )

        # ----------------------------------------------------
        # 6.5 Overall evaluation
        # ----------------------------------------------------

        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        macro_precision = precision_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0
        )

        macro_recall = recall_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0
        )

        macro_f1 = f1_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0
        )

        weighted_f1 = f1_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )

        metrics = {
            "accuracy": accuracy,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
        }

        mlflow.log_metrics(metrics)

        # ----------------------------------------------------
        # 6.6 Detailed classification report
        # ----------------------------------------------------

        labels = sorted(y.unique())

        report_dict = classification_report(
            y_test,
            y_pred,
            labels=labels,
            output_dict=True,
            zero_division=0
        )

        report_text = classification_report(
            y_test,
            y_pred,
            labels=labels,
            zero_division=0
        )

        logger.info(
            "\n%s",
            report_text
        )

        # Log per-category metrics
        for label in labels:

            class_metrics = report_dict[str(label)]

            mlflow.log_metrics({
                f"class_{label}_precision":
                    class_metrics["precision"],

                f"class_{label}_recall":
                    class_metrics["recall"],

                f"class_{label}_f1":
                    class_metrics["f1-score"],
            })

        # ----------------------------------------------------
        # 6.7 Confusion matrix
        # ----------------------------------------------------

        cm = confusion_matrix(
            y_test,
            y_pred,
            labels=labels
        )

        confusion_df = pd.DataFrame(
            cm,
            index=[
                f"True_{label}" for label in labels
            ],
            columns=[
                f"Pred_{label}" for label in labels
            ]
        )

        # ----------------------------------------------------
        # 6.8 Build complete pipeline
        # ----------------------------------------------------

        pipeline = Pipeline([
            ("tfidf", vectorizer),
            ("classifier", classifier)
        ])

        # ----------------------------------------------------
        # 6.9 Save local model files
        # ----------------------------------------------------

        vectorizer_path = os.path.join(
            MODEL_DIR,
            "tfidf_vectorizer.joblib"
        )

        classifier_path = os.path.join(
            MODEL_DIR,
            "patent_classifier.joblib"
        )

        pipeline_path = os.path.join(
            MODEL_DIR,
            "patent_classifier_pipeline.joblib"
        )

        joblib.dump(
            vectorizer,
            vectorizer_path
        )

        joblib.dump(
            classifier,
            classifier_path
        )

        joblib.dump(
            pipeline,
            pipeline_path
        )

        # ----------------------------------------------------
        # 6.10 Save evaluation reports
        # ----------------------------------------------------

        report_paths = save_evaluation_reports(
            metrics=metrics,
            report_dict=report_dict,
            report_text=report_text,
            confusion_df=confusion_df,
            labels=labels,
            raw_rows=raw_rows,
            used_rows=len(df),
            train_rows=len(X_train),
            test_rows=len(X_test),
            dataset_hash=dataset_hash,
            run_id=run.info.run_id,
        )

        # ----------------------------------------------------
        # 6.11 Log model to MLflow
        # ----------------------------------------------------

        logger.info("Logging pipeline to MLflow...")

        # No input_example is passed here.
        # This avoids the automatic signature inference warning
        # observed in the previous run. Model logging still works.

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="patent_classifier"
        )

        # Log reports and configuration
        for artifact_path in report_paths:
            mlflow.log_artifact(artifact_path)

        # ----------------------------------------------------
        # 6.12 Final output
        # ----------------------------------------------------

        print("\n" + "=" * 55)
        print(" PATENTMIND-AI CLASSIFICATION RESULTS")
        print("=" * 55)

        for name, value in metrics.items():
            print(
                f"{name:<20}: {value * 100:.2f}%"
            )

        print("\nLocal model artifacts:")
        print(vectorizer_path)
        print(classifier_path)
        print(pipeline_path)

        print("\nMLflow experiment:")
        print(EXPERIMENT_NAME)

        print("\nMLflow Run ID:")
        print(run.info.run_id)

        print("\nMLflow Tracking URI:")
        print(mlflow.get_tracking_uri())

        print(
            "\n[SUCCESS] Training, evaluation and MLflow logging completed."
        )


# ============================================================
# 7. ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:
        train_and_evaluate()

    except Exception:
        logger.exception(
            "PatentMind-AI classification training failed."
        )
        raise
