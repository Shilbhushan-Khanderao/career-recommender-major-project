"""
Training script for domain classifier.

This script trains a Logistic Regression classifier with TF-IDF features
to predict career domains from user text input.
"""

import os
import sys
from pathlib import Path
import json
from datetime import datetime

import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for file output
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import nltk

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.corpus import stopwords


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the domain dataset from CSV file.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        DataFrame with 'text' and 'label' columns
    """
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} samples from {filepath}")
    print(f"Label distribution:\n{df['label'].value_counts()}\n")
    return df


def preprocess_text(text: str) -> str:
    """
    Minimal preprocessing: lowercase the text.
    
    Args:
        text: Input text string
        
    Returns:
        Preprocessed text
    """
    return text.lower().strip()


def train_model(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Train a domain classifier using TF-IDF and Logistic Regression.
    
    Args:
        df: DataFrame with 'text' and 'label' columns
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (trained_model, fitted_vectorizer, test_accuracy)
    """
    # Preprocess text
    df['text_clean'] = df['text'].apply(preprocess_text)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['text_clean'],
        df['label'],
        test_size=test_size,
        stratify=df['label'],
        random_state=random_state
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}\n")
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=3000,
        ngram_range=(1, 2),
        stop_words=list(stopwords.words('english')),
        lowercase=True,
        min_df=2
    )
    
    # Fit vectorizer and transform training data
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print(f"TF-IDF feature matrix shape: {X_train_tfidf.shape}")
    print(f"Number of features: {len(vectorizer.get_feature_names_out())}\n")
    
    # Train Logistic Regression classifier
    print("Training Logistic Regression classifier...")
    classifier = LogisticRegression(
        max_iter=500,
        random_state=random_state,
        multi_class='multinomial',
        solver='lbfgs'
    )
    classifier.fit(X_train_tfidf, y_train)
    
    # Evaluate on test set
    y_pred = classifier.predict(X_test_tfidf)
    labels = sorted(df['label'].unique())

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n{'='*60}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"{'='*60}\n")

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=labels))

    print("\nConfusion Matrix (console preview):")
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    print(f"\n{'':20} ", end="")
    for label in labels:
        print(f"{label[:15]:>15} ", end="")
    print()
    for i, label in enumerate(labels):
        print(f"{label[:20]:20} ", end="")
        for j in range(len(labels)):
            print(f"{cm[i][j]:>15} ", end="")
        print()

    return classifier, vectorizer, accuracy, y_test, y_pred, labels


def save_evaluation_outputs(
    classifier,
    df: pd.DataFrame,
    y_test,
    y_pred,
    labels: list,
    accuracy: float,
    X_train_size: int,
    X_test_size: int,
    outputs_dir: str = "outputs/appendix_figures",
):
    """
    Persist evaluation artefacts so experimental results are reproducible.

    Saves:
      - classification_metrics.txt   — human-readable summary + full report
      - classification_report.csv    — per-class precision/recall/f1/support
      - confusion_matrix.csv         — raw counts, rows=true, cols=predicted
      - confusion_matrix_full.png    — annotated heatmap

    Args:
        classifier: Trained classifier (used for config metadata).
        df: Full dataframe (used for dataset statistics).
        y_test: True labels from the held-out test split.
        y_pred: Predicted labels.
        labels: Sorted list of class names.
        accuracy: Scalar accuracy on the test split.
        X_train_size: Number of training samples.
        X_test_size: Number of test samples.
        outputs_dir: Directory to write files into.
    """
    out = Path(outputs_dir)
    out.mkdir(parents=True, exist_ok=True)

    report_dict = classification_report(y_test, y_pred, target_names=labels, output_dict=True)
    report_str = classification_report(y_test, y_pred, target_names=labels)
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ------------------------------------------------------------------
    # 1. classification_metrics.txt
    # ------------------------------------------------------------------
    metrics_path = out / "classification_metrics.txt"
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("DOMAIN CLASSIFIER EVALUATION METRICS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Generated: {timestamp}\n\n")
        f.write("Dataset Statistics:\n")
        f.write(f"  - Total Samples: {len(df):,}\n")
        f.write(f"  - Training Samples: {X_train_size:,}\n")
        f.write(f"  - Test Samples: {X_test_size:,}\n")
        f.write(f"  - Train/Test Split: 80/20\n")
        f.write(f"  - Unique Domains: {len(labels)}\n\n")
        f.write("Model Configuration:\n")
        f.write("  - Algorithm: TF-IDF + Logistic Regression\n")
        f.write("  - TF-IDF max_features: 3000\n")
        f.write("  - TF-IDF ngram_range: (1, 2)\n")
        f.write(f"  - Logistic Regression max_iter: {classifier.max_iter}\n")
        f.write(f"  - Solver: {classifier.solver} (multinomial)\n\n")
        f.write("Performance Metrics:\n")
        f.write(f"  - Accuracy: {accuracy * 100:.2f}%\n")
        f.write(f"  - Macro Avg Precision: {report_dict['macro avg']['precision']:.4f}\n")
        f.write(f"  - Macro Avg Recall: {report_dict['macro avg']['recall']:.4f}\n")
        f.write(f"  - Macro Avg F1-Score: {report_dict['macro avg']['f1-score']:.4f}\n\n")
        f.write("=" * 60 + "\n")
        f.write("Full Classification Report:\n")
        f.write("=" * 60 + "\n\n")
        f.write(report_str + "\n")
    print(f"Saved: {metrics_path}")

    # ------------------------------------------------------------------
    # 2. classification_report.csv
    # ------------------------------------------------------------------
    report_rows = [
        {"label": label, **report_dict[label]}
        for label in labels
        if label in report_dict
    ]
    report_df = pd.DataFrame(report_rows).set_index("label")
    report_csv_path = out / "classification_report.csv"
    report_df.to_csv(report_csv_path)
    print(f"Saved: {report_csv_path}")

    # ------------------------------------------------------------------
    # 3. confusion_matrix.csv
    # ------------------------------------------------------------------
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_csv_path = out / "confusion_matrix.csv"
    cm_df.to_csv(cm_csv_path)
    print(f"Saved: {cm_csv_path}")

    # ------------------------------------------------------------------
    # 4. confusion_matrix_full.png
    # ------------------------------------------------------------------
    n = len(labels)
    fig_size = max(12, n // 3)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    sns.heatmap(
        cm_df,
        annot=(n <= 30),       # annotations only when readable
        fmt="d",
        cmap="Blues",
        linewidths=0.4,
        linecolor="lightgrey",
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix — Domain Classifier\nAccuracy: {accuracy * 100:.2f}%", fontsize=13)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=max(5, 9 - n // 10))
    plt.yticks(rotation=0, fontsize=max(5, 9 - n // 10))
    plt.tight_layout()
    cm_png_path = out / "confusion_matrix_full.png"
    plt.savefig(cm_png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {cm_png_path}")


def save_models(classifier, vectorizer, models_dir: str = "models"):
    """
    Save trained classifier and vectorizer to disk.
    
    Args:
        classifier: Trained classifier model
        vectorizer: Fitted TF-IDF vectorizer
        models_dir: Directory to save models
    """
    # Create models directory if it doesn't exist
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    
    # Save models
    classifier_path = os.path.join(models_dir, "domain_classifier.pkl")
    vectorizer_path = os.path.join(models_dir, "domain_tfidf_vectorizer.pkl")
    
    joblib.dump(classifier, classifier_path)
    joblib.dump(vectorizer, vectorizer_path)
    
    print(f"\n{'='*60}")
    print(f"Models saved successfully!")
    print(f"Classifier: {classifier_path}")
    print(f"Vectorizer: {vectorizer_path}")
    print(f"{'='*60}\n")


def main():
    """Main function to train and save the domain classifier."""
    # Set up paths
    project_root = Path(__file__).parent.parent
    data_path = project_root / "data" / "domain_dataset.csv"
    models_dir = project_root / "models"
    
    print("="*60)
    print("Domain Classifier Training Script")
    print("="*60)
    print()
    
    # Load data
    df = load_data(str(data_path))
    
    outputs_dir = project_root / "outputs" / "appendix_figures"

    # Train model
    classifier, vectorizer, accuracy, y_test, y_pred, labels = train_model(df)

    # Persist evaluation artefacts
    X_train_size = int(round(len(df) * 0.8))
    X_test_size = len(df) - X_train_size
    save_evaluation_outputs(
        classifier=classifier,
        df=df,
        y_test=y_test,
        y_pred=y_pred,
        labels=labels,
        accuracy=accuracy,
        X_train_size=X_train_size,
        X_test_size=X_test_size,
        outputs_dir=str(outputs_dir),
    )

    # Save models
    save_models(classifier, vectorizer, str(models_dir))

    print(f"Training completed! Final test accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
