"""
Training script for domain classifier.

This script trains a Logistic Regression classifier with TF-IDF features
to predict career domains from user text input.
"""

import os
import sys
from pathlib import Path

import pandas as pd
import joblib
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
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n{'='*60}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"{'='*60}\n")
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    labels = sorted(df['label'].unique())
    
    # Print confusion matrix with labels
    print(f"\n{'':20} ", end="")
    for label in labels:
        print(f"{label[:15]:>15} ", end="")
    print()
    
    for i, label in enumerate(labels):
        print(f"{label[:20]:20} ", end="")
        for j in range(len(labels)):
            print(f"{cm[i][j]:>15} ", end="")
        print()
    
    return classifier, vectorizer, accuracy


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
    
    # Train model
    classifier, vectorizer, accuracy = train_model(df)
    
    # Save models
    save_models(classifier, vectorizer, str(models_dir))
    
    print(f"Training completed! Final test accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
