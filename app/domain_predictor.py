"""
Domain predictor module using trained ML classifier.

This module loads the trained domain classifier and provides
prediction functionality for user input text.
"""

from pathlib import Path
import joblib


# Load models once at module import time
_project_root = Path(__file__).parent.parent
_model_path = _project_root / "models" / "domain_classifier.pkl"
_vectorizer_path = _project_root / "models" / "domain_tfidf_vectorizer.pkl"

# Try to load models
try:
    _clf = joblib.load(_model_path)
    _tfidf = joblib.load(_vectorizer_path)
    _models_loaded = True
except FileNotFoundError:
    _clf = None
    _tfidf = None
    _models_loaded = False
    print(f"Warning: Models not found. Please train the model first using train/train_domain_classifier.py")


def predict_domain(user_text: str) -> str:
    """
    Predict the domain label for the given user text.
    
    Args:
        user_text: User input text describing interests and skills
        
    Returns:
        Predicted domain label (e.g., 'data_tech', 'education', etc.)
        
    Raises:
        RuntimeError: If models are not loaded
    """
    if not _models_loaded:
        raise RuntimeError(
            "Domain classifier models not loaded. "
            "Please train the model first by running: python train/train_domain_classifier.py"
        )
    
    # Preprocess: lowercase and clean
    text_clean = user_text.lower().strip()
    
    # Transform using TF-IDF vectorizer
    vec = _tfidf.transform([text_clean])
    
    # Predict using classifier
    pred = _clf.predict(vec)[0]
    
    return pred


def predict_domain_with_probabilities(user_text: str) -> dict:
    """
    Predict domain with confidence probabilities for all classes.
    
    Args:
        user_text: User input text
        
    Returns:
        Dictionary with 'domain' and 'probabilities' (dict of domain: probability)
        
    Raises:
        RuntimeError: If models are not loaded
    """
    if not _models_loaded:
        raise RuntimeError(
            "Domain classifier models not loaded. "
            "Please train the model first by running: python train/train_domain_classifier.py"
        )
    
    # Preprocess
    text_clean = user_text.lower().strip()
    
    # Transform
    vec = _tfidf.transform([text_clean])
    
    # Predict
    pred = _clf.predict(vec)[0]
    
    # Get probabilities
    proba = _clf.predict_proba(vec)[0]
    classes = _clf.classes_
    
    # Create probability dictionary
    prob_dict = {cls: float(prob) for cls, prob in zip(classes, proba)}
    
    return {
        'domain': pred,
        'probabilities': prob_dict,
        'confidence': float(max(proba))
    }


def is_model_loaded() -> bool:
    """
    Check if the models are successfully loaded.
    
    Returns:
        True if models are loaded, False otherwise
    """
    return _models_loaded


def get_model_info() -> dict:
    """
    Get information about the loaded models.
    
    Returns:
        Dictionary with model information
    """
    if not _models_loaded:
        return {
            'loaded': False,
            'error': 'Models not found'
        }
    
    return {
        'loaded': True,
        'classifier_type': type(_clf).__name__,
        'num_features': _tfidf.max_features,
        'classes': list(_clf.classes_)
    }
