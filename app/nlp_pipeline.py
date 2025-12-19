"""
NLP pipeline for text preprocessing and similarity computation.

This module provides functions for text cleaning and computing
similarity scores between user input and career keywords.
"""

import re
from typing import List
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


# Initialize lemmatizer globally
_lemmatizer = WordNetLemmatizer()
_stop_words = set(stopwords.words('english'))


def preprocess_text(text: str, use_lemmatization: bool = True) -> str:
    """
    Preprocess text by lowercasing, tokenizing, removing stopwords and non-alphabetic tokens.
    
    Args:
        text: Input text string
        use_lemmatization: Whether to apply lemmatization
        
    Returns:
        Cleaned text as a single string
    """
    # Lowercase
    text = text.lower()
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove non-alphabetic tokens
    tokens = [token for token in tokens if token.isalpha()]
    
    # Remove stopwords
    tokens = [token for token in tokens if token not in _stop_words]
    
    # Lemmatization (optional)
    if use_lemmatization:
        tokens = [_lemmatizer.lemmatize(token) for token in tokens]
    
    # Join back to string
    return ' '.join(tokens)


def tokenize_text(text: str) -> List[str]:
    """
    Tokenize and clean text, returning a list of tokens.
    
    Args:
        text: Input text string
        
    Returns:
        List of cleaned tokens
    """
    # Lowercase and tokenize
    tokens = word_tokenize(text.lower())
    
    # Remove non-alphabetic tokens and stopwords
    tokens = [token for token in tokens if token.isalpha() and token not in _stop_words]
    
    return tokens


def compute_similarity_scores(user_text: str, docs: List[str]) -> List[float]:
    """
    Compute cosine similarity between user text and a list of documents.
    
    Args:
        user_text: User input text
        docs: List of document strings to compare against
        
    Returns:
        List of similarity scores (0-1) for each document
    """
    if not docs:
        return []
    
    # Preprocess user text and documents
    user_clean = preprocess_text(user_text)
    docs_clean = [preprocess_text(doc) for doc in docs]
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=1000,
        ngram_range=(1, 2),
        lowercase=True,
        min_df=1
    )
    
    # Combine user text with documents for fitting
    all_texts = [user_clean] + docs_clean
    
    try:
        # Fit and transform
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        
        # User vector is the first one
        user_vector = tfidf_matrix[0:1]
        
        # Document vectors are the rest
        doc_vectors = tfidf_matrix[1:]
        
        # Compute cosine similarity
        similarities = cosine_similarity(user_vector, doc_vectors)[0]
        
        return similarities.tolist()
    
    except ValueError:
        # If vectorization fails (e.g., empty texts), return zeros
        return [0.0] * len(docs)


def extract_keywords(text: str, top_n: int = 10) -> List[str]:
    """
    Extract top keywords from text using simple frequency.
    
    Args:
        text: Input text
        top_n: Number of top keywords to return
        
    Returns:
        List of top keywords
    """
    tokens = tokenize_text(text)
    
    # Count frequency
    from collections import Counter
    freq = Counter(tokens)
    
    # Get top N most common
    top_keywords = [word for word, count in freq.most_common(top_n)]
    
    return top_keywords


def compute_keyword_overlap(tokens1: List[str], tokens2: List[str]) -> float:
    """
    Compute Jaccard similarity between two token lists.
    
    Args:
        tokens1: First list of tokens
        tokens2: Second list of tokens
        
    Returns:
        Jaccard similarity score (0-1)
    """
    set1 = set(tokens1)
    set2 = set(tokens2)
    
    if not set1 or not set2:
        return 0.0
    
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    
    return intersection / union if union > 0 else 0.0
