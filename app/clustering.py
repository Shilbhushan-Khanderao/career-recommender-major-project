"""
K-Means Clustering Module

This module provides clustering capabilities using both TF-IDF and SBERT
embeddings. It enables discovery of natural groupings in user interests
and career profiles.

Clustering is useful for:
- Discovering interest patterns in user data
- Grouping similar careers together
- Identifying domain clusters
- Understanding feature space structure

Author: M.Tech Project - AI Career Guidance System
Date: November 2025
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional, Union
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from collections import Counter
import logging
import scipy.sparse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def tfidf_kmeans_cluster(
    texts: List[str],
    n_clusters: int = 5,
    max_features: int = 1000,
    ngram_range: Tuple[int, int] = (1, 2),
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, KMeans, TfidfVectorizer]:
    """
    Cluster texts using TF-IDF vectorization and K-Means.
    
    Args:
        texts: List of text documents to cluster
        n_clusters: Number of clusters (k)
        max_features: Maximum TF-IDF features
        ngram_range: N-gram range for TF-IDF
        random_state: Random seed for reproducibility
    
    Returns:
        Tuple of:
        - cluster_labels: Array of cluster assignments (n_samples,)
        - tfidf_vectors: TF-IDF feature matrix (n_samples, n_features)
        - kmeans_model: Fitted KMeans instance
        - vectorizer: Fitted TfidfVectorizer instance
    
    Example:
        >>> texts = ["AI and ML", "data science", "web development"]
        >>> labels, vectors, model, vectorizer = tfidf_kmeans_cluster(texts, n_clusters=2)
        >>> print(labels)  # [0, 0, 1]
    """
    logger.info(f"Clustering {len(texts)} texts using TF-IDF K-Means (k={n_clusters})")
    
    # Vectorize texts
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        lowercase=True,
        min_df=1,
        stop_words='english'
    )
    
    tfidf_vectors = vectorizer.fit_transform(texts)
    logger.info(f"TF-IDF vectors shape: {tfidf_vectors.shape}")
    
    # Cluster
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
        max_iter=300
    )
    
    cluster_labels = kmeans.fit_predict(tfidf_vectors)
    
    # Compute silhouette score
    if len(texts) > n_clusters:
        silhouette = silhouette_score(tfidf_vectors, cluster_labels)
        logger.info(f"Silhouette score: {silhouette:.4f}")
    
    logger.info(f"Clustering complete. Labels shape: {cluster_labels.shape}")
    
    # Convert sparse matrix to dense array - required for consistent return type
    if scipy.sparse.issparse(tfidf_vectors):
        tfidf_array: np.ndarray = tfidf_vectors.toarray()  # type: ignore
    else:
        tfidf_array = np.asarray(tfidf_vectors)
    
    return cluster_labels, tfidf_array, kmeans, vectorizer  # type: ignore


def sbert_kmeans_cluster(
    texts: List[str],
    sbert_embedder,
    n_clusters: int = 5,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, KMeans]:
    """
    Cluster texts using SBERT embeddings and K-Means.
    
    Args:
        texts: List of text documents to cluster
        sbert_embedder: SBERTEmbedder instance
        n_clusters: Number of clusters (k)
        random_state: Random seed for reproducibility
    
    Returns:
        Tuple of:
        - cluster_labels: Array of cluster assignments (n_samples,)
        - sbert_vectors: SBERT embedding matrix (n_samples, 384)
        - kmeans_model: Fitted KMeans instance
    
    Example:
        >>> from app.sbert_semantic import SBERTEmbedder
        >>> embedder = SBERTEmbedder()
        >>> texts = ["AI and ML", "data science", "web development"]
        >>> labels, vectors, model = sbert_kmeans_cluster(texts, embedder, n_clusters=2)
        >>> print(labels)  # [0, 0, 1]
    """
    logger.info(f"Clustering {len(texts)} texts using SBERT K-Means (k={n_clusters})")
    
    # Get SBERT embeddings
    sbert_vectors = sbert_embedder.encode_sentences(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True
    )
    
    logger.info(f"SBERT vectors shape: {sbert_vectors.shape}")
    
    # Cluster
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
        max_iter=300
    )
    
    cluster_labels = kmeans.fit_predict(sbert_vectors)
    
    # Compute silhouette score
    if len(texts) > n_clusters:
        silhouette = silhouette_score(sbert_vectors, cluster_labels)
        logger.info(f"Silhouette score: {silhouette:.4f}")
    
    logger.info(f"Clustering complete. Labels shape: {cluster_labels.shape}")
    
    return cluster_labels, sbert_vectors, kmeans


def analyze_clusters(
    texts: List[str],
    cluster_labels: np.ndarray,
    domain_labels: Optional[List[str]] = None,
    top_n_terms: int = 10
) -> Dict[int, Dict]:
    """
    Analyze cluster characteristics and composition.
    
    Args:
        texts: Original text documents
        cluster_labels: Cluster assignments for each document
        domain_labels: Optional domain/category labels for each document
        top_n_terms: Number of top terms to extract per cluster
    
    Returns:
        Dictionary mapping cluster_id to analysis dict containing:
        - size: Number of documents in cluster
        - percentage: Percentage of total documents
        - sample_texts: Sample documents from cluster
        - top_terms: Most frequent terms (if applicable)
        - domain_distribution: Domain counts (if domain_labels provided)
    
    Example:
        >>> analysis = analyze_clusters(texts, labels, domains)
        >>> print(analysis[0]['size'])  # 150
        >>> print(analysis[0]['domain_distribution'])  # {'AI': 80, 'Data': 70}
    """
    logger.info(f"Analyzing {len(np.unique(cluster_labels))} clusters")
    
    n_samples = len(texts)
    unique_clusters = np.unique(cluster_labels)
    
    cluster_analysis = {}
    
    for cluster_id in unique_clusters:
        # Get indices for this cluster
        cluster_mask = cluster_labels == cluster_id
        cluster_texts = [texts[i] for i in range(len(texts)) if cluster_mask[i]]
        cluster_size = len(cluster_texts)
        
        analysis = {
            'size': cluster_size,
            'percentage': (cluster_size / n_samples) * 100,
            'sample_texts': cluster_texts[:5]  # First 5 samples
        }
        
        # Extract top terms from cluster texts
        if cluster_texts:
            # Simple word frequency analysis
            all_words = ' '.join(cluster_texts).lower().split()
            # Remove common words
            stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'i', 'my', 'me', 'is', 'am', 'are', 'was', 'were'}
            filtered_words = [w for w in all_words if w not in stopwords and len(w) > 2]
            word_counts = Counter(filtered_words)
            analysis['top_terms'] = [word for word, count in word_counts.most_common(top_n_terms)]
        
        # Domain distribution if provided
        if domain_labels is not None:
            cluster_domains = [domain_labels[i] for i in range(len(domain_labels)) if cluster_mask[i]]
            domain_counts = Counter(cluster_domains)
            analysis['domain_distribution'] = dict(domain_counts.most_common())
            analysis['dominant_domain'] = domain_counts.most_common(1)[0][0] if domain_counts else None
        
        cluster_analysis[int(cluster_id)] = analysis
    
    logger.info(f"Cluster analysis complete")
    return cluster_analysis


def find_optimal_k(
    vectors: np.ndarray,
    k_range: Tuple[int, int] = (2, 10),
    method: str = 'silhouette'
) -> Tuple[int, List[float]]:
    """
    Find optimal number of clusters using elbow method or silhouette analysis.
    
    Args:
        vectors: Feature vectors (n_samples, n_features)
        k_range: Range of k values to try (min_k, max_k)
        method: 'silhouette' or 'inertia'
    
    Returns:
        Tuple of:
        - optimal_k: Best number of clusters
        - scores: Scores for each k value
    
    Example:
        >>> optimal_k, scores = find_optimal_k(vectors, k_range=(2, 10))
        >>> print(f"Optimal k: {optimal_k}")
    """
    logger.info(f"Finding optimal k in range {k_range} using {method}")
    
    min_k, max_k = k_range
    k_values = range(min_k, max_k + 1)
    scores = []
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(vectors)
        
        if method == 'silhouette':
            score = silhouette_score(vectors, labels)
        elif method == 'inertia':
            score = -kmeans.inertia_  # Negative because we want to minimize
        else:
            raise ValueError(f"Unknown method: {method}")
        
        scores.append(score)
        logger.info(f"k={k}: {method}={score:.4f}")
    
    # Find best k
    optimal_idx = np.argmax(scores)
    optimal_k = k_values[optimal_idx]
    
    logger.info(f"Optimal k: {optimal_k}")
    return optimal_k, scores


def cluster_careers_by_domain(
    careers_df: pd.DataFrame,
    sbert_embedder,
    n_clusters: int = 8
) -> pd.DataFrame:
    """
    Cluster career profiles and add cluster assignments to dataframe.
    
    Args:
        careers_df: DataFrame with career profiles (must have 'description' column)
        sbert_embedder: SBERTEmbedder instance
        n_clusters: Number of clusters
    
    Returns:
        DataFrame with added 'cluster' column
    
    Example:
        >>> careers_with_clusters = cluster_careers_by_domain(careers_df, embedder, n_clusters=8)
        >>> print(careers_with_clusters[['career', 'cluster']].head())
    """
    logger.info(f"Clustering {len(careers_df)} careers into {n_clusters} groups")
    
    # Create text representations
    career_texts = [
        f"{row['career']}. {row['description']}. Skills: {row['skills']}."
        for _, row in careers_df.iterrows()
    ]
    
    # Cluster using SBERT
    labels, vectors, model = sbert_kmeans_cluster(
        career_texts,
        sbert_embedder,
        n_clusters=n_clusters
    )
    
    # Add to dataframe
    careers_df = careers_df.copy()
    careers_df['cluster'] = labels
    
    # Analyze clusters
    analysis = analyze_clusters(
        career_texts,
        labels,
        domain_labels=careers_df['domain'].tolist()
    )
    
    # Log cluster info
    for cluster_id, info in analysis.items():
        logger.info(f"Cluster {cluster_id}: {info['size']} careers ({info['percentage']:.1f}%)")
        if 'dominant_domain' in info:
            logger.info(f"  Dominant domain: {info['dominant_domain']}")
    
    return careers_df


def get_cluster_representatives(
    texts: List[str],
    vectors: np.ndarray,
    cluster_labels: np.ndarray,
    kmeans_model: KMeans,
    n_representatives: int = 5
) -> Dict[int, List[Tuple[int, str, float]]]:
    """
    Find representative texts for each cluster (closest to centroid).
    
    Args:
        texts: Original text documents
        vectors: Feature vectors
        cluster_labels: Cluster assignments
        kmeans_model: Fitted KMeans model
        n_representatives: Number of representatives per cluster
    
    Returns:
        Dictionary mapping cluster_id to list of (index, text, distance) tuples
    
    Example:
        >>> reps = get_cluster_representatives(texts, vectors, labels, model)
        >>> print(reps[0][0])  # (idx, text, distance to centroid)
    """
    logger.info("Finding cluster representatives")
    
    representatives = {}
    unique_clusters = np.unique(cluster_labels)
    
    for cluster_id in unique_clusters:
        # Get cluster centroid
        centroid = kmeans_model.cluster_centers_[cluster_id]
        
        # Get indices for this cluster
        cluster_mask = cluster_labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        cluster_vectors = vectors[cluster_mask]
        
        # Compute distances to centroid
        distances = np.linalg.norm(cluster_vectors - centroid, axis=1)
        
        # Get top N closest
        closest_indices = np.argsort(distances)[:n_representatives]
        
        reps = [
            (
                int(cluster_indices[idx]),
                texts[cluster_indices[idx]],
                float(distances[idx])
            )
            for idx in closest_indices
        ]
        
        representatives[int(cluster_id)] = reps
    
    return representatives


def demonstrate_clustering():
    """
    Demonstration function showing clustering capabilities.
    
    Run this to verify clustering installation and functionality.
    """
    print("=" * 70)
    print("K-MEANS CLUSTERING DEMONSTRATION")
    print("=" * 70)
    
    # Sample career interests
    sample_texts = [
        "I love programming and software development",
        "I enjoy coding web applications",
        "I want to build mobile apps",
        "I am passionate about helping sick people",
        "I enjoy providing medical care to patients",
        "I want to work as a nurse or doctor",
        "I love teaching children and students",
        "I enjoy creating educational content",
        "I want to be a professor or teacher",
        "I am interested in financial analysis and investment",
        "I enjoy working with stocks and trading",
        "I want to be a financial advisor"
    ]
    
    domains = [
        "software", "software", "software",
        "healthcare", "healthcare", "healthcare",
        "education", "education", "education",
        "finance", "finance", "finance"
    ]
    
    print(f"\nSample texts: {len(sample_texts)} interest statements")
    print(f"True domains: {len(set(domains))} categories")
    
    # TF-IDF clustering
    print("\n" + "-" * 70)
    print("TF-IDF K-Means Clustering (k=4)")
    print("-" * 70)
    
    labels_tfidf, vectors_tfidf, model_tfidf, vectorizer = tfidf_kmeans_cluster(
        sample_texts,
        n_clusters=4
    )
    
    analysis_tfidf = analyze_clusters(sample_texts, labels_tfidf, domains)
    
    for cluster_id, info in sorted(analysis_tfidf.items()):
        print(f"\nCluster {cluster_id}: {info['size']} items ({info['percentage']:.1f}%)")
        print(f"  Dominant domain: {info.get('dominant_domain', 'N/A')}")
        print(f"  Top terms: {', '.join(info['top_terms'][:5])}")
        print(f"  Sample: {info['sample_texts'][0][:60]}...")
    
    # SBERT clustering
    print("\n" + "-" * 70)
    print("SBERT K-Means Clustering (k=4)")
    print("-" * 70)
    
    import sys
    import os
    # Add parent directory to path
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    # Import from sbert_semantic module directly
    from sbert_semantic import SBERTEmbedder
    embedder = SBERTEmbedder()
    
    labels_sbert, vectors_sbert, model_sbert = sbert_kmeans_cluster(
        sample_texts,
        embedder,
        n_clusters=4
    )
    
    analysis_sbert = analyze_clusters(sample_texts, labels_sbert, domains)
    
    for cluster_id, info in sorted(analysis_sbert.items()):
        print(f"\nCluster {cluster_id}: {info['size']} items ({info['percentage']:.1f}%)")
        print(f"  Dominant domain: {info.get('dominant_domain', 'N/A')}")
        print(f"  Domain distribution: {info.get('domain_distribution', {})}")
        print(f"  Sample: {info['sample_texts'][0][:60]}...")
    
    print("\n" + "=" * 70)
    print("[OK] CLUSTERING DEMONSTRATION COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    # Run demonstration
    demonstrate_clustering()
