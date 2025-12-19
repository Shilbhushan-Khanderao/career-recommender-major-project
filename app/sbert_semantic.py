"""
SBERT Semantic Similarity Module

This module provides semantic embedding capabilities using Sentence-BERT (SBERT)
for computing semantic similarity between user interests and career descriptions.

SBERT offers superior semantic understanding compared to TF-IDF by:
- Capturing contextual meaning of sentences
- Understanding synonyms and paraphrases
- Providing dense vector representations

Model: all-MiniLM-L6-v2
- Dimensions: 384
- Speed: Fast inference (~10-20ms per sentence)
- Quality: Strong semantic similarity performance
- Use case: Sentence-level semantic search

Author: M.Tech Project - AI Career Guidance System
Date: November 2025
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from sentence_transformers import SentenceTransformer  # type: ignore
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SBERTEmbedder:
    """
    SBERT-based semantic embedder for computing sentence-level similarity.
    
    Uses the all-MiniLM-L6-v2 model which provides a good balance between
    speed and accuracy for semantic similarity tasks.
    
    Attributes:
        model: SentenceTransformer model instance
        model_name: Name of the pretrained model
        embedding_dim: Dimensionality of embeddings (384 for MiniLM)
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the SBERT embedder with specified model.
        
        Args:
            model_name: Name of pretrained SentenceTransformer model
                       Default: 'all-MiniLM-L6-v2' (fast, balanced)
                       Alternative: 'all-mpnet-base-v2' (slower, more accurate)
        """
        logger.info(f"Loading SBERT model: {model_name}")
        try:
            self.model = SentenceTransformer(model_name)
            self.model_name = model_name
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dim}")
        except Exception as e:
            logger.error(f"Failed to load SBERT model: {e}")
            raise
    
    def encode_sentences(
        self, 
        texts: List[str], 
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False
    ) -> np.ndarray:
        """
        Encode sentences into dense vector embeddings.
        
        Args:
            texts: List of sentences/documents to encode
            batch_size: Number of sentences to process in parallel
            normalize_embeddings: Whether to L2-normalize embeddings (recommended)
            show_progress_bar: Whether to display encoding progress
        
        Returns:
            numpy array of shape (len(texts), embedding_dim)
            Each row is a normalized embedding vector
        
        Example:
            >>> embedder = SBERTEmbedder()
            >>> texts = ["I love programming", "I enjoy coding"]
            >>> embeddings = embedder.encode_sentences(texts)
            >>> print(embeddings.shape)  # (2, 384)
        """
        if not texts:
            logger.warning("Empty text list provided to encode_sentences")
            return np.array([])
        
        logger.info(f"Encoding {len(texts)} sentences...")
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=normalize_embeddings,
                show_progress_bar=show_progress_bar,
                convert_to_numpy=True
            )
            logger.info(f"Encoding complete. Shape: {embeddings.shape}")
            return embeddings
        except Exception as e:
            logger.error(f"Error encoding sentences: {e}")
            raise
    
    def compute_similarity(
        self, 
        query: str, 
        documents: List[str],
        top_k: Optional[int] = None
    ) -> List[Tuple[int, float]]:
        """
        Compute semantic similarity between query and documents.
        
        Uses cosine similarity on normalized embeddings (equivalent to dot product).
        
        Args:
            query: Single query sentence
            documents: List of document sentences to compare against
            top_k: If provided, return only top-k most similar documents
        
        Returns:
            List of tuples (document_index, similarity_score)
            Scores range from -1 to 1 (typically 0.0 to 1.0 for normalized)
            Sorted in descending order of similarity
        
        Example:
            >>> embedder = SBERTEmbedder()
            >>> query = "I want to work with artificial intelligence"
            >>> docs = ["Machine learning engineer", "Web developer", "AI researcher"]
            >>> results = embedder.compute_similarity(query, docs, top_k=2)
            >>> print(results)  # [(2, 0.87), (0, 0.82)]
        """
        if not documents:
            logger.warning("Empty document list provided")
            return []
        
        logger.info(f"Computing similarity for query against {len(documents)} documents")
        
        # Encode query and documents
        query_embedding = self.model.encode(
            [query], 
            normalize_embeddings=True,
            convert_to_numpy=True
        )[0]
        
        doc_embeddings = self.model.encode(
            documents,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        
        # Compute cosine similarity (dot product for normalized vectors)
        similarities = np.dot(doc_embeddings, query_embedding)
        
        # Create ranked results
        results = [(idx, float(score)) for idx, score in enumerate(similarities)]
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k if specified
        if top_k is not None:
            results = results[:top_k]
        
        logger.info(f"Similarity computation complete. Top score: {results[0][1]:.4f}")
        return results
    
    def compute_pairwise_similarity(
        self, 
        texts1: List[str], 
        texts2: List[str]
    ) -> np.ndarray:
        """
        Compute pairwise similarity matrix between two sets of texts.
        
        Args:
            texts1: First list of sentences
            texts2: Second list of sentences
        
        Returns:
            Similarity matrix of shape (len(texts1), len(texts2))
            Each element [i,j] is similarity between texts1[i] and texts2[j]
        
        Example:
            >>> embedder = SBERTEmbedder()
            >>> queries = ["AI research", "web development"]
            >>> docs = ["machine learning", "frontend coding", "neural networks"]
            >>> sim_matrix = embedder.compute_pairwise_similarity(queries, docs)
            >>> print(sim_matrix.shape)  # (2, 3)
        """
        logger.info(f"Computing pairwise similarity: {len(texts1)} x {len(texts2)}")
        
        embeddings1 = self.encode_sentences(texts1)
        embeddings2 = self.encode_sentences(texts2)
        
        # Compute similarity matrix: (n1, dim) @ (dim, n2) = (n1, n2)
        similarity_matrix = np.dot(embeddings1, embeddings2.T)
        
        logger.info(f"Pairwise similarity matrix computed: {similarity_matrix.shape}")
        return similarity_matrix
    
    def batch_similarity_scores(
        self, 
        query_embedding: np.ndarray,
        document_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Compute similarity scores for precomputed embeddings.
        
        Useful when you have already computed embeddings and want to
        compare a query against them multiple times.
        
        Args:
            query_embedding: Single embedding vector (384,)
            document_embeddings: Multiple embeddings (n_docs, 384)
        
        Returns:
            Array of similarity scores (n_docs,)
        
        Example:
            >>> embedder = SBERTEmbedder()
            >>> query_emb = embedder.encode_sentences(["AI career"])[0]
            >>> doc_embs = embedder.encode_sentences(["ML engineer", "Data scientist"])
            >>> scores = embedder.batch_similarity_scores(query_emb, doc_embs)
            >>> print(scores)  # [0.85, 0.82]
        """
        if query_embedding.ndim == 1:
            # Ensure query is 2D (1, dim) for broadcasting
            query_embedding = query_embedding.reshape(1, -1)
        
        # Cosine similarity (normalized to 0-1 range using (1 + score) / 2)
        # Raw cosine similarity ranges from -1 to 1, normalize to 0-1
        raw_scores = np.dot(document_embeddings, query_embedding.T).flatten()
        scores = (raw_scores + 1.0) / 2.0  # Map [-1, 1] to [0, 1]
        
        return scores
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model metadata
        """
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dim,
            "max_sequence_length": self.model.max_seq_length,
            "pooling": "mean",  # Default for sentence-transformers
            "normalization": "L2"
        }


def demonstrate_sbert():
    """
    Demonstration function showing SBERT capabilities.
    
    Run this to verify SBERT installation and functionality.
    """
    print("=" * 70)
    print("SBERT Semantic Similarity Demonstration")
    print("=" * 70)
    
    # Initialize embedder
    embedder = SBERTEmbedder()
    
    # Model info
    info = embedder.get_model_info()
    print(f"\nModel Information:")
    print(f"  Name: {info['model_name']}")
    print(f"  Embedding Dimension: {info['embedding_dimension']}")
    print(f"  Max Sequence Length: {info['max_sequence_length']}")
    
    # Example 1: Career similarity
    print("\n" + "-" * 70)
    print("Example 1: Career Interest Matching")
    print("-" * 70)
    
    query = "I love working with artificial intelligence and neural networks"
    careers = [
        "Machine Learning Engineer",
        "Web Developer",
        "AI Research Scientist",
        "Data Scientist",
        "Network Administrator",
        "Software Engineer",
        "Deep Learning Specialist"
    ]
    
    print(f"\nQuery: '{query}'")
    print(f"\nCareers to compare:")
    for i, career in enumerate(careers, 1):
        print(f"  {i}. {career}")
    
    results = embedder.compute_similarity(query, careers, top_k=5)
    
    print(f"\nTop 5 Most Similar Careers:")
    for rank, (idx, score) in enumerate(results, 1):
        print(f"  {rank}. {careers[idx]:<30} (Score: {score:.4f})")
    
    # Example 2: Semantic understanding
    print("\n" + "-" * 70)
    print("Example 2: Semantic Understanding (Paraphrases)")
    print("-" * 70)
    
    sentences = [
        "I enjoy coding and programming",
        "I love software development",
        "I like to write computer programs",
        "I prefer working with animals",
        "I want to help sick people"
    ]
    
    query_sent = "I am passionate about writing code"
    
    print(f"\nQuery: '{query_sent}'")
    print(f"\nSentences:")
    for i, sent in enumerate(sentences, 1):
        print(f"  {i}. {sent}")
    
    results = embedder.compute_similarity(query_sent, sentences)
    
    print(f"\nSimilarity Scores (showing semantic understanding):")
    for idx, score in results:
        print(f"  {sentences[idx]:<45} → {score:.4f}")
    
    print("\n" + "=" * 70)
    print("✓ SBERT demonstration complete!")
    print("=" * 70)


if __name__ == "__main__":
    # Run demonstration
    demonstrate_sbert()
