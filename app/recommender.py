"""
Hybrid recommender system for career guidance.

This module combines multiple scoring methods (similarity, keywords, personality)
to recommend careers based on user input and predicted domain.

Supports both TF-IDF (classical) and SBERT (semantic) similarity engines.
"""

from typing import List, Dict, Literal, Optional
import pandas as pd

from .data_loader import load_careers_dataset, filter_careers_by_domain
from .nlp_pipeline import preprocess_text, compute_similarity_scores, tokenize_text
from .personality import personality_scores_from_text, compute_personality_match

# Lazy import for SBERT to avoid loading unless needed
_sbert_embedder = None
_career_embedding_cache = {}


def _get_sbert_embedder():
    """Lazy load SBERT embedder."""
    global _sbert_embedder
    if _sbert_embedder is None:
        from .sbert_semantic import SBERTEmbedder
        _sbert_embedder = SBERTEmbedder()
    return _sbert_embedder


def compute_sbert_similarity_scores(user_text: str, career_texts: List[str]) -> List[float]:
    """
    Compute semantic similarity scores using SBERT.
    
    Args:
        user_text: User input text
        career_texts: List of career text descriptions (keywords, descriptions, etc.)
        
    Returns:
        List of similarity scores (0-1 range)
    """
    embedder = _get_sbert_embedder()
    # Encode user text and career texts to embeddings
    user_embedding = embedder.encode_sentences([user_text])[0]
    # Career embeddings are computed once per distinct text list and cached
    # (as described in the dissertation); only the user's text is encoded
    # on every request.
    cache_key = tuple(career_texts)
    career_embeddings = _career_embedding_cache.get(cache_key)
    if career_embeddings is None:
        career_embeddings = embedder.encode_sentences(career_texts)
        _career_embedding_cache[cache_key] = career_embeddings
    # Compute similarity scores
    scores = embedder.batch_similarity_scores(user_embedding, career_embeddings)
    return scores.tolist()


def recommend_careers_for_domain(
    user_text: str,
    domain: str,
    top_k: int = 5,
    similarity_model: Literal["tfidf", "sbert", "hybrid"] = "hybrid"
) -> List[Dict]:
    """
    Recommend careers for a given user_text and predicted domain using hybrid scoring.
    
    The hybrid score combines:
    - Similarity score: TF-IDF cosine similarity or SBERT semantic similarity
    - Keyword score: Token overlap between user text and career keywords
    - Personality score: Match between user personality traits and career requirements
    - Skills score: Match between user skills and career skill requirements
    
    Args:
        user_text: User input text describing interests, skills, and preferences
        domain: Predicted domain label
        top_k: Number of top recommendations to return
        similarity_model: Which similarity engine to use ("tfidf", "sbert", or "hybrid")
        
    Returns:
        List of dictionaries containing career recommendations with scores
    """
    # Load careers dataset
    df = load_careers_dataset()
    
    # Filter by domain
    domain_careers = filter_careers_by_domain(df, domain)
    
    if len(domain_careers) == 0:
        return []
    
    # Preprocess user text
    user_tokens = tokenize_text(user_text)
    user_personality = personality_scores_from_text(user_tokens)
    
    # Prepare career keywords for similarity computation
    career_keywords_list = domain_careers['keywords'].tolist()
    
    # Compute similarity scores based on selected model
    if similarity_model == "tfidf":
        similarity_scores = compute_similarity_scores(user_text, career_keywords_list)
        sbert_scores = None
    elif similarity_model == "sbert":
        similarity_scores = compute_sbert_similarity_scores(user_text, career_keywords_list)
        sbert_scores = None
    else:  # hybrid
        similarity_scores = compute_similarity_scores(user_text, career_keywords_list)
        sbert_scores = compute_sbert_similarity_scores(user_text, career_keywords_list)
    
    # Compute individual scores for each career
    recommendations = []
    
    for idx, (_, career_row) in enumerate(domain_careers.iterrows()):
        # 1. Similarity score (TF-IDF, SBERT, or hybrid)
        if similarity_model == "hybrid" and sbert_scores is not None:
            # Average TF-IDF and SBERT scores
            similarity_score = (similarity_scores[idx] + sbert_scores[idx]) / 2.0
            tfidf_score = similarity_scores[idx]
            sbert_score = sbert_scores[idx]
        else:
            similarity_score = similarity_scores[idx]
            tfidf_score = similarity_scores[idx] if similarity_model == "tfidf" else 0.0
            sbert_score = similarity_scores[idx] if similarity_model == "sbert" else 0.0
        
        # 2. Keyword overlap score (improved with fuzzy matching)
        career_keywords = career_row['keywords'].lower().split(',')
        career_keywords = [kw.strip() for kw in career_keywords]
        
        # Improve keyword matching - check for substring matches too
        keyword_overlap = 0
        for keyword in career_keywords:
            for user_token in user_tokens:
                if len(keyword) > 2:  # Skip very short keywords
                    if user_token in keyword or keyword in user_token:
                        keyword_overlap += 1
                        break
        
        # Normalize by average of user and career keywords
        avg_keywords = (len(user_tokens) + len(career_keywords)) / 2
        keyword_score = min(keyword_overlap / max(avg_keywords, 1), 1.0)
        
        # 3. Personality match score (improved with partial matching)
        personality_score = compute_personality_match(
            user_personality,
            career_row['personality']
        )
        
        # 4. Skills match score (improved with fuzzy matching)
        career_skills = career_row['skills'].lower().split(',')
        career_skills = [skill.strip() for skill in career_skills]
        
        skills_overlap = 0
        for skill in career_skills:
            for user_token in user_tokens:
                if len(skill) > 2:  # Skip very short skills
                    if user_token in skill or skill in user_token:
                        skills_overlap += 1
                        break
        
        skills_score = min(skills_overlap / max(len(career_skills), 1), 1.0)
        
        # Hybrid score formula (weighted combination)
        # Increased weights for more meaningful scores
        hybrid_score = (
            0.40 * similarity_score +
            0.25 * keyword_score +
            0.20 * personality_score +
            0.15 * skills_score
        )
        
        # Create recommendation entry
        rec = {
            'career': career_row['career'],
            'domain': career_row['domain'],
            'similarity_score': round(similarity_score, 3),
            'tfidf_score': round(tfidf_score, 3),
            'sbert_score': round(sbert_score, 3),
            'keyword_score': round(keyword_score, 3),
            'personality_score': round(personality_score, 3),
            'skills_score': round(skills_score, 3),
            'hybrid_score': round(hybrid_score, 3),
            'description': career_row['description'],
            'skills': career_row['skills'],
            'personality': career_row['personality']
        }
        
        recommendations.append(rec)
    
    # Sort by hybrid score (descending)
    recommendations.sort(key=lambda x: x['hybrid_score'], reverse=True)
    
    # Return top K
    return recommendations[:top_k]


def recommend_careers_cross_domain(
    user_text: str,
    top_k: int = 5,
    domains_to_search: Optional[List[str]] = None
) -> List[Dict]:
    """
    Recommend careers across multiple domains (not limited to predicted domain).
    
    Args:
        user_text: User input text
        top_k: Number of top recommendations to return
        domains_to_search: List of domains to search (None = all domains)
        
    Returns:
        List of career recommendations across domains
    """
    # Load all careers
    df = load_careers_dataset()
    
    # Filter domains if specified
    if domains_to_search:
        df = df[df['domain'].isin(domains_to_search)]
    
    if len(df) == 0:
        return []
    
    # Preprocess user text
    user_tokens = tokenize_text(user_text)
    user_personality = personality_scores_from_text(user_tokens)
    
    # Prepare career keywords
    career_keywords_list = df['keywords'].tolist()
    
    # Compute similarity scores
    similarity_scores = compute_similarity_scores(user_text, career_keywords_list)
    
    # Compute scores for each career
    recommendations = []
    
    for idx, (_, career_row) in enumerate(df.iterrows()):
        similarity_score = similarity_scores[idx]
        
        # Keyword overlap
        career_keywords = career_row['keywords'].lower().split(',')
        career_keywords = [kw.strip() for kw in career_keywords]
        keyword_overlap = len(set(user_tokens) & set(career_keywords))
        max_overlap = max(len(user_tokens), len(career_keywords))
        keyword_score = keyword_overlap / max_overlap if max_overlap > 0 else 0.0
        
        # Personality match
        personality_score = compute_personality_match(
            user_personality,
            career_row['personality']
        )
        
        # Skills match
        career_skills = career_row['skills'].lower().split(',')
        career_skills = [skill.strip() for skill in career_skills]
        skills_overlap = len(set(user_tokens) & set(career_skills))
        skills_score = min(skills_overlap / 5.0, 1.0)
        
        # Hybrid score (consistent with main recommend_careers function)
        hybrid_score = (
            0.40 * similarity_score +
            0.25 * keyword_score +
            0.20 * personality_score +
            0.15 * skills_score
        )
        
        rec = {
            'career': career_row['career'],
            'domain': career_row['domain'],
            'similarity_score': round(similarity_score, 3),
            'keyword_score': round(keyword_score, 3),
            'personality_score': round(personality_score, 3),
            'skills_score': round(skills_score, 3),
            'hybrid_score': round(hybrid_score, 3),
            'description': career_row['description'],
            'skills': career_row['skills'],
            'personality': career_row['personality']
        }
        
        recommendations.append(rec)
    
    # Sort and return top K
    recommendations.sort(key=lambda x: x['hybrid_score'], reverse=True)
    return recommendations[:top_k]


def explain_recommendation(recommendation: Dict) -> str:
    """
    Generate a human-readable explanation for a recommendation.
    
    Args:
        recommendation: Recommendation dictionary
        
    Returns:
        Explanation string
    """
    career = recommendation['career']
    scores = []
    
    if recommendation['similarity_score'] > 0.3:
        scores.append(f"strong content match ({recommendation['similarity_score']:.2f})")
    
    if recommendation['keyword_score'] > 0.3:
        scores.append(f"good keyword alignment ({recommendation['keyword_score']:.2f})")
    
    if recommendation['personality_score'] > 0.5:
        scores.append(f"personality fit ({recommendation['personality_score']:.2f})")
    
    if recommendation['skills_score'] > 0.3:
        scores.append(f"skills match ({recommendation['skills_score']:.2f})")
    
    if not scores:
        return f"{career} is recommended based on your domain interests."
    
    explanation = f"{career} is recommended due to " + ", ".join(scores) + "."
    return explanation
