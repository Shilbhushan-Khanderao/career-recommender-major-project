"""
Personality scoring module based on Big Five traits.

This module provides simple keyword-based personality scoring
using the Big Five personality model.
"""

from typing import Dict, List


# Personality trait keywords (simplified Big Five model)
PERSONALITY_KEYWORDS = {
    "openness": [
        "creative", "curious", "imaginative", "innovative", "artistic",
        "explore", "ideas", "abstract", "experiment", "novel",
        "unconventional", "variety", "adventure", "learn", "discover",
        "design", "art", "music", "culture", "philosophy"
    ],
    "conscientiousness": [
        "organized", "disciplined", "responsible", "careful", "thorough",
        "plan", "detail", "precise", "accurate", "systematic",
        "efficient", "reliable", "structured", "methodical", "goal",
        "achievement", "diligent", "focused", "dedicated", "persistent"
    ],
    "extraversion": [
        "outgoing", "social", "energetic", "enthusiastic", "talkative",
        "people", "team", "interact", "communicate", "present",
        "leadership", "assertive", "active", "networking", "collaborative",
        "groups", "meetings", "public", "speaking", "events"
    ],
    "agreeableness": [
        "helpful", "caring", "kind", "supportive", "cooperative",
        "empathy", "compassion", "understanding", "patient", "friendly",
        "trust", "altruistic", "considerate", "generous", "warm",
        "help", "support", "assist", "service", "community"
    ],
    "neuroticism": [
        "stress", "anxiety", "worry", "pressure", "nervous",
        "emotional", "sensitive", "intense", "challenging", "demanding",
        "difficult", "complex", "uncertain", "risk", "chaos"
    ]
}


def personality_scores_from_text(tokens: List[str]) -> Dict[str, int]:
    """
    Count how many cue words for each personality trait appear in the token list.
    
    Args:
        tokens: List of lowercase tokens from user input
        
    Returns:
        Dictionary with personality trait names as keys and counts as values
    """
    scores = {trait: 0 for trait in PERSONALITY_KEYWORDS.keys()}
    
    # Convert tokens to set for faster lookup
    token_set = set(tokens)
    
    # Count matches for each trait
    for trait, keywords in PERSONALITY_KEYWORDS.items():
        for keyword in keywords:
            if keyword in token_set:
                scores[trait] += 1
    
    return scores


def get_dominant_traits(scores: Dict[str, int], top_n: int = 2) -> List[str]:
    """
    Get the top N personality traits based on scores.
    
    Args:
        scores: Dictionary of trait scores
        top_n: Number of top traits to return
        
    Returns:
        List of dominant trait names
    """
    # Sort by score (descending)
    sorted_traits = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    # Get top N with non-zero scores
    dominant = [trait for trait, score in sorted_traits[:top_n] if score > 0]
    
    return dominant


def compute_personality_match(user_scores: Dict[str, int], career_personality: str) -> float:
    """
    Compute match score between user personality and career personality requirements.
    
    Args:
        user_scores: Dictionary of user's personality trait scores
        career_personality: Comma-separated personality tags (e.g., "high_openness,moderate_conscientiousness")
        
    Returns:
        Normalized match score (0-1)
    """
    if not career_personality or career_personality == 'nan':
        return 0.5  # Neutral score if no requirements
    
    # Parse career personality tags
    tags = [tag.strip().lower() for tag in career_personality.split(',')]
    
    match_score = 0.0
    total_requirements = 0
    
    # If user has no personality traits detected, give partial credit
    user_total_score = sum(user_scores.values())
    if user_total_score == 0:
        # Generic input - return moderate score (0.4-0.5 range)
        return 0.45
    
    for tag in tags:
        # Parse tag format: "high_openness", "moderate_conscientiousness", etc.
        parts = tag.split('_', 1)
        if len(parts) != 2:
            continue
        
        level, trait = parts
        
        if trait not in user_scores:
            continue
        
        total_requirements += 1
        user_score = user_scores[trait]
        
        # Score based on level requirements - more lenient
        if level == 'high' and user_score >= 2:
            match_score += 1.0
        elif level == 'moderate' and user_score >= 1:
            match_score += 0.8
        elif level == 'low' and user_score == 0:
            match_score += 0.6
        elif user_score > 0:  # Some match is better than none
            match_score += 0.5  # Increased from 0.3
        else:  # No personality trait found in user input
            match_score += 0.4  # Give some credit for unknown traits
    
    # Normalize by number of requirements
    if total_requirements == 0:
        return 0.5
    
    return match_score / total_requirements


def personality_summary(scores: Dict[str, int]) -> str:
    """
    Generate a text summary of personality traits.
    
    Args:
        scores: Dictionary of personality trait scores
        
    Returns:
        Human-readable summary string
    """
    dominant = get_dominant_traits(scores, top_n=3)
    
    if not dominant:
        return "No clear personality traits detected from input."
    
    trait_names = {
        'openness': 'Open to Experience',
        'conscientiousness': 'Conscientious',
        'extraversion': 'Extraverted',
        'agreeableness': 'Agreeable',
        'neuroticism': 'Emotionally Sensitive'
    }
    
    readable = [trait_names.get(t, t.title()) for t in dominant]
    
    if len(readable) == 1:
        return f"Primarily {readable[0]}"
    elif len(readable) == 2:
        return f"{readable[0]} and {readable[1]}"
    else:
        return f"{', '.join(readable[:-1])}, and {readable[-1]}"
