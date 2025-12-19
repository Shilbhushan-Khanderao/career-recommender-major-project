"""
Data loader module for careers dataset.

This module provides functions to load and filter the careers dataset.
"""

from pathlib import Path
from typing import Optional
import pandas as pd


def load_careers_dataset(filepath: Optional[str] = None) -> pd.DataFrame:
    """
    Load careers_dataset.csv into a cleaned DataFrame.
    
    Args:
        filepath: Path to careers dataset CSV. If None, uses default path.
        
    Returns:
        DataFrame with career information
    """
    if filepath is None:
        # Default path relative to this file
        project_root = Path(__file__).parent.parent
        filepath = project_root / "data" / "careers_dataset.csv"
    
    df = pd.read_csv(filepath)
    
    # Clean domain values
    df['domain'] = df['domain'].str.strip().str.lower()
    
    # Ensure all text columns are strings
    text_columns = ['career', 'skills', 'personality', 'keywords', 'description']
    for col in text_columns:
        df[col] = df[col].astype(str).str.strip()
    
    return df


def filter_careers_by_domain(df: pd.DataFrame, domain: str) -> pd.DataFrame:
    """
    Return only rows where df['domain'] == domain.
    
    Args:
        df: Careers DataFrame
        domain: Domain label to filter by
        
    Returns:
        Filtered DataFrame with only matching domain
    """
    domain_clean = domain.strip().lower()
    filtered_df = df[df['domain'] == domain_clean].copy()
    return filtered_df


def get_all_domains(df: pd.DataFrame) -> list[str]:
    """
    Get list of all unique domains in the dataset.
    
    Args:
        df: Careers DataFrame
        
    Returns:
        Sorted list of unique domain names
    """
    return sorted(df['domain'].unique().tolist())


def get_career_details(df: pd.DataFrame, career_name: str) -> Optional[dict]:
    """
    Get details for a specific career.
    
    Args:
        df: Careers DataFrame
        career_name: Name of the career
        
    Returns:
        Dictionary with career details or None if not found
    """
    matches = df[df['career'].str.lower() == career_name.lower()]
    
    if len(matches) == 0:
        return None
    
    row = matches.iloc[0]
    return {
        'career': row['career'],
        'domain': row['domain'],
        'skills': row['skills'],
        'personality': row['personality'],
        'keywords': row['keywords'],
        'description': row['description']
    }
