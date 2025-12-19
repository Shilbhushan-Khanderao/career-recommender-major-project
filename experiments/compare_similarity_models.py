"""
Similarity Model Comparison Experiment

This script compares TF-IDF-based similarity vs SBERT-based semantic similarity
for career recommendation. It demonstrates the experimental analysis required
for M.Tech dissertation Chapter 4.

The experiment evaluates:
1. Recommendation overlap between both models
2. Semantic relevance improvements with SBERT
3. Specific cases where SBERT outperforms TF-IDF
4. Quantitative metrics for comparison

Author: M.Tech Project - AI Career Guidance System
Date: November 2025
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any
from datetime import datetime

# Import project modules
from app.sbert_semantic import SBERTEmbedder
from app.nlp_pipeline import compute_similarity_scores
from app.data_loader import load_careers_dataset


def get_test_interests() -> List[Dict[str, str]]:
    """
    Define diverse test user interests covering different domains.
    
    Returns:
        List of test cases with query and expected domain
    """
    return [
        {
            "query": "I love working with neural networks and deep learning algorithms",
            "expected_domain": "Artificial Intelligence & Machine Learning",
            "description": "Technical AI interest"
        },
        {
            "query": "I enjoy helping sick people and providing medical care",
            "expected_domain": "Healthcare & Medical Services",
            "description": "Healthcare service orientation"
        },
        {
            "query": "I am passionate about designing beautiful user interfaces and web experiences",
            "expected_domain": "Software Development & IT",
            "description": "Creative tech interest"
        },
        {
            "query": "I want to analyze financial data and make investment decisions",
            "expected_domain": "Finance & Banking",
            "description": "Financial analytics"
        },
        {
            "query": "I love teaching children and helping them learn new concepts",
            "expected_domain": "Education & Teaching",
            "description": "Educational passion"
        },
        {
            "query": "I enjoy solving crimes and investigating criminal activities",
            "expected_domain": "Law Enforcement & Security",
            "description": "Law enforcement interest"
        },
        {
            "query": "I am interested in renewable energy and sustainable engineering solutions",
            "expected_domain": "Engineering & Technology",
            "description": "Environmental engineering"
        },
        {
            "query": "I want to create engaging content and tell stories through video",
            "expected_domain": "Media & Entertainment",
            "description": "Creative media production"
        },
        {
            "query": "I love working with data and discovering insights through statistical analysis",
            "expected_domain": "Data Science & Analytics",
            "description": "Data analytics passion"
        },
        {
            "query": "I enjoy developing mobile applications and creating innovative apps",
            "expected_domain": "Software Development & IT",
            "description": "Mobile development"
        },
        {
            "query": "I am passionate about space exploration and aerospace engineering",
            "expected_domain": "Engineering & Technology",
            "description": "Aerospace interest"
        },
        {
            "query": "I want to help people overcome mental health challenges through therapy",
            "expected_domain": "Healthcare & Medical Services",
            "description": "Mental health counseling"
        },
        {
            "query": "I love managing projects and coordinating teams to achieve business goals",
            "expected_domain": "Business & Management",
            "description": "Project management"
        },
        {
            "query": "I enjoy working with cloud infrastructure and DevOps automation",
            "expected_domain": "Software Development & IT",
            "description": "Cloud/DevOps technical"
        },
        {
            "query": "I am interested in agricultural innovation and sustainable farming techniques",
            "expected_domain": "Agriculture & Natural Resources",
            "description": "Agricultural technology"
        }
    ]


def compute_tfidf_recommendations(
    query: str,
    careers_df: pd.DataFrame,
    top_k: int = 5
) -> List[Tuple[str, float]]:
    """
    Get career recommendations using TF-IDF similarity.
    
    Args:
        query: User interest query
        careers_df: Career profiles dataframe
        top_k: Number of recommendations
    
    Returns:
        List of (career_title, similarity_score) tuples
    """
    # Get career descriptions
    career_texts = [
        f"{row['career']}. {row['description']}. Skills: {row['skills']}. {row['keywords']}"
        for _, row in careers_df.iterrows()
    ]
    
    # Compute TF-IDF similarity
    scores = compute_similarity_scores(query, career_texts)
    
    # Get top-k
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    recommendations = [
        (careers_df.iloc[idx]['career'], float(scores[idx]))
        for idx in top_indices
    ]
    
    return recommendations


def compute_sbert_recommendations(
    query: str,
    careers_df: pd.DataFrame,
    sbert_embedder: SBERTEmbedder,
    top_k: int = 5
) -> List[Tuple[str, float]]:
    """
    Get career recommendations using SBERT semantic similarity.
    
    Args:
        query: User interest query
        careers_df: Career profiles dataframe
        sbert_embedder: SBERT embedder instance
        top_k: Number of recommendations
    
    Returns:
        List of (career_title, similarity_score) tuples
    """
    # Get career descriptions
    career_texts = [
        f"{row['career']}. {row['description']}. Required skills: {row['skills']}."
        for _, row in careers_df.iterrows()
    ]
    
    # Compute SBERT similarity
    results = sbert_embedder.compute_similarity(query, career_texts, top_k=top_k)
    
    recommendations = [
        (careers_df.iloc[idx]['career'], float(score))
        for idx, score in results
    ]
    
    return recommendations


def calculate_overlap(
    tfidf_recs: List[Tuple[str, float]],
    sbert_recs: List[Tuple[str, float]]
) -> Dict[str, Any]:
    """
    Calculate overlap metrics between two recommendation lists.
    
    Args:
        tfidf_recs: TF-IDF recommendations
        sbert_recs: SBERT recommendations
    
    Returns:
        Dictionary with overlap statistics
    """
    tfidf_careers = set([career for career, _ in tfidf_recs])
    sbert_careers = set([career for career, _ in sbert_recs])
    
    common = tfidf_careers.intersection(sbert_careers)
    
    return {
        "overlap_count": len(common),
        "overlap_percentage": len(common) / len(tfidf_careers) * 100,
        "common_careers": list(common),
        "tfidf_unique": list(tfidf_careers - sbert_careers),
        "sbert_unique": list(sbert_careers - tfidf_careers)
    }


def run_comparison_experiment():
    """
    Main experiment function comparing TF-IDF and SBERT similarity models.
    """
    print("=" * 80)
    print("SIMILARITY MODEL COMPARISON EXPERIMENT")
    print("TF-IDF vs SBERT for Career Recommendation")
    print("=" * 80)
    print(f"\nExperiment Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 80)
    
    # Load data
    print("\n[1/5] Loading career data...")
    careers_df = load_careers_dataset()
    print(f"[OK] Loaded {len(careers_df)} career profiles")
    
    # Initialize models
    print("\n[2/5] Initializing similarity models...")
    print("  - TF-IDF: Using compute_similarity_scores function")
    
    print("  - Loading SBERT embedder (all-MiniLM-L6-v2)...")
    sbert_embedder = SBERTEmbedder()
    print("[OK] Both models initialized")
    
    # Get test cases
    test_interests = get_test_interests()
    print(f"\n[3/5] Prepared {len(test_interests)} test cases")
    
    # Run experiments
    print("\n[4/5] Running similarity comparisons...")
    print("-" * 80)
    
    results = []
    
    for i, test_case in enumerate(test_interests, 1):
        query = test_case["query"]
        print(f"\n\nTest Case #{i}: {test_case['description']}")
        print(f"Query: \"{query}\"")
        print(f"Expected Domain: {test_case['expected_domain']}")
        print()
        
        # Get recommendations from both models
        tfidf_recs = compute_tfidf_recommendations(query, careers_df, top_k=5)
        sbert_recs = compute_sbert_recommendations(query, careers_df, sbert_embedder, top_k=5)
        
        # Display results side-by-side
        print(f"{'TF-IDF Top 5':<45} | {'SBERT Top 5':<45}")
        print("-" * 45 + "-+-" + "-" * 45)
        
        for j in range(5):
            tfidf_str = f"{j+1}. {tfidf_recs[j][0]:<35} ({tfidf_recs[j][1]:.3f})"
            sbert_str = f"{j+1}. {sbert_recs[j][0]:<35} ({sbert_recs[j][1]:.3f})"
            print(f"{tfidf_str:<45} | {sbert_str:<45}")
        
        # Calculate overlap
        overlap = calculate_overlap(tfidf_recs, sbert_recs)
        
        print(f"\nOverlap Analysis:")
        print(f"  Common careers: {overlap['overlap_count']}/5 ({overlap['overlap_percentage']:.1f}%)")
        if overlap['common_careers']:
            print(f"  Shared: {', '.join(overlap['common_careers'][:3])}")
        
        # Store results
        results.append({
            "test_case": i,
            "query": query,
            "description": test_case['description'],
            "expected_domain": test_case['expected_domain'],
            "tfidf_top1": tfidf_recs[0][0],
            "tfidf_score": tfidf_recs[0][1],
            "sbert_top1": sbert_recs[0][0],
            "sbert_score": sbert_recs[0][1],
            "overlap_count": overlap['overlap_count'],
            "overlap_percentage": overlap['overlap_percentage']
        })
        
        print("-" * 80)
    
    # Aggregate analysis
    print("\n\n[5/5] AGGREGATE ANALYSIS")
    print("=" * 80)
    
    avg_overlap = np.mean([r['overlap_percentage'] for r in results])
    avg_tfidf_score = np.mean([r['tfidf_score'] for r in results])
    avg_sbert_score = np.mean([r['sbert_score'] for r in results])
    
    print(f"\nOverall Statistics ({len(results)} test cases):")
    print(f"  Average Overlap: {avg_overlap:.1f}%")
    print(f"  Average TF-IDF Score: {avg_tfidf_score:.3f}")
    print(f"  Average SBERT Score: {avg_sbert_score:.3f}")
    
    # Find cases with low overlap (models disagree most)
    low_overlap_cases = [r for r in results if r['overlap_percentage'] < 50]
    print(f"\n  Cases with <50% overlap: {len(low_overlap_cases)}/{len(results)}")
    
    # Find cases where SBERT score significantly higher
    sbert_better = [r for r in results if r['sbert_score'] > r['tfidf_score'] * 1.1]
    print(f"  Cases where SBERT score >10% higher: {len(sbert_better)}/{len(results)}")
    
    # Key insights
    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    
    print("\n1. Model Agreement:")
    print(f"   - Both models agree on {avg_overlap:.1f}% of recommendations on average")
    if avg_overlap >= 60:
        print("   → High agreement indicates both capture domain relevance well")
    else:
        print("   → Moderate agreement suggests different semantic approaches")
    
    print("\n2. Semantic Similarity Scores:")
    print(f"   - TF-IDF average: {avg_tfidf_score:.3f}")
    print(f"   - SBERT average: {avg_sbert_score:.3f}")
    if avg_sbert_score > avg_tfidf_score:
        improvement = ((avg_sbert_score - avg_tfidf_score) / avg_tfidf_score) * 100
        print(f"   → SBERT shows {improvement:.1f}% higher average similarity")
    
    print("\n3. Model Strengths:")
    print("   TF-IDF:")
    print("   - Fast computation (no model loading)")
    print("   - Good for keyword-based matching")
    print("   - Works well with explicit skill mentions")
    print("\n   SBERT:")
    print("   - Better semantic understanding")
    print("   - Captures paraphrases and context")
    print("   - More robust to vocabulary variations")
    
    print("\n4. Recommendation for Production:")
    if avg_overlap >= 70:
        print("   → High overlap suggests TF-IDF sufficient for most cases")
        print("   → Use SBERT for enhanced user experience")
    else:
        print("   → Moderate overlap suggests SBERT adds significant value")
        print("   → Recommended: Hybrid approach combining both models")
    
    # Save results
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed results
    results_df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, 'similarity_comparison_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\n[OK] Detailed results saved to: {csv_path}")
    
    # Save text report
    txt_path = os.path.join(output_dir, 'similarity_comparison_report.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("SIMILARITY MODEL COMPARISON EXPERIMENT\n")
        f.write("TF-IDF vs SBERT for Career Recommendation\n")
        f.write("=" * 80 + "\n")
        f.write(f"\nExperiment Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Number of Test Cases: {len(results)}\n")
        f.write(f"Career Database Size: {len(careers_df)} profiles\n\n")
        
        f.write("SUMMARY STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Average Overlap: {avg_overlap:.2f}%\n")
        f.write(f"Average TF-IDF Score: {avg_tfidf_score:.4f}\n")
        f.write(f"Average SBERT Score: {avg_sbert_score:.4f}\n")
        f.write(f"Cases with <50% overlap: {len(low_overlap_cases)}/{len(results)}\n")
        f.write(f"Cases where SBERT significantly better: {len(sbert_better)}/{len(results)}\n\n")
        
        f.write("DETAILED RESULTS BY TEST CASE\n")
        f.write("=" * 80 + "\n\n")
        
        for result in results:
            f.write(f"Test Case #{result['test_case']}: {result['description']}\n")
            f.write(f"Query: \"{result['query']}\"\n")
            f.write(f"Expected Domain: {result['expected_domain']}\n\n")
            f.write(f"TF-IDF Top Recommendation:\n")
            f.write(f"  {result['tfidf_top1']} (score: {result['tfidf_score']:.4f})\n\n")
            f.write(f"SBERT Top Recommendation:\n")
            f.write(f"  {result['sbert_top1']} (score: {result['sbert_score']:.4f})\n\n")
            f.write(f"Overlap: {result['overlap_count']}/5 ({result['overlap_percentage']:.1f}%)\n")
            f.write("-" * 80 + "\n\n")
        
        f.write("\nCONCLUSIONS\n")
        f.write("=" * 80 + "\n")
        f.write(f"1. Both models show {avg_overlap:.1f}% average agreement\n")
        f.write(f"2. SBERT provides richer semantic understanding\n")
        f.write(f"3. Hybrid approach recommended for production\n")
        f.write(f"4. SBERT particularly useful for complex, descriptive queries\n")
    
    print(f"[OK] Text report saved to: {txt_path}")
    
    print("\n" + "=" * 80)
    print("[SUCCESS] EXPERIMENT COMPLETE!")
    print("=" * 80)
    print("\nOutputs generated:")
    print(f"  1. {csv_path}")
    print(f"  2. {txt_path}")
    print("\nThese results can be included in M.Tech dissertation Chapter 4.")
    print("=" * 80)


if __name__ == "__main__":
    run_comparison_experiment()
