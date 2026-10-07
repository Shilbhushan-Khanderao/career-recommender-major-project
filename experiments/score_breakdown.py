"""
Print the real hybrid score breakdown for a query (basis for report Table 5.5 /
Appendix V). Every number comes from the deployed recommender, nothing is
hand-written.

Usage (from project root):
    python -m experiments.score_breakdown "I enjoy analysing data and building machine learning models"
    python -m experiments.score_breakdown "..." --domain data_science_ai --engine tfidf
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.domain_predictor import predict_domain  # noqa: E402
from app.recommender import recommend_careers_for_domain  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text")
    ap.add_argument("--domain", default=None, help="skip prediction and use this domain")
    ap.add_argument("--engine", default="hybrid", choices=["tfidf", "sbert", "hybrid"])
    ap.add_argument("--top", type=int, default=5)
    a = ap.parse_args()

    domain = a.domain
    if domain is None:
        res = predict_domain(a.text)
        domain = res if isinstance(res, str) else res["domain"]
    print(f"Input : {a.text}\nDomain: {domain}   Engine: {a.engine}\n")

    recs = recommend_careers_for_domain(a.text, domain, top_k=a.top, similarity_model=a.engine)
    hdr = f"{'Career':34s}{'Semantic':>9s}{'Keyword':>9s}{'Personal.':>10s}{'Skills':>8s}{'Hybrid':>8s}{'RuleBoost':>10s}{'Final':>8s}"
    print(hdr + "\n" + "-" * len(hdr))
    for r in recs:
        print(f"{r['career']:34s}{r['similarity_score']:9.3f}{r['keyword_score']:9.3f}"
              f"{r['personality_score']:10.3f}{r['skills_score']:8.3f}{r['hybrid_score']:8.3f}"
              f"{r['rule_boost']:10.3f}{r['final_score']:8.3f}")
    print("\nHybrid = 0.40*Semantic + 0.25*Keyword + 0.20*Personality + 0.15*Skills  (Equation 4.1)")
    print("Final  = Hybrid + rule-based domain boost (constant for the predicted domain; ranking is by Hybrid)")


if __name__ == "__main__":
    main()
