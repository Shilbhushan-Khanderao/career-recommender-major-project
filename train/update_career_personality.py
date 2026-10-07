"""
Re-derive the `personality` column of data/careers_dataset.csv from each
career's job title (see role_personality in generate_comprehensive_careers.py).

Only the personality column changes; every other column is left untouched, so
the rest of the dataset (and the reported results that depend on it) is stable.

Run from the project root:  python train/update_career_personality.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).parent))
from generate_comprehensive_careers import role_personality  # noqa: E402

CSV = Path(__file__).parent.parent / "data" / "careers_dataset.csv"

df = pd.read_csv(CSV)
before = df["personality"].nunique()
df["personality"] = [role_personality(d, c) for c, d in zip(df["career"], df["domain"])]
df.to_csv(CSV, index=False)

per_domain = df.groupby("domain")["personality"].nunique()
print(f"Distinct personality strings: {before} -> {df['personality'].nunique()}")
print(f"Domains whose careers all share one personality: {(per_domain == 1).sum()} / {len(per_domain)}")
print(df["personality"].value_counts().head(10))
