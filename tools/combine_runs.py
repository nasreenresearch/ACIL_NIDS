# tools/combine_runs.py
"""
Combine multiple *_accuracy_per_experience.csv files into one wide table.

Usage:
  python tools/combine_runs.py results/gem_cicids2017_accuracy_per_experience.csv \
                               results/ewc_cicids2017_accuracy_per_experience.csv \
                               results/replay_cicids2017_accuracy_per_experience.csv
"""
import sys
from pathlib import Path
import pandas as pd

def label_from_path(p: Path) -> str:
    # Expect pattern: <strategy>_<dataset>_accuracy_per_experience.csv
    stem = p.stem
    if stem.endswith("_accuracy_per_experience"):
        stem = stem[: -len("_accuracy_per_experience")]
    return stem  # e.g., "gem_cicids2017"

def main(paths):
    dfs = []
    for p in map(Path, paths):
        df = pd.read_csv(p)
        for col in list(df.columns):
            if col.startswith("Unnamed"):
                del df[col]
        if "accuracy" not in df.columns and "value" in df.columns:
            df = df.rename(columns={"value": "accuracy"})
        if "experience" not in df.columns:
            raise ValueError(f"'experience' column not found in {p}")
        df = df[["experience", "accuracy"]].copy()
        df = df.rename(columns={"accuracy": label_from_path(p)})
        dfs.append(df)

    wide = dfs[0]
    for df in dfs[1:]:
        wide = pd.merge(wide, df, on="experience", how="outer")
    wide = wide.sort_values("experience")
    out = Path(paths[0]).parent / "combined_accuracy_per_experience.csv"
    wide.to_csv(out, index=False)
    print(f"✅ Wrote combined table: {out}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1:])
