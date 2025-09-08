# tools/plot_metrics.py
from pathlib import Path
import sys
import pandas as pd
import matplotlib.pyplot as plt

def plot_accuracy_per_experience(csv_path: Path, title: str | None = None, savepath: Path | None = None):
    df = pd.read_csv(csv_path)
    # Clean leftover index columns if any
    for col in list(df.columns):
        if col.startswith("Unnamed"):
            del df[col]

    # Normalize column names
    if "accuracy" not in df.columns and "value" in df.columns:
        df = df.rename(columns={"value": "accuracy"})
    if "experience" not in df.columns:
        raise ValueError(f"'experience' column not found in {csv_path}")

    df = df.sort_values("experience")

    plt.figure()
    plt.plot(df["experience"], df["accuracy"], marker="o")
    plt.xlabel("Experience")
    plt.ylabel("Accuracy")
    if title:
        plt.title(title)
    plt.grid(True, linestyle="--", linewidth=0.6)
    plt.tight_layout()
    if savepath is None:
        savepath = csv_path.with_suffix(".png")
    plt.savefig(savepath, dpi=220)
    print(f"✅ Saved plot: {savepath}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tools/plot_metrics.py results/gem_cicids2017_accuracy_per_experience.csv ['Optional Title']")
        sys.exit(1)
    csv_in = Path(sys.argv[1])
    title = sys.argv[2] if len(sys.argv) >= 3 else None
    plot_accuracy_per_experience(csv_in, title=title)
