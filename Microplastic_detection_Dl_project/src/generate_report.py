"""Generate a CSV + Markdown report of particle counts and sizes."""
import argparse
from pathlib import Path
import pandas as pd
from estimate_size import estimate_sizes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="models/best.pt")
    p.add_argument("--source", required=True)
    p.add_argument("--um-per-pixel", type=float, required=True)
    p.add_argument("--out", default="results/reports")
    a = p.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(estimate_sizes(a.weights, a.source, a.um_per_pixel))
    if df.empty:
        print("No particles detected."); return
    df.to_csv(out / "particles.csv", index=False)

    summary = df.groupby("image").agg(count=("particle", "count"),
                                      mean_size_um=("max_dim_um", "mean")).round(2)
    (out / "report.md").write_text(
        "# Microplastic Detection Report\n\n"
        f"Total particles: {len(df)}\n\n" + summary.to_markdown() + "\n")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    main()
