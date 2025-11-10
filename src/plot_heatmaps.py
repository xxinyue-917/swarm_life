
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

METRICS = ["revs", "d12", "Rdiff", "K", "R1", "R2", "d11", "d22"]

def plot_one(df_case, metric, outdir):
    # Average over seeds
    g = df_case.groupby(["k12_pos", "k21_pos"], as_index=False)[metric].mean()
    # Pivot to grid
    table = g.pivot(index="k21_pos", columns="k12_pos", values=metric).sort_index(ascending=True)
    # Plot
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(table.values, origin="lower", aspect="auto",
                   extent=[table.columns.min(), table.columns.max(),
                           table.index.min(), table.index.max()])
    ax.set_xlabel("k12_pos")
    ax.set_ylabel("k21_pos")
    ax.set_title(f"{metric}")
    cb = plt.colorbar(im, ax=ax)
    # Save
    outpath = Path(outdir) / f"{metric}.png"
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    return outpath

def main():
    ap = argparse.ArgumentParser(description="Plot heatmaps from sweep CSV.")
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--out", type=str, default="plots")
    args = ap.parse_args()

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv)

    for case in df["ori_case"].unique():
        sub = df[df["ori_case"] == case]
        case_dir = outdir / f"case_{case}"
        case_dir.mkdir(exist_ok=True, parents=True)
        for m in METRICS:
            path = plot_one(sub, m, case_dir)
            print(f"Saved: {path}")

if __name__ == "__main__":
    main()
