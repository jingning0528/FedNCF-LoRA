#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# Global plot style (same as analyze_AB.py)
# =========================================================
plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 17,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 15,
    "figure.titlesize": 18,
})

# =========================================================
# Paths
# =========================================================
REPO_ROOT = Path(__file__).resolve().parent
INPUT_CSV = REPO_ROOT / "csv" / "ab_convergence" / "AB_Comparison.csv"
OUT_DIR = REPO_ROOT / "figures" / "ab_convergence" / "comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)
    

def line_plot(df: pd.DataFrame, x: str, ys: list[str], ylabel: str, out_path: Path):
    plt.figure(figsize=(4.5, 3.5))

    legend_map = {
        "norm_delta_A": "A",
        "cos_delta_A": "A",
        "norm_delta_B": "B",
        "cos_delta_B": "B",
    }

    for y in ys:
        if y in df.columns:
            plt.plot(
                df[x],
                df[y],
                linewidth=2.5,
                label=legend_map.get(y, y),
            )

    plt.xlabel("Communication Round", fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.tick_params(axis="both", labelsize=14)

    plt.grid(
        True,
        linestyle="--",
        linewidth=0.5,
        alpha=0.5,
    )

    plt.legend(fontsize=13, frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")  # PDF by suffix
    plt.close()


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    required_cols = ["turn", "norm_delta_A", "norm_delta_B", "cos_delta_A", "cos_delta_B"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {INPUT_CSV}: {missing}")

    df["turn"] = pd.to_numeric(df["turn"], errors="coerce")
    for c in ["norm_delta_A", "norm_delta_B", "cos_delta_A", "cos_delta_B"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["turn"]).sort_values("turn").reset_index(drop=True)

    # Figure 1: A/B norm magnitude
    line_plot(
        df=df,
        x="turn",
        ys=["norm_delta_A", "norm_delta_B"],
        ylabel="Normalized Magnitude",
        out_path=OUT_DIR / "AB_norm_magnitude.pdf",
    )

    # Figure 2: A/B cosine similarity
    line_plot(
        df=df,
        x="turn",
        ys=["cos_delta_A", "cos_delta_B"],
        ylabel="Cosine similarity",
        out_path=OUT_DIR / "AB_cosine_similarity.pdf",
    )

    print(f"Saved: {OUT_DIR / 'AB_norm_magnitude.pdf'}")
    print(f"Saved: {OUT_DIR / 'AB_cosine_similarity.pdf'}")


if __name__ == "__main__":
    main()