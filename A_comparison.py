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
INPUT_CSV = REPO_ROOT / "csv" / "ab_convergence" / "A_Comparison.csv"
OUT_DIR = REPO_ROOT / "figures" / "ab_convergence" / "comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def line_plot(df: pd.DataFrame, x: str, ys: list[str], ylabel: str, out_path: Path):
    plt.figure(figsize=(4.5, 3.5))

    color_map = {
        "LoRA": "tab:orange",
        "LoRA-FixedB": "tab:green",
        "MofiLoRA": "tab:red",
    }

    for y in ys:
        if y in df.columns:
            plt.plot(
                df[x],
                df[y],
                linewidth=2.5,
                label=y,
                color=color_map.get(y, None),
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

    plt.legend(
        fontsize=13,
        frameon=False,
    )

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")  # saves as PDF by suffix
    plt.close()


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    raw = pd.read_csv(INPUT_CSV)

    cols = list(raw.columns)
    if len(cols) < 9:
        raise ValueError(f"Unexpected format in {INPUT_CSV}. Need 9 columns, got {len(cols)}")

    norm_df = raw.iloc[:, [0, 1, 2, 3, 4]].copy()
    norm_df.columns = ["experiment", "turn", "LoRA", "LoRA-FixedB", "MofiLoRA"]

    cos_df = raw.iloc[:, [0, 5, 6, 7, 8]].copy()
    cos_df.columns = ["experiment", "turn", "LoRA", "LoRA-FixedB", "MofiLoRA"]

    for df in (norm_df, cos_df):
        df["turn"] = pd.to_numeric(df["turn"], errors="coerce")
        for c in ["LoRA", "LoRA-FixedB", "MofiLoRA"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df.dropna(subset=["turn"], inplace=True)
        df.sort_values("turn", inplace=True)
        df.reset_index(drop=True, inplace=True)

    line_plot(
        norm_df,
        x="turn",
        ys=["LoRA", "LoRA-FixedB", "MofiLoRA"],
        ylabel="Normalized Magnitude",
        out_path=OUT_DIR / "A_compare_norm_magnitude.pdf",
    )

    line_plot(
        cos_df,
        x="turn",
        ys=["LoRA", "LoRA-FixedB", "MofiLoRA"],
        ylabel="Cosine similarity",
        out_path=OUT_DIR / "A_compare_cosine_similarity.pdf",
    )

    print(f"Saved: {OUT_DIR / 'A_compare_norm_magnitude.pdf'}")
    print(f"Saved: {OUT_DIR / 'A_compare_cosine_similarity.pdf'}")


if __name__ == "__main__":
    main()