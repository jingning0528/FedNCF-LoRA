#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyze full-item-embedding convergence logs from FedNCF full analyze runs.

Expected log lines (from fedncf_full_analyze.py):
[Item_Track] turn=9 delta_item_F=... norm_delta_item=... cos_delta_prev=... item_F=... norm_item_vs_init=...
"""

import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# =========================
# Paths
# =========================
REPO_ROOT = Path(__file__).resolve().parent
LOG_DIR = REPO_ROOT / "log"
FIG_DIR = REPO_ROOT / "figures" / "full_convergence"
CSV_DIR = REPO_ROOT / "csv" / "full_convergence"

FIG_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR.mkdir(parents=True, exist_ok=True)

# Put your selected logs here (relative to LOG_DIR)
SELECTED_LOG_FILES = [
    "analyze_1000/FedNCF-Full.txt",
    
]


# =========================
# Regex
# =========================
NUM = r"(?:[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?|nan|inf|-inf)"

ITEM_TRACK_PATTERN = re.compile(
    rf"\[Item_Track\]\s+turn=(?P<turn>-?\d+)\s+"
    rf"delta_item_F=(?P<delta_item_F>{NUM})\s+"
    rf"norm_delta_item=(?P<norm_delta_item>{NUM})\s+"
    rf"cos_delta_prev=(?P<cos_delta_prev>{NUM})\s+"
    rf"item_F=(?P<item_F>{NUM})\s+"
    rf"norm_item_vs_init=(?P<norm_item_vs_init>{NUM})",
    flags=re.IGNORECASE,
)

# Optional baseline line:
# [Item_Track] turn=-1 initialized tracking baseline. item_F=12.34
ITEM_TRACK_BASELINE_PATTERN = re.compile(
    rf"\[Item_Track\]\s+turn=(?P<turn>-?\d+)\s+initialized tracking baseline\.\s+item_F=(?P<item_F>{NUM})",
    flags=re.IGNORECASE,
)


def to_float(x: str) -> float:
    xl = x.lower()
    if xl == "nan":
        return float("nan")
    if xl == "inf":
        return float("inf")
    if xl == "-inf":
        return float("-inf")
    return float(x)


def parse_log(log_path: Path) -> pd.DataFrame:
    rows = []
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = ITEM_TRACK_PATTERN.search(line)
            if m:
                d = m.groupdict()
                rows.append(
                    {
                        "turn": int(d["turn"]),
                        "delta_item_F": to_float(d["delta_item_F"]),
                        "norm_delta_item": to_float(d["norm_delta_item"]),
                        "cos_delta_prev": to_float(d["cos_delta_prev"]),
                        "item_F": to_float(d["item_F"]),
                        "norm_item_vs_init": to_float(d["norm_item_vs_init"]),
                    }
                )
                continue

            b = ITEM_TRACK_BASELINE_PATTERN.search(line)
            if b:
                d = b.groupdict()
                rows.append(
                    {
                        "turn": int(d["turn"]),
                        "delta_item_F": float("nan"),
                        "norm_delta_item": float("nan"),
                        "cos_delta_prev": float("nan"),
                        "item_F": to_float(d["item_F"]),
                        "norm_item_vs_init": float("nan"),
                    }
                )

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No [Item_Track] rows found in: {log_path}")
    return df.sort_values("turn").reset_index(drop=True)


def line_plot(df: pd.DataFrame, y_cols: list[str], title: str, ylabel: str, out_path: Path):
    plt.figure(figsize=(9, 5))
    for c in y_cols:
        if c in df.columns:
            plt.plot(df["turn"], df[c], marker="o", linewidth=2.0, markersize=5, label=c)
    # plt.title(title)
    plt.xlabel("round", fontsize=15)
    plt.ylabel(ylabel, fontsize=15)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=15)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def safe_exp_name(rel_path: str) -> str:
    return Path(rel_path).with_suffix("").as_posix().replace("/", "__").replace(" ", "_")


def main():
    if not SELECTED_LOG_FILES:
        raise ValueError("SELECTED_LOG_FILES is empty. Add at least one log file.")

    for rel in SELECTED_LOG_FILES:
        log_path = LOG_DIR / rel
        if not log_path.exists():
            print(f"[Skip] not found: {log_path}")
            continue

        exp = safe_exp_name(rel)
        exp_fig = FIG_DIR / exp
        exp_csv = CSV_DIR / exp
        exp_fig.mkdir(parents=True, exist_ok=True)
        exp_csv.mkdir(parents=True, exist_ok=True)

        df = parse_log(log_path)

        # Save parsed rows
        df.to_csv(exp_csv / "item_track_metrics.csv", index=False)

        # 1) Magnitude (Frobenius norm)
        line_plot(
            df[df["turn"] >= 0],
            ["delta_item_F", "item_F"],
            "Item Embedding Magnitude",
            "Frobenius norm",
            exp_fig / "01_magnitude.png",
        )

        # 2) Normalized magnitude
        line_plot(
            df[df["turn"] >= 0],
            ["norm_delta_item", "norm_item_vs_init"],
            "Item Embedding Normalized Magnitude",
            "Normalized value",
            exp_fig / "02_normalized_magnitude.png",
        )

        # 3) Cosine similarity
        line_plot(
            df[df["turn"] >= 0],
            ["cos_delta_prev"],
            "Cosine Similarity of Consecutive Item Updates",
            "Cosine similarity",
            exp_fig / "03_cosine_similarity.png",
        )

        print(f"[Done] {log_path}")
        print(f"  CSV: {exp_csv}")
        print(f"  FIG: {exp_fig}")


if __name__ == "__main__":
    main()