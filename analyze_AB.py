#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze A/B convergence metrics from FedNCF-LoRA log txt files.

How to use:
1. Put this script in your project root, or adjust REPO_ROOT below.
2. Put your log txt files under LOG_DIR = REPO_ROOT / "log".
3. Add/uncomment file names in SELECTED_LOG_FILES.
4. Run:
       python analyze_ab_convergence_selected_logs.py

Outputs:
    figures/ab_convergence/<experiment_name>/
    csv/ab_convergence/<experiment_name>/
    figures/ab_convergence/comparison/
    csv/ab_convergence/comparison/

Expected log lines:
    [AB-Convergence] turn=109 delta_A_F=... delta_B_F=... norm_delta_A=... norm_delta_B=... cos_delta_A=... cos_delta_B=... effective_item_update_F=...
    [Metrics] logloss: ... - MRR: ... - NDCG(5): ... - HR(5): ...
"""

import re
import math
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# Global plot style (bigger text for all figures)
# =========================================================
plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 17,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 12,
    "figure.titlesize": 18,
})


# =========================================================
# Project paths
# =========================================================

# If this script is directly inside your project root, use this:
REPO_ROOT = Path(__file__).resolve().parent

# If this script is inside a subfolder, use this instead:
# REPO_ROOT = Path(__file__).resolve().parents[1]

LOG_DIR = REPO_ROOT / "log"
FIG_DIR = REPO_ROOT / "figures"
CSV_DIR = REPO_ROOT / "csv"

AB_FIG_DIR = FIG_DIR / "ab_convergence"
AB_CSV_DIR = CSV_DIR / "ab_convergence"

AB_FIG_DIR.mkdir(parents=True, exist_ok=True)
AB_CSV_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# Select log files here
# The paths below are relative to LOG_DIR
# Example: LOG_DIR / "baseline_1000/fedncf_lora.txt"
# =========================================================

SELECTED_LOG_FILES = [

    # ************************************** baseline ********************************
    # "analyze/lora.txt",
    # "analyze/momentumA_fixedB.txt",
    "analyze/lora_1000.txt",


]


# =========================================================
# Regex patterns
# =========================================================

# Accept float / sci-notation / nan / inf
NUM_TOKEN = r"(?:[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?|nan|inf|-inf)"

AB_PATTERN = re.compile(
    rf"\[AB-Convergence\]\s+"
    rf"turn=(?P<turn>\d+)\s+"
    rf"delta_A_F=(?P<delta_A_F>{NUM_TOKEN})\s+"
    rf"delta_B_F=(?P<delta_B_F>{NUM_TOKEN})\s+"
    rf"norm_delta_A=(?P<norm_delta_A>{NUM_TOKEN})\s+"
    rf"norm_delta_B=(?P<norm_delta_B>{NUM_TOKEN})\s+"
    rf"cos_delta_A=(?P<cos_delta_A>{NUM_TOKEN})\s+"
    rf"cos_delta_B=(?P<cos_delta_B>{NUM_TOKEN})\s+"
    rf"effective_item_update_F=(?P<effective_item_update_F>{NUM_TOKEN})",
    flags=re.IGNORECASE
)

AB_TRACK_PATTERN = re.compile(
    rf"\[AB_Track\]\s+"
    rf"turn=(?P<turn>-?\d+)\s+"
    rf"delta_A_F=(?P<delta_A_F>{NUM_TOKEN})\s+"
    rf"delta_B_F=(?P<delta_B_F>{NUM_TOKEN})\s+"
    rf"norm_delta_A=(?P<norm_delta_A>{NUM_TOKEN})\s+"
    rf"norm_delta_B=(?P<norm_delta_B>{NUM_TOKEN})\s+"
    rf"cos_delta_A_prev=(?P<cos_delta_A>{NUM_TOKEN})\s+"
    rf"cos_delta_B_prev=(?P<cos_delta_B>{NUM_TOKEN})\s+"
    rf"effective_embedding_delta_F=(?P<effective_item_update_F>{NUM_TOKEN})",
    flags=re.IGNORECASE
)

METRIC_PATTERN = re.compile(r"\[Metrics\]\s+(?P<body>.*)")
METRIC_KV_PATTERN = re.compile(r"([A-Za-z0-9_@()]+):\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")

def _to_float(x: str) -> float:
    xl = x.lower()
    if xl == "nan":
        return float("nan")
    if xl == "inf":
        return float("inf")
    if xl == "-inf":
        return float("-inf")
    return float(x)

def parse_ab_row(line: str):
    m = AB_PATTERN.search(line) or AB_TRACK_PATTERN.search(line)
    if not m:
        return None
    row = m.groupdict()
    out = {}
    for k, v in row.items():
        out[k] = int(v) if k == "turn" else _to_float(v)
    return out


# =========================================================
# Parsing
# =========================================================

def safe_exp_name(log_relative_path: str) -> str:
    """Create a safe experiment name from a relative log path."""
    return Path(log_relative_path).with_suffix("").as_posix().replace("/", "__").replace(" ", "_")


def parse_log(txt_path: Path):
    """Parse AB convergence rows and evaluation metric rows from one txt log."""
    ab_rows = []
    metric_rows = []
    last_ab_turn = None
    current_eval_turn = None

    eval_turn_pattern = re.compile(r"Eval @ Turn\s+(?P<turn>\d+)")

    with txt_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            ab_row = parse_ab_row(line)
            if ab_row is not None:
                ab_rows.append(ab_row)
                last_ab_turn = ab_row["turn"]
                continue

            eval_match = eval_turn_pattern.search(line)
            if eval_match:
                current_eval_turn = int(eval_match.group("turn"))
                continue

            metric_match = METRIC_PATTERN.search(line)
            if metric_match:
                body = metric_match.group("body")
                row = {k: float(v) for k, v in METRIC_KV_PATTERN.findall(body)}

                if current_eval_turn is not None:
                    row["turn"] = current_eval_turn
                elif last_ab_turn is not None:
                    row["turn"] = last_ab_turn
                else:
                    continue

                metric_rows.append(row)

    ab_df = pd.DataFrame(ab_rows)
    metric_df = pd.DataFrame(metric_rows)

    if ab_df.empty:
        raise ValueError(f"No [AB-Convergence]/[AB_Track] lines found in: {txt_path}")

    ab_df = ab_df.sort_values("turn").reset_index(drop=True)

    if not metric_df.empty:
        metric_df = (
            metric_df
            .drop_duplicates(subset=["turn"], keep="last")
            .sort_values("turn")
            .reset_index(drop=True)
        )

    return ab_df, metric_df


# =========================================================
# Derived columns and summary
# =========================================================

def add_summary_columns(ab_df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns for A/B comparison and stability analysis."""
    df = ab_df.copy()
    eps = 1e-12

    df["delta_A_over_B"] = df["delta_A_F"] / (df["delta_B_F"].abs() + eps)
    df["norm_delta_A_over_B"] = df["norm_delta_A"] / (df["norm_delta_B"].abs() + eps)

    # Change between logged points, useful for update stability.
    df["delta_A_F_change"] = df["delta_A_F"].diff().abs()
    df["delta_B_F_change"] = df["delta_B_F"].diff().abs()
    df["effective_update_change"] = df["effective_item_update_F"].diff().abs()

    return df


def summarize(ab_df: pd.DataFrame, merged_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Create compact summary statistics for one experiment."""
    cols = [
        "delta_A_F",
        "delta_B_F",
        "norm_delta_A",
        "norm_delta_B",
        "cos_delta_A",
        "cos_delta_B",
        "effective_item_update_F",
        "delta_A_over_B",
        "norm_delta_A_over_B",
        "delta_A_F_change",
        "delta_B_F_change",
        "effective_update_change",
    ]

    available = [c for c in cols if c in ab_df.columns]
    summary = ab_df[available].agg(["mean", "median", "std", "min", "max"]).T
    summary["cv_std_over_mean_abs"] = summary["std"] / summary["mean"].abs().replace(0, pd.NA)

    if merged_df is not None and not merged_df.empty:
        target_metrics = [c for c in ["HR(10)", "NDCG(10)", "logloss", "MRR"] if c in merged_df.columns]
        ab_cols = [
            "delta_A_F",
            "delta_B_F",
            "norm_delta_A",
            "norm_delta_B",
            "cos_delta_A",
            "cos_delta_B",
            "effective_item_update_F",
        ]
        for metric in target_metrics:
            corr = {}
            for c in ab_cols:
                if c in merged_df.columns:
                    corr[c] = merged_df[c].corr(merged_df[metric])
            if corr:
                summary[f"corr_with_{metric}"] = pd.Series(corr)

    return summary


# =========================================================
# Plot helpers
# =========================================================

def line_plot(df: pd.DataFrame, x: str, ys: list[str], title: str, ylabel: str, out_path: Path):
    plt.figure(figsize=(9, 5))
    for y in ys:
        if y in df.columns:
            plt.plot(df[x], df[y], marker="o", label=y, linewidth=2.0, markersize=5)
    plt.xlabel("round")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def scatter_plot(df: pd.DataFrame, x: str, y: str, title: str, out_path: Path):
    if x not in df.columns or y not in df.columns:
        return
    plt.figure(figsize=(7, 5))
    plt.scatter(df[x], df[y], s=32)
    plt.xlabel("round" if x.lower() == "turn" else x)
    plt.ylabel(y)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def comparison_line_plot(all_df: pd.DataFrame, y: str, title: str, ylabel: str, out_path: Path):
    if all_df.empty or y not in all_df.columns or "experiment" not in all_df.columns:
        return

    plt.figure(figsize=(10, 5))
    for exp_name, group in all_df.groupby("experiment"):
        group = group.sort_values("turn")
        plt.plot(group["turn"], group[y], marker="o", label=exp_name, linewidth=2.0, markersize=5)

    plt.xlabel("round")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# =========================================================
# Figures for one experiment
# =========================================================

def make_single_experiment_figures(ab_df: pd.DataFrame, metric_df: pd.DataFrame, out_dir: Path):
    """Draw figures for one log file."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # 0. Combined A/B figure (requested)
    line_plot(
        ab_df,
        "turn",
        ["delta_A_F", "delta_B_F", "norm_delta_A", "norm_delta_B"],
        "A/B Update Magnitudes (Raw + Normalized)",
        "Value",
        out_dir / "00_A_B_four_metrics_in_one.png",
    )

    # 1. Matrix magnitude
    line_plot(
        ab_df,
        "turn",
        ["delta_A_F", "delta_B_F"],
        "Matrix Magnitude: ||Delta A||_F vs ||Delta B||_F",
        "Frobenius norm",
        out_dir / "01_matrix_magnitude_A_B.png",
    )

    # 2. Normalized matrix magnitude
    line_plot(
        ab_df,
        "turn",
        ["norm_delta_A", "norm_delta_B"],
        "Normalized Matrix Magnitude",
        "Normalized Frobenius norm",
        out_dir / "02_normalized_matrix_magnitude_A_B.png",
    )

    # 3. Cosine similarity
    line_plot(
        ab_df,
        "turn",
        ["cos_delta_A", "cos_delta_B"],
        "Cosine Similarity of Consecutive Updates",
        "Cosine similarity",
        out_dir / "03_cosine_similarity_A_B.png",
    )

    # 4. Effective embedding update
    line_plot(
        ab_df,
        "turn",
        ["effective_item_update_F"],
        "Effective Item Embedding Update Magnitude",
        "Frobenius norm",
        out_dir / "04_effective_item_update.png",
    )

    # Extra: A/B ratio
    line_plot(
        ab_df,
        "turn",
        ["delta_A_over_B", "norm_delta_A_over_B"],
        "A/B Update Ratio",
        "Ratio",
        out_dir / "05_A_B_update_ratio.png",
    )

    # Extra: update stability
    line_plot(
        ab_df,
        "turn",
        ["delta_A_F_change", "delta_B_F_change", "effective_update_change"],
        "Update Magnitude Stability: Absolute Change Between Logged Points",
        "Absolute change",
        out_dir / "06_update_stability_change.png",
    )

    merged = pd.DataFrame()

    if metric_df is not None and not metric_df.empty:
        # Keep only one 'experiment' column from ab_df to avoid experiment_x/experiment_y
        metric_df_for_merge = metric_df.drop(columns=["experiment"], errors="ignore")
        merged = pd.merge(ab_df, metric_df_for_merge, on="turn", how="inner")

        eval_cols = [c for c in ["HR(10)", "NDCG(10)", "logloss", "MRR"] if c in merged.columns]
        if eval_cols:
            line_plot(
                merged,
                "turn",
                eval_cols,
                "Evaluation Metrics at AB Tracking Turns",
                "Metric value",
                out_dir / "07_eval_metrics.png",
            )

        for metric in ["HR(10)", "NDCG(10)"]:
            if metric in merged.columns:
                safe_metric = metric.replace("(", "").replace(")", "")
                scatter_plot(
                    merged,
                    "effective_item_update_F",
                    metric,
                    f"{metric} vs Effective Item Update",
                    out_dir / f"08_{safe_metric}_vs_effective_update.png",
                )
                scatter_plot(
                    merged,
                    "cos_delta_A",
                    metric,
                    f"{metric} vs Cosine Similarity of Delta A",
                    out_dir / f"09_{safe_metric}_vs_cos_delta_A.png",
                )

    return merged


# =========================================================
# Comparison figures across multiple logs
# =========================================================

def make_comparison_figures(all_ab_df: pd.DataFrame, all_merged_df: pd.DataFrame, out_dir: Path):
    """Draw comparison figures across selected logs."""
    out_dir.mkdir(parents=True, exist_ok=True)

    comparison_line_plot(
        all_ab_df,
        "delta_A_F",
        "Comparison: ||Delta A||_F",
        "Frobenius norm",
        out_dir / "compare_01_delta_A_F.png",
    )
    comparison_line_plot(
        all_ab_df,
        "delta_B_F",
        "Comparison: ||Delta B||_F",
        "Frobenius norm",
        out_dir / "compare_02_delta_B_F.png",
    )
    comparison_line_plot(
        all_ab_df,
        "norm_delta_A",
        "Comparison: Normalized ||Delta A||_F",
        "Normalized Frobenius norm",
        out_dir / "compare_03_norm_delta_A.png",
    )
    comparison_line_plot(
        all_ab_df,
        "norm_delta_B",
        "Comparison: Normalized ||Delta B||_F",
        "Normalized Frobenius norm",
        out_dir / "compare_04_norm_delta_B.png",
    )
    comparison_line_plot(
        all_ab_df,
        "cos_delta_A",
        "Comparison: cos(Delta A_t, Delta A_{t-1})",
        "Cosine similarity",
        out_dir / "compare_05_cos_delta_A.png",
    )
    comparison_line_plot(
        all_ab_df,
        "cos_delta_B",
        "Comparison: cos(Delta B_t, Delta B_{t-1})",
        "Cosine similarity",
        out_dir / "compare_06_cos_delta_B.png",
    )
    comparison_line_plot(
        all_ab_df,
        "effective_item_update_F",
        "Comparison: Effective Item Embedding Update",
        "Frobenius norm",
        out_dir / "compare_07_effective_item_update_F.png",
    )
    comparison_line_plot(
        all_ab_df,
        "effective_update_change",
        "Comparison: Effective Update Stability",
        "Absolute change",
        out_dir / "compare_08_effective_update_change.png",
    )

    if all_merged_df is not None and not all_merged_df.empty:
        comparison_line_plot(
            all_merged_df,
            "HR(10)",
            "Comparison: HR@10",
            "HR@10",
            out_dir / "compare_09_HR10.png",
        )
        comparison_line_plot(
            all_merged_df,
            "NDCG(10)",
            "Comparison: NDCG@10",
            "NDCG@10",
            out_dir / "compare_10_NDCG10.png",
        )
        comparison_line_plot(
            all_merged_df,
            "logloss",
            "Comparison: logloss",
            "logloss",
            out_dir / "compare_11_logloss.png",
        )


# =========================================================
# Main
# =========================================================

def main():
    if len(SELECTED_LOG_FILES) == 0:
        raise ValueError(
            "SELECTED_LOG_FILES is empty. Please add at least one log file name, "
            "for example: 'baseline_1000/fedncf_lora.txt'."
        )

    print("=" * 70)
    print("Selected log files:")
    for f in SELECTED_LOG_FILES:
        print("-", LOG_DIR / f)
    print("=" * 70)

    all_ab_list = []
    all_metric_list = []
    all_merged_list = []
    summary_list = []

    for log_relative in SELECTED_LOG_FILES:
        log_path = LOG_DIR / log_relative
        exp_name = safe_exp_name(log_relative)

        if not log_path.exists():
            print(f"[Skip] File not found: {log_path}")
            continue

        print(f"\nProcessing: {log_path}")
        print(f"Experiment name: {exp_name}")

        exp_fig_dir = AB_FIG_DIR / exp_name
        exp_csv_dir = AB_CSV_DIR / exp_name
        exp_fig_dir.mkdir(parents=True, exist_ok=True)
        exp_csv_dir.mkdir(parents=True, exist_ok=True)

        ab_df, metric_df = parse_log(log_path)
        ab_df = add_summary_columns(ab_df)

        ab_df.insert(0, "experiment", exp_name)
        metric_df.insert(0, "experiment", exp_name) if not metric_df.empty else None

        # For single-experiment plotting, remove experiment column only if not needed.
        merged_df = make_single_experiment_figures(ab_df, metric_df, exp_fig_dir)

        summary_df = summarize(ab_df, merged_df if not merged_df.empty else None)
        summary_df.insert(0, "experiment", exp_name)

        ab_csv = exp_csv_dir / "ab_convergence_metrics.csv"
        metric_csv = exp_csv_dir / "eval_metrics.csv"
        merged_csv = exp_csv_dir / "ab_plus_eval_metrics.csv"
        summary_csv = exp_csv_dir / "ab_summary.csv"

        ab_df.to_csv(ab_csv, index=False)
        metric_df.to_csv(metric_csv, index=False)
        if not merged_df.empty:
            merged_df.to_csv(merged_csv, index=False)
        summary_df.to_csv(summary_csv)

        all_ab_list.append(ab_df)
        if not metric_df.empty:
            all_metric_list.append(metric_df)
        if not merged_df.empty:
            all_merged_list.append(merged_df)
        summary_list.append(summary_df.reset_index().rename(columns={"index": "metric"}))

        print(f"Saved CSV to: {exp_csv_dir}")
        print(f"Saved figures to: {exp_fig_dir}")
        print(f"AB rows: {len(ab_df)} | Metric rows: {len(metric_df)}")

    if len(all_ab_list) == 0:
        raise ValueError("No valid log files were processed. Please check LOG_DIR and SELECTED_LOG_FILES.")

    all_ab_df = pd.concat(all_ab_list, ignore_index=True)
    all_metric_df = pd.concat(all_metric_list, ignore_index=True) if all_metric_list else pd.DataFrame()
    all_merged_df = pd.concat(all_merged_list, ignore_index=True) if all_merged_list else pd.DataFrame()
    all_summary_df = pd.concat(summary_list, ignore_index=True) if summary_list else pd.DataFrame()

    comparison_csv_dir = AB_CSV_DIR / "comparison"
    comparison_fig_dir = AB_FIG_DIR / "comparison"
    comparison_csv_dir.mkdir(parents=True, exist_ok=True)
    comparison_fig_dir.mkdir(parents=True, exist_ok=True)

    all_ab_df.to_csv(comparison_csv_dir / "all_ab_convergence_metrics.csv", index=False)
    all_metric_df.to_csv(comparison_csv_dir / "all_eval_metrics.csv", index=False)
    all_merged_df.to_csv(comparison_csv_dir / "all_ab_plus_eval_metrics.csv", index=False)
    all_summary_df.to_csv(comparison_csv_dir / "all_ab_summary.csv", index=False)

    make_comparison_figures(all_ab_df, all_merged_df, comparison_fig_dir)

    print("\nDone.")
    print(f"All comparison CSVs saved to: {comparison_csv_dir}")
    print(f"All comparison figures saved to: {comparison_fig_dir}")


if __name__ == "__main__":
    main()
