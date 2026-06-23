import argparse
from pathlib import Path
import pandas as pd

# ===== Editable defaults (change here in code) =====
DEFAULT_INPUT_CSV = Path("csv/Industrial_All_Result.csv")
DEFAULT_DATASET = "Industrial"               # e.g., "ML1M", "Industrial", "Software", or None
DEFAULT_OUTPUT_DIR = Path("csv/")
DEFAULT_OUTPUT_FILENAME = None          # e.g., "ML1M.csv"; if None -> "{dataset}.csv"
DEFAULT_COMPARE_SUFFIX = "_vs_LoRA"     # second table: "{dataset}{suffix}.csv"
DEFAULT_STD_SUFFIX = "_mean_std"        # third table: "{dataset}{suffix}.csv"
# ================================================

# Keep output row order fixed (8 rows)
METHOD_ORDER = [
    "Full",
    "LoRA",
    "LoRA_FixedA",
    "LoRA_FixedB",
    "LoRA_MomA",
    "LoRA_MomB",
    "LoRA_MomB_FixedA",
    "LoRA_MomA_FixedB",
]

# Higher is better
HIGHER_BETTER_COLS = {
    "HR@10", "NDCG@10", "HR@20", "NDCG@20", "HR@50", "NDCG@50"
}


def _is_accuracy_col(col: str) -> bool:
    return col.startswith("HR@") or col.startswith("NDCG@")


def _pct_change(val: float, base: float, higher_better: bool) -> float | None:
    if pd.isna(val) or pd.isna(base) or base == 0:
        return None
    if higher_better:
        # positive means improved
        return (val - base) / base * 100.0
    # lower is better -> positive means reduced
    return (base - val) / base * 100.0


def _format_pct(x: float | None) -> str:
    if x is None or pd.isna(x):
        return ""
    return f"{x:+.2f}%"


def _format_number(val: float, col: str) -> str:
    if pd.isna(val):
        return ""
    if "Time" in col:
        return f"{val:.2f}"   # time columns: 2 decimals
    return f"{val:.8f}"       # others: 4 decimals


def _format_mean_std(mean_val: float, std_val: float, col: str) -> str:
    if pd.isna(mean_val):
        return ""
    if pd.isna(std_val):
        std_val = 0.0

    # Accuracy columns shown in percentage
    if _is_accuracy_col(col):
        return f"{mean_val * 100:.2f} ± {std_val * 100:.2f}"

    # Time columns keep 2 decimals
    if "Time" in col:
        return f"{mean_val:.2f} ± {std_val:.2f}"

    # Other metrics
    return f"{mean_val:.8f} ± {std_val:.8f}"


def build_avg_table(
    input_csv: Path,
    dataset: str | None,
    output_dir: Path,
    output_filename: str | None = None,
) -> tuple[Path, Path, Path]:
    df = pd.read_csv(input_csv)

    # Parse "Method" like: ML1M_LoRA_MomA_FixedB_2024
    parsed = df["Method"].str.extract(r"^(?P<dataset>[^_]+)_(?P<variant>.+)_(?P<seed>\d+)$")
    if parsed.isna().any().any():
        bad = df.loc[parsed.isna().any(axis=1), "Method"].tolist()
        raise ValueError(f"Invalid Method format for rows: {bad}")

    df = pd.concat([df, parsed], axis=1)
    df["seed"] = df["seed"].astype(int)

    # If dataset not passed, infer from first row
    if dataset is None:
        dataset = df["dataset"].iloc[0]

    df = df[df["dataset"] == dataset].copy()
    if df.empty:
        raise ValueError(f"No rows found for dataset '{dataset}' in {input_csv}")

    # Replace '>900' -> 900, then convert to numeric
    metric_cols = [c for c in df.columns if c not in ["Method", "dataset", "variant", "seed"]]
    for c in metric_cols:
        df[c] = (
            df[c]
            .astype(str)
            .str.replace(">", "", regex=False)
            .replace("", pd.NA)
            .astype(float)
        )

    # Mean/std across seeds, keep method order
    mean_df = df.groupby("variant")[metric_cols].mean(numeric_only=True).reindex(METHOD_ORDER)
    std_df = df.groupby("variant")[metric_cols].std(ddof=1, numeric_only=True).reindex(METHOD_ORDER)

    # Drop methods not present in input
    keep_mask = ~mean_df.isna().all(axis=1)
    mean_df = mean_df[keep_mask]
    std_df = std_df[keep_mask]

    avg = mean_df.reset_index().rename(columns={"variant": "Method"})

    # Table 1: average table (time columns -> 2 decimals)
    time_cols = [c for c in metric_cols if "Time" in c]
    for c in time_cols:
        avg[c] = avg[c].round(2)

    output_dir.mkdir(parents=True, exist_ok=True)
    filename = output_filename if output_filename else f"{dataset}.csv"
    out_avg_path = output_dir / filename
    avg.to_csv(out_avg_path, index=False)

    # Table 2: value + percentage vs LoRA in same cell
    if "LoRA" not in set(avg["Method"]):
        raise ValueError("LoRA row not found after averaging; cannot build comparison table.")

    lora_row = avg.loc[avg["Method"] == "LoRA"].iloc[0]
    comp = avg.copy()
    comp[metric_cols] = comp[metric_cols].astype("object")

    for i, row in comp.iterrows():
        method = row["Method"]
        for c in metric_cols:
            val = avg.at[i, c]
            val_txt = _format_number(val, c)

            if method == "LoRA":
                comp.loc[i, c] = val_txt
                continue

            pct = _pct_change(val, lora_row[c], c in HIGHER_BETTER_COLS)
            pct_txt = _format_pct(pct)
            comp.loc[i, c] = f"{val_txt} ({pct_txt})" if pct_txt else val_txt

    out_comp_path = output_dir / f"{dataset}{DEFAULT_COMPARE_SUFFIX}.csv"
    comp.to_csv(out_comp_path, index=False)

    # Table 3: mean ± std over seeds (accuracy as %)
    std_table = pd.DataFrame({"Method": mean_df.index})
    for c in metric_cols:
        display_col = f"{c} (%)" if _is_accuracy_col(c) else c
        std_table[display_col] = [
            _format_mean_std(m, s, c)
            for m, s in zip(mean_df[c].tolist(), std_df[c].tolist())
        ]
    std_table = std_table.reset_index(drop=True)

    out_std_path = output_dir / f"{dataset}{DEFAULT_STD_SUFFIX}.csv"
    std_table.to_csv(out_std_path, index=False)

    return out_avg_path, out_comp_path, out_std_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help="Path to summary CSV",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        help="Dataset prefix in Method column (e.g., ML1M, Industrial, Software). If omitted, inferred.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save output CSV",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=DEFAULT_OUTPUT_FILENAME,
        help="Output file name (e.g., ML1M.csv). If omitted, uses {dataset}.csv.",
    )
    args = parser.parse_args()

    out_avg, out_comp, out_std = build_avg_table(args.input, args.dataset, args.output_dir, args.output_file)
    print(f"Saved: {out_avg}")
    print(f"Saved: {out_comp}")
    print(f"Saved: {out_std}")


if __name__ == "__main__":
    main()