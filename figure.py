import os
import re
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parent
LOG_DIR = REPO_ROOT / "log"
FIG_DIR = REPO_ROOT / "figures"

# Set log file names here (1 to 10 files)
SELECTED_LOG_FILES = [


    # **************************************baseline********************************
    


    # "baseline_1000/FedNCF-Full.txt",
    # "baseline_1000/FedNCF-LoRA.txt",
    # "baseline_1000/MofiLoRA.txt",
    # "baseline_1000/FedNCF-Full-Mom.txt",

    "baseline_2000/FedNCF-Full.txt",
    "baseline_2000/FedNCF-LoRA.txt",
    "baseline_2000/MofiLoRA.txt",
    "baseline_2000/FedNCF-FixedB.txt",

    # "baseline_500/fedncf_lora.txt",
    # "baseline_500/fedncf_base.txt",

    # "baseline_200/fedncf_lora.txt",
    # "baseline_200/fedncf_base.txt",

    # **************************************EMA Finetune 1000********************************

    # "finetune_1000/ema/ema_0.9_5.0.txt",
    # "finetune_1000/ema/ema_0.9_10.0.txt",
    # "finetune_1000/ema/ema_0.9_20.0.txt",

    # "finetune_1000/ema/ema_0.8_10.0.txt",
    # "finetune_1000/ema/ema_0.9_10.0.txt",
    # "finetune_1000/ema/ema_0.99_10.0.txt",

    # "finetune_1000/ema/ema_0.8_5.0.txt",
    # "finetune_1000/ema/ema_0.8_10.0.txt",
    # "finetune_1000/ema/ema_0.8_20.0.txt",
    
    # "finetune_1000/ema/ema_0.5_10.0.txt",
    # "finetune_1000/ema/ema_0.7_10.0.txt",
    # "finetune_1000/ema/ema_0.95_10.0.txt",
    # "finetune_1000/ema/ema_0.97_10.0.txt",

    # **************************************Heavyball Finetune 1000********************************
    # "finetune_1000/heavyball/beta=0.9, ηs=0.5.txt",
    # "finetune_1000/heavyball/beta=0.9, ηs=1.0.txt",
    # "finetune_1000/heavyball/beta=0.9, ηs=2.0.txt",
    
    # "finetune_1000/heavyball/beta=0.8, ηs=1.0.txt",
    # "finetune_1000/heavyball/beta=0.9, ηs=1.0.txt",
    # "finetune_1000/heavyball/beta=0.99, ηs=1.0.txt",

    # "finetune_1000/heavyball/heavyball_0.99_2.0.txt",

    # **************************************AB test 1000********************************
    # "AB_test_1000/LoRA-FixedA.txt",
    # "AB_test_1000/LoRA-FixedB.txt",

    # "AB_test_1000/LoRA-MomA.txt",
    # "AB_test_1000/LoRA-MomB.txt",

    # "AB_test_1000/LoRA-MomB-FixedA.txt",
    # "AB_test_1000/LoRA-MomA-FixedB.txt",

    # "AB_test_1000/LoRA-FixedB.txt",
    # "AB_test_1000/LoRA-MomA.txt",
    # "AB_test_1000/LoRA-MomA-FixedB.txt",

    # **************************************AB test********************************
    # "AB_test/fedncf_fixedA.txt",
    # "AB_test/fedncf_fixedB.txt",

    # "AB_test/fedncf_momentumA.txt",
    # "AB_test/fedncf_momentumB.txt",

    # "AB_test/fedncf_momentumB_fixedA.txt",
    # "AB_test/fedncf_momentumA_fixedB.txt",
    
    # "AB_test/fedncf_fixedB.txt",
    # "AB_test/fedncf_momentumA.txt",
    # "AB_test/fedncf_momentumA_fixedB.txt",



    # ********************************Finetune Best Result*****************************
    # "finetune/heavyball/Heavy_ball_0.99_2.0.txt",
    # "finetune/ema/EMA_normalization_0.8_20.0.txt",
    # "finetune/adam/Adam_0.005_0.99_0.99.txt",

    # ********************************Adam*****************************
    # "finetune/adam/Adam_0.5_0.9_0.99.txt",
    # "finetune/adam/Adam_0.1_0.9_0.999.txt",
    # "finetune/adam/Adam_0.05_0.9_0.999.txt",
    # "finetune/adam/Adam_0.1_0.95_0.999.txt",
    # "finetune/adam/Adam_0.05_0.95_0.999.txt", #0.05>0.1 0.95>0.9 
    # "finetune/adam/Adam_0.05_0.95_0.99.txt",
    
    # "finetune/adam/Adam_0.05_0.99_0.9.txt", # 0.05>0.1 0.99>0.95>0.9 0.9==0.99


    # "finetune/adam/Adam_0.05_0.99_0.99.txt",
    # "finetune/adam/Adam_0.01_0.99_0.99.txt",
    # "finetune/adam/Adam_0.005_0.99_0.99.txt", #best

    # ********************************Heavyball*****************************
    # "finetune/heavyball/Heavy_ball_0.9_0.5.txt",
    # "finetune/heavyball/Heavy_ball_0.9_1.0.txt",
    # "finetune/heavyball/Heavy_ball_0.9_2.0.txt",
    # "finetune/heavyball/Heavy_ball_0.9_3.0.txt",
    # "finetune/heavyball/Heavy_ball_0.99_0.5.txt",
    # "finetune/heavyball/Heavy_ball_0.99_1.0.txt",
    # "finetune/heavyball/Heavy_ball_0.99_2.0.txt", #best
    # "finetune/heavyball/Heavy_ball_0.99_3.0.txt", 

    # "finetune/heavyball/Heavy_ball_0.99_0.5.txt",
    # "finetune/heavyball/Heavy_ball_0.99_1.0.txt",
    # "finetune/heavyball/Heavy_ball_0.99_2.0.txt",
    # "finetune/heavyball/Heavy_ball_0.99_3.0.txt",
    # "finetune/heavyball/Heavy_ball_0.99_5.0.txt",

    # "finetune/heavyball/Heavy_ball_0.5_1.0.txt",
    # "finetune/heavyball/Heavy_ball_0.7_1.0.txt",
    # "finetune/heavyball/Heavy_ball_0.9_1.0.txt",
    # "finetune/heavyball/Heavy_ball_0.95_1.0.txt",
    # "finetune/heavyball/Heavy_ball_0.99_1.0.txt",

    # "finetune/heavyball/Heavy_ball_0.5_2.0.txt",
    # "finetune/heavyball/Heavy_ball_0.7_2.0.txt",
    # "finetune/heavyball/Heavy_ball_0.9_2.0.txt",
    # "finetune/heavyball/Heavy_ball_0.95_2.0.txt",
    # "finetune/heavyball/Heavy_ball_0.99_2.0.txt",
    # "finetune/heavyball/Heavy_ball_0.999_2.0.txt",

 
    # ********************************EMA Normalization*****************************
    # "finetune/ema/EMA_normalization_0.8_1.0.txt",
    # "finetune/ema/EMA_normalization_0.8_10.0.txt",
    # "finetune/ema/EMA_normalization_0.8_20.0.txt", #best

    # "finetune/ema/EMA_normalization_0.95_1.0.txt",
    # "finetune/ema/EMA_normalization_0.95_10.0.txt",
    # "finetune/ema/EMA_normalization_0.95_20.0.txt", 

    # "finetune/ema/EMA_normalization_0.9_1.0.txt",
    # "finetune/ema/EMA_normalization_0.9_10.0.txt",
    # "finetune/ema/EMA_normalization_0.9_20.0.txt",
    # "finetune/ema/EMA_normalization_0.8_1.0.txt",
    # "finetune/ema/EMA_normalization_0.8_10.0.txt",
    # "finetune/ema/EMA_normalization_0.8_20.0.txt",
    
    # "finetune/ema/EMA_normalization_0.8_1.0.txt",
    # "finetune/ema/EMA_normalization_0.8_10.0.txt",
    # "finetune/ema/EMA_normalization_0.8_20.0.txt",
    # "finetune/ema/EMA_normalization_0.8_30.0.txt",
    # "finetune/ema/EMA_normalization_0.8_50.0.txt",

    # "finetune/ema/EMA_normalization_0.95_20.0.txt",
    # "finetune/ema/EMA_normalization_0.9_20.0.txt",
    # "finetune/ema/EMA_normalization_0.8_20.0.txt",
    # "finetune/ema/EMA_normalization_0.7_20.0.txt",
    # "finetune/ema/EMA_normalization_0.5_20.0.txt",
    # "finetune/ema/EMA_normalization_0.1_20.0.txt",

    # "finetune/ema/EMA_normalization_0.7_20.0.txt", 
    # "finetune/ema/abandon/EMA_normalization_0.7_10.0.txt", 
    # "finetune/ema/abandon/EMA_normalization_0.5_1.0.txt", 
    
    # "finetune/ema/abandon/EMA_normalization_0.5_50.0.txt",
    # "finetune/ema/abandon/EMA_normalization_0.1_1.0.txt", 
    # "finetune/ema/abandon/EMA_normalization_0.1_50.0.txt",



]

# Select which eval metrics to plot (from logs [Metrics] line)
# Supported examples:
# "logloss", "MRR",
# "NDCG(5)", "HR(5)",
# "NDCG(10)", "HR(10)",
# "NDCG(20)", "HR(20)",
# "NDCG(50)", "HR(50)"
SELECTED_METRICS = [
    # "HR(20)",
    # "NDCG(20)",
    # "MRR",
    # "logloss",
    "HR(10)",
    "NDCG(10)",

]

# All known metric options (for validation/help)
AVAILABLE_METRICS = [
    "logloss", "MRR",
    "NDCG(5)", "HR(5)",
    "NDCG(10)", "HR(10)",
    "NDCG(20)", "HR(20)",
    "NDCG(50)", "HR(50)",
]


def parse_log(path: Path):
    train_turns, train_losses = [], []
    current_turn = None
    pending_eval_turn = None

    # metric_name -> {"turns": [...], "values": [...]}
    eval_metrics = {}

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            # Train turn
            m_turn = re.search(r"Train Turn\s+(\d+)", line)
            if m_turn:
                current_turn = int(m_turn.group(1))

            # Loss: supports "0.123" and "tensor(0.123)"
            m_loss = re.search(r"Clients average loss:\s*(?:tensor\()?([0-9]*\.?[0-9]+)", line)
            if m_loss and current_turn is not None:
                train_turns.append(current_turn)
                train_losses.append(float(m_loss.group(1)))

            # Eval turn
            m_eval = re.search(r"Eval @ Turn\s+(\d+)", line)
            if m_eval:
                pending_eval_turn = int(m_eval.group(1))

            # Metrics line (generic parser)
            if "[Metrics]" in line:
                turn = pending_eval_turn if pending_eval_turn is not None else current_turn
                if turn is not None:
                    # parse pairs like "NDCG(10): 0.088666" or "MRR: 0.083082"
                    pairs = re.findall(r"([A-Za-z0-9_()]+)\s*:\s*([0-9]*\.?[0-9]+)", line)
                    for metric_name, metric_val in pairs:
                        if metric_name not in eval_metrics:
                            eval_metrics[metric_name] = {"turns": [], "values": []}
                        eval_metrics[metric_name]["turns"].append(turn)
                        eval_metrics[metric_name]["values"].append(float(metric_val))
                pending_eval_turn = None

    # convert arrays
    train_turns = np.array(train_turns, dtype=int)
    train_losses = np.array(train_losses, dtype=float)

    for k in eval_metrics:
        turns = np.array(eval_metrics[k]["turns"], dtype=int)
        vals = np.array(eval_metrics[k]["values"], dtype=float)
        n = min(len(turns), len(vals))
        eval_metrics[k]["turns"] = turns[:n]
        eval_metrics[k]["values"] = vals[:n]

    return {
        "train_turns": train_turns,
        "train_losses": train_losses,
        "eval_metrics": eval_metrics,
    }


def smooth(y, w=9):
    if len(y) == 0:
        return np.array([]), 0
    w = max(1, min(w, len(y)))
    kernel = np.ones(w) / w
    ys = np.convolve(y, kernel, mode="valid")
    return ys, w // 2


def sample_every_n_rounds(x, y, n=10):
    if len(x) == 0 or len(y) == 0:
        return np.array([]), np.array([])
    m = min(len(x), len(y))
    x = np.asarray(x[:m])
    y = np.asarray(y[:m])
    mask = (x % n == 0)
    return x[mask], y[mask]


def plot_metric(data, x_key, y_key, title, y_label, out_path):
    plt.figure(figsize=(5, 5))
    for name, d in data.items():
        x = d.get(x_key, np.array([]))
        y = d.get(y_key, np.array([]))
        if len(x) == 0 or len(y) == 0:
            continue
        m = min(len(x), len(y))
        plt.plot(x[:m], y[:m], linewidth=2, marker="o", markersize=3, label=name)

    # plt.title(title)
    plt.xlabel("round", fontsize=15)
    plt.ylabel(y_label, fontsize=15)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=15)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

# def plot_metric(data, x_key, y_key, title, y_label, out_path):
#     plt.figure(figsize=(8.8, 5.2))
#     for name, d in data.items():
#         x = d.get(x_key, np.array([]))
#         y = d.get(y_key, np.array([]))
#         if len(x) == 0 or len(y) == 0:
#             continue
#         m = min(len(x), len(y))
#         plt.plot(x[:m], y[:m], linewidth=2, marker="o", markersize=3, label=name)

#     plt.title(title)
#     plt.xlabel("round")
#     plt.ylabel(y_label)
#     plt.grid(alpha=0.3)
#     plt.legend(fontsize=9)
#     plt.tight_layout()
#     plt.savefig(out_path, dpi=300, bbox_inches="tight")
#     plt.close()


def _safe_name(metric: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", metric).strip("_").lower()


def main():
    files = SELECTED_LOG_FILES

    if not (1 <= len(files) <= 10):
        raise ValueError("Please provide 1 to 10 file names in SELECTED_LOG_FILES.")

    if len(files) > 5:
        print(f"[Note] You selected {len(files)} files. Plot may look crowded (recommended <= 5).")

    # validate selected metrics
    unknown = [m for m in SELECTED_METRICS if m not in AVAILABLE_METRICS]
    if unknown:
        raise ValueError(f"Unknown metrics in SELECTED_METRICS: {unknown}\nAvailable: {AVAILABLE_METRICS}")

    selected_paths = []
    for fname in files:
        p = LOG_DIR / fname
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")
        selected_paths.append(p)

    data = {p.stem: parse_log(p) for p in selected_paths}

    # Smooth loss per file
    for name, d in data.items():
        ys, off = smooth(d["train_losses"], w=9)
        xs = d["train_turns"][off:off + len(ys)] if len(ys) > 0 else np.array([])
        xs_10, ys_10 = sample_every_n_rounds(xs, ys, n=10)
        d["train_turns_smooth"] = xs_10
        d["train_losses_smooth"] = ys_10

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1) Loss in one figure
    plot_metric(
        data={k: {"train_turns": v["train_turns_smooth"], "train_losses": v["train_losses_smooth"]} for k, v in data.items()},
        x_key="train_turns",
        y_key="train_losses",
        title="Training Loss",
        y_label="Loss",
        out_path=FIG_DIR / f"{ts}_loss_comparison.png",
    )

    # 2) Selected eval metrics (one figure per metric)
    for metric in SELECTED_METRICS:
        metric_plot_data = {}
        for run_name, d in data.items():
            metric_dict = d.get("eval_metrics", {})
            if metric in metric_dict:
                metric_plot_data[run_name] = {
                    "eval_turns": metric_dict[metric]["turns"],
                    "metric_values": metric_dict[metric]["values"],
                }

        if len(metric_plot_data) == 0:
            print(f"[Warning] Metric '{metric}' not found in selected logs. Skip.")
            continue

        safe_metric = _safe_name(metric)
        plot_metric(
            data=metric_plot_data,
            x_key="eval_turns",
            y_key="metric_values",
            title=f"{metric}",
            y_label=metric,
            out_path=FIG_DIR / f"{ts}_{safe_metric}_comparison.png",
        )

    print(f"Saved to: {FIG_DIR}")
    print(f"- {ts}_loss_comparison.png")
    for metric in SELECTED_METRICS:
        print(f"- {ts}_{_safe_name(metric)}_comparison.png")


if __name__ == "__main__":
    main()