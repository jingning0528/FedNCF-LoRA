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

# Set log file names here (1 to 5 files)
SELECTED_LOG_FILES = [
    
    # "LoRA_rank2.txt",
    # "LoRA_rank4.txt",
    # "LoRA_rank8.txt",
    # "LoRA_rank16.txt",

    "LoRA_rank4_accuracy.txt",
    # "FixedB.txt",
    # "FixedB_Momentum.txt",
    # "DeltaB.txt",
    # "DeltaB_Momentum_deltascale0.3.txt",
    # "PeriodicB.txt",

    "DeltaB_update1.txt",
    "DeltaB.txt",
    # "DeltaB_update20.txt",

    # "FixedB_Momentum_Warmup.txt",

    # "DeltaB_Momentum_deltascale0.1.txt",
    # "DeltaB_Momentum_deltascale0.3.txt",
    # "DeltaB_Momentum_deltascale0.5.txt",

    # "DeltaB_Momentum_delta5.txt",
    # "DeltaB_Momentum_deltascale0.3.txt",
    # "DeltaB_Momentum_delta20.txt",

    # "LoRA_rank4_1000.txt",

]


def parse_log(path: Path):
    train_turns, train_losses = [], []
    eval_turns, hr10, ndcg10 = [], [], []

    current_turn = None
    pending_eval_turn = None

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

            # Metrics line
            if "[Metrics]" in line:
                m_hr10 = re.search(r"HR\(10\):\s*([0-9]*\.?[0-9]+)", line)
                m_ndcg10 = re.search(r"NDCG\(10\):\s*([0-9]*\.?[0-9]+)", line)
                if m_hr10 and m_ndcg10:
                    turn = pending_eval_turn if pending_eval_turn is not None else current_turn
                    if turn is not None:
                        eval_turns.append(turn)
                        hr10.append(float(m_hr10.group(1)))
                        ndcg10.append(float(m_ndcg10.group(1)))
                    pending_eval_turn = None

    # Align eval arrays in case logs have extra metrics lines
    n = min(len(eval_turns), len(hr10), len(ndcg10))
    eval_turns = np.array(eval_turns[:n], dtype=int)
    hr10 = np.array(hr10[:n], dtype=float)
    ndcg10 = np.array(ndcg10[:n], dtype=float)

    # Train arrays
    train_turns = np.array(train_turns, dtype=int)
    train_losses = np.array(train_losses, dtype=float)

    return {
        "train_turns": train_turns,
        "train_losses": train_losses,
        "eval_turns": eval_turns,
        "hr10": hr10,
        "ndcg10": ndcg10,
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
    plt.figure(figsize=(8.8, 5.2))
    for name, d in data.items():
        x = d.get(x_key, np.array([]))
        y = d.get(y_key, np.array([]))
        if len(x) == 0 or len(y) == 0:
            continue
        m = min(len(x), len(y))
        plt.plot(x[:m], y[:m], linewidth=2, marker="o", markersize=3, label=name)

    plt.title(title)
    plt.xlabel("round")  # changed axis label
    plt.ylabel(y_label)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    # Use file names from code (no terminal input needed)
    files = SELECTED_LOG_FILES

    # allow up to 10 files
    if not (1 <= len(files) <= 10):
        raise ValueError("Please provide 1 to 10 file names in SELECTED_LOG_FILES.")

    # optional note: usually best readability is <= 5 lines per figure
    if len(files) > 5:
        print(f"[Note] You selected {len(files)} files. Plot may look crowded (recommended <= 5).")

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

        # keep one loss point every 10 rounds (like eval metrics frequency)
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
        title="Training Loss Comparison",
        y_label="BPR Loss",
        out_path=FIG_DIR / f"{ts}_loss_comparison.png",
    )

    # 2) HR@10 in one figure
    plot_metric(
        data=data,
        x_key="eval_turns",
        y_key="hr10",
        title="HR@10 Comparison",
        y_label="HR@10",
        out_path=FIG_DIR / f"{ts}_hr10_comparison.png",
    )

    # 3) NDCG@10 in one figure
    plot_metric(
        data=data,
        x_key="eval_turns",
        y_key="ndcg10",
        title="NDCG@10 Comparison",
        y_label="NDCG@10",
        out_path=FIG_DIR / f"{ts}_ndcg10_comparison.png",
    )

    print(f"Saved to: {FIG_DIR}")
    print(f"- {ts}_loss_comparison.png")
    print(f"- {ts}_hr10_comparison.png")
    print(f"- {ts}_ndcg10_comparison.png")


if __name__ == "__main__":
    main()