import re
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parent
LOG_DIR = REPO_ROOT / "log"
FIG_DIR = REPO_ROOT / "figures"

# ===== one input file only =====
INPUT_LOG_FILE = "baseline_1000/LoRA.txt"

# metrics
HR_METRIC = "HR(10)"
NDCG_METRIC = "NDCG(10)"

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 15,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})


def parse_log(path: Path):
    train_turns, train_losses = [], []
    current_turn = None
    pending_eval_turn = None
    eval_metrics = {}

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m_turn = re.search(r"Train Turn\s+(\d+)", line)
            if m_turn:
                current_turn = int(m_turn.group(1))

            m_loss = re.search(r"Clients average loss:\s*(?:tensor\()?([0-9]*\.?[0-9]+)", line)
            if m_loss and current_turn is not None:
                train_turns.append(current_turn)
                train_losses.append(float(m_loss.group(1)))

            m_eval = re.search(r"Eval @ Turn\s+(\d+)", line)
            if m_eval:
                pending_eval_turn = int(m_eval.group(1))

            if "[Metrics]" in line:
                turn = pending_eval_turn if pending_eval_turn is not None else current_turn
                if turn is not None:
                    pairs = re.findall(r"([A-Za-z0-9_()]+)\s*:\s*([0-9]*\.?[0-9]+)", line)
                    for metric_name, metric_val in pairs:
                        if metric_name not in eval_metrics:
                            eval_metrics[metric_name] = {"turns": [], "values": []}
                        eval_metrics[metric_name]["turns"].append(turn)
                        eval_metrics[metric_name]["values"].append(float(metric_val))
                pending_eval_turn = None

    train_turns = np.array(train_turns, dtype=int)
    train_losses = np.array(train_losses, dtype=float)

    for k in eval_metrics:
        t = np.array(eval_metrics[k]["turns"], dtype=int)
        v = np.array(eval_metrics[k]["values"], dtype=float)
        n = min(len(t), len(v))
        eval_metrics[k]["turns"] = t[:n]
        eval_metrics[k]["values"] = v[:n]

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
    plt.figure(figsize=(4.5, 3.5))

    for name, d in data.items():
        x = d.get(x_key, np.array([]))
        y = d.get(y_key, np.array([]))

        if len(x) == 0 or len(y) == 0:
            continue

        m = min(len(x), len(y))

        plt.plot(
            x[:m],
            y[:m],
            linewidth=2.5,
            label=name,
        )

    plt.xlabel("Communication Round", fontsize=16)
    plt.ylabel(y_label, fontsize=16)

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
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def main():
    log_path = LOG_DIR / INPUT_LOG_FILE
    if not log_path.exists():
        raise FileNotFoundError(f"File not found: {log_path}")

    d = parse_log(log_path)

    # Loss (smoothed + sampled)
    ys, off = smooth(d["train_losses"], w=9)
    xs = d["train_turns"][off:off + len(ys)] if len(ys) > 0 else np.array([])
    loss_x, loss_y = sample_every_n_rounds(xs, ys, n=10)

    # HR / NDCG
    hr_x = hr_y = np.array([])
    ndcg_x = ndcg_y = np.array([])

    if HR_METRIC in d["eval_metrics"]:
        hr_x = d["eval_metrics"][HR_METRIC]["turns"]
        hr_y = d["eval_metrics"][HR_METRIC]["values"]
    else:
        print(f"[Warning] {HR_METRIC} not found in {log_path.name}")

    if NDCG_METRIC in d["eval_metrics"]:
        ndcg_x = d["eval_metrics"][NDCG_METRIC]["turns"]
        ndcg_y = d["eval_metrics"][NDCG_METRIC]["values"]
    else:
        print(f"[Warning] {NDCG_METRIC} not found in {log_path.name}")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    out_loss = FIG_DIR / f"{ts}_{log_path.stem}_loss.pdf"
    out_hr_ndcg = FIG_DIR / f"{ts}_{log_path.stem}_hr_ndcg.pdf"

    # Figure 1: Loss
    loss_data = {
        "Loss": {
            "x": loss_x,
            "y": loss_y,
        }
    }
    plot_metric(
        data=loss_data,
        x_key="x",
        y_key="y",
        title="Loss",
        y_label="Loss",
        out_path=out_loss,
    )

    # Figure 2: HR + NDCG
    metric_data = {
        HR_METRIC: {
            "x": hr_x,
            "y": hr_y,
        },
        NDCG_METRIC: {
            "x": ndcg_x,
            "y": ndcg_y,
        },
    }
    plot_metric(
        data=metric_data,
        x_key="x",
        y_key="y",
        title="HR and NDCG",
        y_label="Value",
        out_path=out_hr_ndcg,
    )

    print(f"Saved: {out_loss}")
    print(f"Saved: {out_hr_ndcg}")


if __name__ == "__main__":
    main()