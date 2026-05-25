import re
from pathlib import Path
from statistics import mean, median

# ===== Config =====
LOG_FILE = Path("log/baseline_1000/momentumA_fixedB.txt")
# LOG_FILE = Path("log/baseline_1000/fedncf_base.txt")
# LOG_FILE = Path("log/baseline_1000/fedncf_lora.txt")
# LOG_FILE = Path("log/finetune/heavyball/Heavy_ball_0.99_2.0.txt")
START_ROUND = 20
END_ROUND = 999

WINDOW_LEN = 50          # loss: 50 consecutive rounds
METRIC_EVAL_WINDOW = 5   # HR/NDCG: 5 eval points ≈ 50 rounds if eval every 10 rounds
THRESHOLD = 0.95
# ==================

TURN_PATTERN = re.compile(r"\[Time\]\s+turn=(?P<turn>\d+)")
LOCAL_PATTERN = re.compile(r"local_train_time=(?P<local>[0-9]*\.?[0-9]+)s")
AGG_PATTERN = re.compile(r"aggregation_time=(?P<agg>[0-9]*\.?[0-9]+)s")
ROUND_PATTERN = re.compile(r"round_time=(?P<round>[0-9]*\.?[0-9]+)s")
PRETRAIN_PATTERN = re.compile(r"\[Time\]\s+pretrain_time=(?P<pre>[0-9]*\.?[0-9]+)s")

TRAIN_TURN_PATTERN = re.compile(r"\*+\s*Train Turn\s+(?P<turn>\d+)\s*\*+")
LOSS_PATTERN = re.compile(r"Clients average loss:\s*(?P<loss>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")

EVAL_TURN_PATTERN = re.compile(r"\*+\s*Eval\s*@\s*Turn\s+(?P<turn>\d+)\s*\*+")
METRIC_LINE_PATTERN = re.compile(r"\[Metrics\]\s+(?P<body>.*)")
METRIC_KV_PATTERN = re.compile(r"([A-Za-z0-9_@()]+):\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")


def _pick_metric(d, keys):
    for k in keys:
        if k in d:
            return d[k]
    return None


def parse_log(log_path: Path):
    rows = []
    pretrain_time = None
    loss_by_turn = {}
    metric_by_turn = {}

    current_turn = None
    current_eval_turn = None

    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m_pre = PRETRAIN_PATTERN.search(line)
            if m_pre:
                pretrain_time = float(m_pre.group("pre"))

            m_train = TRAIN_TURN_PATTERN.search(line)
            if m_train:
                current_turn = int(m_train.group("turn"))

            m_eval = EVAL_TURN_PATTERN.search(line)
            if m_eval:
                current_eval_turn = int(m_eval.group("turn"))

            m_loss = LOSS_PATTERN.search(line)
            if m_loss and current_turn is not None:
                loss_by_turn[current_turn] = float(m_loss.group("loss"))

            m_metrics = METRIC_LINE_PATTERN.search(line)
            if m_metrics:
                kv = {
                    k: float(v)
                    for k, v in METRIC_KV_PATTERN.findall(m_metrics.group("body"))
                }

                t = current_eval_turn if current_eval_turn is not None else current_turn
                if t is not None:
                    metric_by_turn[t] = {
                        "hr10": _pick_metric(kv, ["HR(10)", "HR@10"]),
                        "ndcg10": _pick_metric(kv, ["NDCG(10)", "NDCG@10"]),
                    }

            m_turn = TURN_PATTERN.search(line)
            if not m_turn:
                continue

            turn = int(m_turn.group("turn"))
            current_turn = turn

            m_local = LOCAL_PATTERN.search(line)
            if not m_local:
                continue

            m_agg = AGG_PATTERN.search(line)
            m_round = ROUND_PATTERN.search(line)

            rows.append({
                "turn": turn,
                "local_train_time": float(m_local.group("local")),
                "aggregation_time": float(m_agg.group("agg")) if m_agg else None,
                "round_time": float(m_round.group("round")) if m_round else None,
                "client_avg_loss": loss_by_turn.get(turn),
            })

    return rows, pretrain_time, metric_by_turn


def progress_score(value, v_min, v_max, kind):
    """
    Loss:
        (L_max - L_t) / (L_max - L_min)

    Metric:
        (M_t - M_min) / (M_max - M_min)
    """
    denom = v_max - v_min
    if abs(denom) < 1e-12:
        return None

    if kind == "loss":
        return (v_max - value) / denom
    else:
        return (value - v_min) / denom


def report_loss_convergence(loss_pairs):
    """
    Loss convergence:
    for all t in [T, T+49],
    (L_max - L_t) / (L_max - L_min) >= 0.95
    """
    print("-" * 50)

    if not loss_pairs:
        print("Clients average loss convergence: N/A")
        return

    vals = [v for _, v in loss_pairs if v is not None]
    if not vals:
        print("Clients average loss convergence: N/A")
        return

    l_min = min(vals)
    l_max = max(vals)

    print(f"Clients average loss min: {l_min:.8f}, max: {l_max:.8f}")
    print(f"Clients average loss window length: {WINDOW_LEN}")

    valid_starts = []

    for i in range(len(loss_pairs) - WINDOW_LEN + 1):
        window = loss_pairs[i:i + WINDOW_LEN]
        ok = True

        for _, loss in window:
            score = progress_score(loss, l_min, l_max, kind="loss")
            if score is None or score < THRESHOLD:
                ok = False
                break

        if ok:
            valid_starts.append(window[0][0])

    print(f"Clients average loss converged start rounds: {len(valid_starts)}")

    if valid_starts:
        first_T = valid_starts[0]
        print(f"Clients average loss first converged round T: {first_T}")
        print(f"Clients average loss rounds to converge from START_ROUND: {first_T - START_ROUND + 1}")
    else:
        print("Clients average loss first converged round T: N/A")


def report_metric_convergence(name, metric_pairs):
    """
    Metric convergence:
    for 5 consecutive eval points,
    (M_t - M_min) / (M_max - M_min) >= 0.95
    """
    print("-" * 50)

    if not metric_pairs:
        print(f"{name} convergence: N/A")
        return

    vals = [v for _, v in metric_pairs if v is not None]
    if not vals:
        print(f"{name} convergence: N/A")
        return

    m_min = min(vals)
    m_max = max(vals)

    print(f"{name} min: {m_min:.8f}, max: {m_max:.8f}")
    print(f"{name} eval-window length: {METRIC_EVAL_WINDOW}")

    valid_starts = []

    for i in range(len(metric_pairs) - METRIC_EVAL_WINDOW + 1):
        window = metric_pairs[i:i + METRIC_EVAL_WINDOW]
        ok = True

        for _, metric in window:
            score = progress_score(metric, m_min, m_max, kind="metric")
            if score is None or score < THRESHOLD:
                ok = False
                break

        if ok:
            valid_starts.append(window[0][0])

    print(f"{name} converged start rounds: {len(valid_starts)}")

    if valid_starts:
        first_T = valid_starts[0]
        print(f"{name} first converged round T: {first_T}")
        print(f"{name} rounds to converge from START_ROUND: {first_T - START_ROUND + 1}")
    else:
        print(f"{name} first converged round T: N/A")


def main():
    if not LOG_FILE.exists():
        raise FileNotFoundError(f"Log file not found: {LOG_FILE}")

    rows, pretrain_time, metric_by_turn = parse_log(LOG_FILE)

    selected = [
        r for r in rows
        if START_ROUND <= r["turn"] <= END_ROUND
    ]

    if not selected:
        print(f"No records found in round range [{START_ROUND}, {END_ROUND}].")
        return

    local_times = [r["local_train_time"] for r in selected]
    agg_times = [r["aggregation_time"] for r in selected if r["aggregation_time"] is not None]
    round_times = [r["round_time"] for r in selected if r["round_time"] is not None]

    print(f"File: {LOG_FILE}")
    print(f"Round range: [{START_ROUND}, {END_ROUND}]")
    print(f"Matched rounds: {len(selected)}")
    print("-" * 50)

    print(f"Average local_train_time: {mean(local_times):.4f}s")
    print(f"Median  local_train_time: {median(local_times):.4f}s")
    print(f"Accumulated local_train_time: {sum(local_times):.4f}s ({sum(local_times)/60:.2f} min)")

    if agg_times:
        print(f"Average aggregation_time: {mean(agg_times):.4f}s")
        print(f"Median  aggregation_time: {median(agg_times):.4f}s")
        print(f"Accumulated aggregation_time: {sum(agg_times):.4f}s ({sum(agg_times)/60:.2f} min)")

    if round_times:
        print(f"Accumulated round_time: {sum(round_times):.4f}s ({sum(round_times)/60:.2f} min)")

    if pretrain_time is not None:
        total_time = pretrain_time + sum(local_times)
        print(f"Pretrain time: {pretrain_time:.4f}s ({pretrain_time/60:.2f} min)")
        print(f"Total (pretrain + accumulated local_train_time): {total_time:.4f}s ({total_time/60:.2f} min)")

    loss_pairs = [
        (r["turn"], r["client_avg_loss"])
        for r in selected
        if r["client_avg_loss"] is not None
    ]

    hr_pairs = sorted([
        (t, m["hr10"])
        for t, m in metric_by_turn.items()
        if START_ROUND <= t <= END_ROUND and m.get("hr10") is not None
    ])

    ndcg_pairs = sorted([
        (t, m["ndcg10"])
        for t, m in metric_by_turn.items()
        if START_ROUND <= t <= END_ROUND and m.get("ndcg10") is not None
    ])

    report_loss_convergence(loss_pairs)
    report_metric_convergence("HR@10", hr_pairs)
    report_metric_convergence("NDCG@10", ndcg_pairs)


if __name__ == "__main__":
    main()