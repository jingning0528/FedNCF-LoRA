import re
from pathlib import Path
from statistics import pstdev
import csv

# ===== Config =====
LOG_FILES = [
    Path("log/baseline_2000/FedNCF-Full.txt"),
    Path("log/baseline_2000/FedNCF-LoRA.txt"),
    Path("log/baseline_2000/MoFiLoRA.txt"),
]
START_ROUND = 20
END_ROUND = 999
CSV_OUT = Path("csv/baseline_1000_summary.csv")

# add these back
WINDOW_LEN_LOSS = 50
WINDOW_LEN_METRIC = 10
THRESHOLD = 0.95
LOSS_CONVERGE_THRESHOLD = 0.3
# ==================

TURN_PATTERN = re.compile(r"\[Time\]\s+turn=(?P<turn>\d+)")
LOCAL_PATTERN = re.compile(r"local_train_time=(?P<local>[0-9]*\.?[0-9]+)s")
AGG_PATTERN = re.compile(r"aggregation_time=(?P<agg>[0-9]*\.?[0-9]+)s")

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
    loss_by_turn = {}
    metric_by_turn = {}

    current_turn = None
    current_eval_turn = None

    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
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
                kv = {k: float(v) for k, v in METRIC_KV_PATTERN.findall(m_metrics.group("body"))}
                t = current_eval_turn if current_eval_turn is not None else current_turn
                if t is not None:
                    metric_by_turn[t] = {
                        "hr10": _pick_metric(kv, ["HR(10)", "HR@10"]),
                        "ndcg10": _pick_metric(kv, ["NDCG(10)", "NDCG@10"]),
                        "hr20": _pick_metric(kv, ["HR(20)", "HR@20"]),
                        "ndcg20": _pick_metric(kv, ["NDCG(20)", "NDCG@20"]),
                        "hr50": _pick_metric(kv, ["HR(50)", "HR@50"]),
                        "ndcg50": _pick_metric(kv, ["NDCG(50)", "NDCG@50"]),
                    }

            m_turn = TURN_PATTERN.search(line)
            if not m_turn:
                continue

            turn = int(m_turn.group("turn"))
            m_local = LOCAL_PATTERN.search(line)
            if not m_local:
                continue
            m_agg = AGG_PATTERN.search(line)

            rows.append({
                "turn": turn,
                "local_train_time": float(m_local.group("local")),
                "aggregation_time": float(m_agg.group("agg")) if m_agg else None,
                "client_avg_loss": loss_by_turn.get(turn),
            })

    return rows, metric_by_turn


def progress_score(value, v_min, v_max, kind):
    denom = v_max - v_min
    if abs(denom) < 1e-12:
        return None
    if kind == "loss":
        return (v_max - value) / denom
    return (value - v_min) / denom


def first_converged_round(pairs, window_len, kind):
    if len(pairs) < window_len:
        return None

    values = [v for _, v in pairs if v is not None]
    if not values:
        return None

    v_min, v_max = min(values), max(values)
    if abs(v_max - v_min) < 1e-12:
        return None

    for i in range(len(pairs) - window_len + 1):
        window = pairs[i:i + window_len]
        ok = True
        for _, v in window:
            s = progress_score(v, v_min, v_max, kind)
            if s is None or s < THRESHOLD:
                ok = False
                break
        if ok:
            return window[0][0]
    return None


def first_round_loss_below(pairs, threshold=0.3):
    """
    Loss convergence = first round where loss < threshold.
    """
    for t, v in pairs:
        if v is not None and v < threshold:
            return t
    return None


def first_round_loss_below_consecutive(pairs, threshold=0.3, window_len=100):
    """
    Loss convergence = first round where loss <= threshold continuously
    for at least `window_len` rounds.
    """
    if not pairs:
        return None

    pairs = sorted(pairs, key=lambda x: x[0])  # (turn, loss)

    run_start = None
    run_len = 0
    prev_turn = None

    for t, v in pairs:
        ok = (v is not None and v <= threshold)

        if ok:
            if run_start is None:
                run_start = t
                run_len = 1
            else:
                # require consecutive rounds
                if prev_turn is not None and t == prev_turn + 1:
                    run_len += 1
                else:
                    run_start = t
                    run_len = 1

            if run_len >= window_len:
                return run_start
        else:
            run_start = None
            run_len = 0

        prev_turn = t

    return None


def metric_stability_std(metric_pairs, start_round, end_round):
    vals = [v for t, v in metric_pairs if start_round <= t <= end_round and v is not None]
    if not vals:
        return None
    if len(vals) == 1:
        return 0.0
    return pstdev(vals)


def _fmt(x, nd=6):
    if x is None:
        return "N/A"
    if isinstance(x, int):
        return str(x)
    return f"{x:.{nd}f}"


def _method_name(log_file: Path):
    name = log_file.stem.lower()
    if "fedncf_base" in name:
        return "FedNCF Base"
    if "fedncf_lora" in name:
        return "FedNCF LoRA"
    if "momentuma_fixedb" in name or "momentuma" in name:
        return "MomentumA FixedB"
    return log_file.stem


def analyze_one_file(log_file: Path):
    if not log_file.exists():
        return {"Method": _method_name(log_file), "skip": f"not found: {log_file}"}

    rows, metric_by_turn = parse_log(log_file)
    selected = [r for r in rows if START_ROUND <= r["turn"] <= END_ROUND]

    if not selected:
        return {"Method": _method_name(log_file), "skip": f"no records in [{START_ROUND}, {END_ROUND}]"}

    local_sum = sum(r["local_train_time"] for r in selected)
    agg_sum = sum(r["aggregation_time"] for r in selected if r["aggregation_time"] is not None)

    loss_pairs = [(r["turn"], r["client_avg_loss"]) for r in selected if r["client_avg_loss"] is not None]
    hr_pairs = sorted(
        (t, m["hr10"]) for t, m in metric_by_turn.items()
        if START_ROUND <= t <= END_ROUND and m.get("hr10") is not None
    )
    ndcg_pairs = sorted(
        (t, m["ndcg10"]) for t, m in metric_by_turn.items()
        if START_ROUND <= t <= END_ROUND and m.get("ndcg10") is not None
    )

    # changed: loss must stay <= threshold continuously for WINDOW_LEN_LOSS rounds
    loss_T = first_round_loss_below_consecutive(
        loss_pairs,
        threshold=LOSS_CONVERGE_THRESHOLD,
        window_len=WINDOW_LEN_LOSS,
    )
    hr_T = first_converged_round(hr_pairs, WINDOW_LEN_METRIC, "metric")
    ndcg_T = first_converged_round(ndcg_pairs, WINDOW_LEN_METRIC, "metric")

    hr_last = hr_pairs[-1][1] if hr_pairs else None
    ndcg_last = ndcg_pairs[-1][1] if ndcg_pairs else None
    hr20_last = next((m["hr20"] for t, m in sorted(metric_by_turn.items(), reverse=True)
                      if START_ROUND <= t <= END_ROUND and m.get("hr20") is not None), None)
    ndcg20_last = next((m["ndcg20"] for t, m in sorted(metric_by_turn.items(), reverse=True)
                        if START_ROUND <= t <= END_ROUND and m.get("ndcg20") is not None), None)
    hr50_last = next((m["hr50"] for t, m in sorted(metric_by_turn.items(), reverse=True)
                      if START_ROUND <= t <= END_ROUND and m.get("hr50") is not None), None)
    ndcg50_last = next((m["ndcg50"] for t, m in sorted(metric_by_turn.items(), reverse=True)
                        if START_ROUND <= t <= END_ROUND and m.get("ndcg50") is not None), None)

    if loss_T is not None:
        loss_std_after_loss_conv = metric_stability_std(loss_pairs, loss_T, END_ROUND)
        hr_std_after_loss_conv = metric_stability_std(hr_pairs, loss_T, END_ROUND)
        ndcg_std_after_loss_conv = metric_stability_std(ndcg_pairs, loss_T, END_ROUND)
    else:
        loss_std_after_loss_conv = None
        hr_std_after_loss_conv = None
        ndcg_std_after_loss_conv = None

    return {
        "Method": _method_name(log_file),
        "HR@10": hr_last,
        "NDCG@10": ndcg_last,
        "HR@20": hr20_last,
        "NDCG@20": ndcg20_last,
        "HR@50": hr50_last,
        "NDCG@50": ndcg50_last,
        "Loss Convergence": loss_T,
        "HR Convergence": hr_T,
        "NDCG Convergence": ndcg_T,
        "Loss Stability (Std)": loss_std_after_loss_conv,
        "HR Stability (Std)": hr_std_after_loss_conv,
        "NDCG Stability (Std)": ndcg_std_after_loss_conv,
        "Local Training Time": local_sum,
        "Aggregation Time": agg_sum,
    }


def main():
    results = [analyze_one_file(p) for p in LOG_FILES]

    print("| Method | HR@10 | NDCG@10 | HR@20 | NDCG@20 | HR@50 | NDCG@50 | Loss Convergence | HR Convergence | NDCG Convergence | Loss Stability (Std) | HR Stability (Std) | NDCG Stability (Std) | Local Training Time | Aggregation Time |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    csv_rows = []

    for r in results:
        if "skip" in r:
            print(f"| {r['Method']} | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |")
            csv_rows.append({
                "Method": r["Method"],
                "HR@10": "N/A",
                "NDCG@10": "N/A",
                "HR@20": "N/A",
                "NDCG@20": "N/A",
                "HR@50": "N/A",
                "NDCG@50": "N/A",
                "Loss Convergence": "N/A",
                "HR Convergence": "N/A",
                "NDCG Convergence": "N/A",
                "Loss Stability (Std)": "N/A",
                "HR Stability (Std)": "N/A",
                "NDCG Stability (Std)": "N/A",
                "Local Training Time": "N/A",
                "Aggregation Time": "N/A",
            })
            continue

        local_cell = f"{r['Local Training Time']:.2f}s"
        agg_cell = f"{r['Aggregation Time']:.2f}s"

        print(
            f"| {r['Method']} "
            f"| {_fmt(r['HR@10'])} "
            f"| {_fmt(r['NDCG@10'])} "
            f"| {_fmt(r['HR@20'])} "
            f"| {_fmt(r['NDCG@20'])} "
            f"| {_fmt(r['HR@50'])} "
            f"| {_fmt(r['NDCG@50'])} "
            f"| {_fmt(r['Loss Convergence'], 0)} "
            f"| {_fmt(r['HR Convergence'], 0)} "
            f"| {_fmt(r['NDCG Convergence'], 0)} "
            f"| {_fmt(r['Loss Stability (Std)'], 8)} "
            f"| {_fmt(r['HR Stability (Std)'], 8)} "
            f"| {_fmt(r['NDCG Stability (Std)'], 8)} "
            f"| {local_cell} "
            f"| {agg_cell} |"
        )

        csv_rows.append({
            "Method": r["Method"],
            "HR@10": _fmt(r["HR@10"]),
            "NDCG@10": _fmt(r["NDCG@10"]),
            "HR@20": _fmt(r["HR@20"]),
            "NDCG@20": _fmt(r["NDCG@20"]),
            "HR@50": _fmt(r["HR@50"]),
            "NDCG@50": _fmt(r["NDCG@50"]),
            "Loss Convergence": _fmt(r["Loss Convergence"], 0),
            "HR Convergence": _fmt(r["HR Convergence"], 0),
            "NDCG Convergence": _fmt(r["NDCG Convergence"], 0),
            "Loss Stability (Std)": _fmt(r["Loss Stability (Std)"], 8),
            "HR Stability (Std)": _fmt(r["HR Stability (Std)"], 8),
            "NDCG Stability (Std)": _fmt(r["NDCG Stability (Std)"], 8),
            "Local Training Time": local_cell,
            "Aggregation Time": agg_cell,
        })

    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    with CSV_OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "Method",
                "HR@10",
                "NDCG@10",
                "HR@20",
                "NDCG@20",
                "HR@50",
                "NDCG@50",
                "Loss Convergence",
                "HR Convergence",
                "NDCG Convergence",
                "Loss Stability (Std)",
                "HR Stability (Std)",
                "NDCG Stability (Std)",
                "Local Training Time",
                "Aggregation Time",
            ],
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\nSaved CSV: {CSV_OUT}")


if __name__ == "__main__":
    main()