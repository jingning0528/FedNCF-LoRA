import re
from pathlib import Path
from statistics import pstdev
import csv

# ===== Config =====
LOG_FILES = [
    Path("log/baseline_1000/FedNCF-Full.txt"),
    Path("log/baseline_1000/FedNCF-LoRA.txt"),
    Path("log/baseline_1000/MoFiLoRA.txt"),
    Path("log/AB_test_1000/LoRA-MomA.txt"),
    Path("log/AB_test_1000/LoRA-MomB.txt"),
    Path("log/AB_test_1000/LoRA-FixedA.txt"),
    Path("log/AB_test_1000/LoRA-FixedB.txt"),
    Path("log/AB_test_1000/LoRA-MomB-FixedA.txt"),
    Path("log/AB_test_1000/LoRA-MomA-FixedB.txt"),

    # Path("log/industrial/FedNCF-Full.txt"),
    # Path("log/industrial/FedNCF-LoRA.txt"),
    # Path("log/industrial/MoFiLoRA.txt"), #-0.9-0.5

    # Path("log/software/FedNCF-Full.txt"),
    # Path("log/software/FedNCF-LoRA.txt"),
    # Path("log/software/MoFiLoRA.txt"), #-0.9-2.0

]
START_ROUND = 20
END_ROUND = 999
CSV_OUT = Path("csv/baseline_1000_summary.csv")
# add these back
WINDOW_LEN_LOSS = 100   # exactly t_c ... t_c+99
WINDOW_LEN_METRIC = 10
THRESHOLD = 0.9
# LOSS_CONVERGE_THRESHOLD = 0.3   # removed (not used)

# fixed stability range
STABILITY_START_ROUND = 499
STABILITY_END_ROUND = 999
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
    """
    Convergence round t_c is the first round such that for all t in [t_c, t_c+window_len-1]:
        (L_max - L_t) / (L_max - L_min) >= THRESHOLD      if kind == "loss"
        (M_t - M_min) / (M_max - M_min) >= THRESHOLD      if kind == "metric"
    """
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


def metric_stability_std(metric_pairs, start_round, end_round):
    vals = [v for t, v in metric_pairs if start_round <= t <= end_round and v is not None]
    if not vals:
        return None
    if len(vals) == 1:
        return 0.0
    return pstdev(vals)


def metric_delta_std(metric_pairs, start_round, end_round):
    """
    Std of first-order differences inside [start_round, end_round]:
      delta_t = value_t - value_(t-1 in filtered sequence)
    """
    seq = [(t, v) for t, v in metric_pairs if start_round <= t <= end_round and v is not None]
    seq = sorted(seq, key=lambda x: x[0])
    if len(seq) < 2:
        return None

    deltas = [seq[i][1] - seq[i - 1][1] for i in range(1, len(seq))]
    if len(deltas) == 1:
        return 0.0
    return pstdev(deltas)


def _fmt(x, nd=4):
    if x is None:
        return "N/A"
    if isinstance(x, int):
        return str(x)
    return f"{x:.{nd}f}"


def _fmt_conv(x):
    # convergence None -> >900
    if x is None:
        return ">900"
    if isinstance(x, int):
        return str(x)
    return f"{x:.0f}"


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

    # 1) loss convergence: back to >=95% progress for >100 rounds
    loss_T = first_converged_round(loss_pairs, WINDOW_LEN_LOSS, "loss")
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

    # 2) stability: fixed window [499, 999], DELTA std
    loss_delta_std = metric_delta_std(loss_pairs, STABILITY_START_ROUND, STABILITY_END_ROUND)
    hr_delta_std = metric_delta_std(hr_pairs, STABILITY_START_ROUND, STABILITY_END_ROUND)
    ndcg_delta_std = metric_delta_std(ndcg_pairs, STABILITY_START_ROUND, STABILITY_END_ROUND)

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
        "Loss Delta Stability (Std)": loss_delta_std,
        "HR Delta Stability (Std)": hr_delta_std,
        "NDCG Delta Stability (Std)": ndcg_delta_std,
        "Local Training Time": local_sum,
        "Aggregation Time": agg_sum,
    }


def _is_baseline_method(method_name: str) -> bool:
    m = (method_name or "").strip().lower()
    return m in {"fedncf-full", "fedncf-lora"}


def _fmt_with_improve(val, base, higher_better=True, nd=4):
    """
    Example: 0.0118(-50%) or 0.2453(+10%)
    """
    if val is None:
        return "N/A"
    v_txt = _fmt(val, nd)
    if base is None or abs(base) < 1e-12:
        return v_txt
    if higher_better:
        pct = (val - base) / abs(base) * 100.0
    else:
        pct = (base - val) / abs(base) * 100.0
    return f"{v_txt}({pct:+.0f}%)"


def main():
    results = [analyze_one_file(p) for p in LOG_FILES]

    # baseline: FedNCF-LoRA
    lora_row = next(
        (r for r in results if "skip" not in r and (r.get("Method", "").strip().lower() == "fedncf-lora")),
        None
    )

    print("| Method | Loss Delta Stability (Std) | HR Delta Stability (Std) | NDCG Delta Stability (Std)| Loss Convergence | HR Convergence | NDCG Convergence  | HR@10 | NDCG@10 | HR@20 | NDCG@20 | HR@50 | NDCG@50   | Local Training Time | Aggregation Time |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    csv_rows = []

    for r in results:
        method = r["Method"]

        if "skip" in r:
            print(f"| {method} | N/A | N/A | N/A | >900 | >900 | >900 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |")
            csv_rows.append({
                "Method": method,
                "Loss Delta Stability (Std)": "N/A",
                "HR Delta Stability (Std)": "N/A",
                "NDCG Delta Stability (Std)": "N/A",
                "Loss Convergence": ">900",
                "HR Convergence": ">900",
                "NDCG Convergence": ">900",
                "HR@10": "N/A",
                "NDCG@10": "N/A",
                "HR@20": "N/A",
                "NDCG@20": "N/A",
                "HR@50": "N/A",
                "NDCG@50": "N/A",
                "Local Training Time": "N/A",
                "Aggregation Time": "N/A",
            })
            continue

        # raw (baseline rows) or improvement-vs-LoRA (non-baseline rows)
        if _is_baseline_method(method) or lora_row is None:
            loss_std_txt = _fmt(r["Loss Delta Stability (Std)"], 4)
            hr_std_txt = _fmt(r["HR Delta Stability (Std)"], 4)
            ndcg_std_txt = _fmt(r["NDCG Delta Stability (Std)"], 4)

            hr10_txt = _fmt(r["HR@10"], 4)
            ndcg10_txt = _fmt(r["NDCG@10"], 4)
            hr20_txt = _fmt(r["HR@20"], 4)
            ndcg20_txt = _fmt(r["NDCG@20"], 4)
            hr50_txt = _fmt(r["HR@50"], 4)
            ndcg50_txt = _fmt(r["NDCG@50"], 4)

            local_txt = _fmt(r["Local Training Time"], 4)
            agg_txt = _fmt(r["Aggregation Time"], 4)
        else:
            # stability/time: lower is better
            loss_std_txt = _fmt_with_improve(r["Loss Delta Stability (Std)"], lora_row["Loss Delta Stability (Std)"], higher_better=False, nd=4)
            hr_std_txt = _fmt_with_improve(r["HR Delta Stability (Std)"], lora_row["HR Delta Stability (Std)"], higher_better=False, nd=4)
            ndcg_std_txt = _fmt_with_improve(r["NDCG Delta Stability (Std)"], lora_row["NDCG Delta Stability (Std)"], higher_better=False, nd=4)

            # accuracy: higher is better
            hr10_txt = _fmt_with_improve(r["HR@10"], lora_row["HR@10"], higher_better=True, nd=4)
            ndcg10_txt = _fmt_with_improve(r["NDCG@10"], lora_row["NDCG@10"], higher_better=True, nd=4)
            hr20_txt = _fmt_with_improve(r["HR@20"], lora_row["HR@20"], higher_better=True, nd=4)
            ndcg20_txt = _fmt_with_improve(r["NDCG@20"], lora_row["NDCG@20"], higher_better=True, nd=4)
            hr50_txt = _fmt_with_improve(r["HR@50"], lora_row["HR@50"], higher_better=True, nd=4)
            ndcg50_txt = _fmt_with_improve(r["NDCG@50"], lora_row["NDCG@50"], higher_better=True, nd=4)

            local_txt = _fmt_with_improve(r["Local Training Time"], lora_row["Local Training Time"], higher_better=False, nd=4)
            agg_txt = _fmt_with_improve(r["Aggregation Time"], lora_row["Aggregation Time"], higher_better=False, nd=4)

        loss_conv_txt = _fmt_conv(r["Loss Convergence"])
        hr_conv_txt = _fmt_conv(r["HR Convergence"])
        ndcg_conv_txt = _fmt_conv(r["NDCG Convergence"])

        print(
            f"| {method} "
            f"| {loss_std_txt} "
            f"| {hr_std_txt} "
            f"| {ndcg_std_txt} "
            f"| {loss_conv_txt} "
            f"| {hr_conv_txt} "
            f"| {ndcg_conv_txt} "
            f"| {hr10_txt} "
            f"| {ndcg10_txt} "
            f"| {hr20_txt} "
            f"| {ndcg20_txt} "
            f"| {hr50_txt} "
            f"| {ndcg50_txt} "
            f"| {local_txt} "
            f"| {agg_txt} |"
        )

        csv_rows.append({
            "Method": method,
            "Loss Delta Stability (Std)": loss_std_txt,
            "HR Delta Stability (Std)": hr_std_txt,
            "NDCG Delta Stability (Std)": ndcg_std_txt,
            "Loss Convergence": loss_conv_txt,
            "HR Convergence": hr_conv_txt,
            "NDCG Convergence": ndcg_conv_txt,
            "HR@10": hr10_txt,
            "NDCG@10": ndcg10_txt,
            "HR@20": hr20_txt,
            "NDCG@20": ndcg20_txt,
            "HR@50": hr50_txt,
            "NDCG@50": ndcg50_txt,
            "Local Training Time": local_txt,
            "Aggregation Time": agg_txt,
        })

    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    with CSV_OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "Method",
                "Loss Delta Stability (Std)",
                "HR Delta Stability (Std)",
                "NDCG Delta Stability (Std)",
                "Loss Convergence",
                "HR Convergence",
                "NDCG Convergence",
                "HR@10",
                "NDCG@10",
                "HR@20",
                "NDCG@20",
                "HR@50",
                "NDCG@50",
                "Local Training Time",
                "Aggregation Time",
            ],
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\nSaved CSV: {CSV_OUT}")


if __name__ == "__main__":
    main()