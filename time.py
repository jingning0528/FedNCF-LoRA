import re
from pathlib import Path
from statistics import mean, median

# ===== Config (change here) =====
LOG_FILE = Path("log/DeltaB_time.txt")  # change file name here
START_ROUND = 20                            # default start round
END_ROUND = 200                             # default end round (inclusive)
# ================================

# Example line:
# [Time] turn=20 local_train_time=8.1430s avg_client_train_time=... aggregation_time=... round_time=...
PATTERN = re.compile(
    r"\[Time\]\s+turn=(?P<turn>\d+).*?"
    r"local_train_time=(?P<local>[0-9]*\.?[0-9]+)s.*?"
    r"(?:aggregation_time=(?P<agg>[0-9]*\.?[0-9]+)s.*?)?"
    r"(?:round_time=(?P<round>[0-9]*\.?[0-9]+)s)?"
)

PRETRAIN_PATTERN = re.compile(r"\[Time\]\s+pretrain_time=(?P<pre>[0-9]*\.?[0-9]+)s")


def parse_times(log_path: Path):
    rows = []
    pretrain_time = None

    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m_pre = PRETRAIN_PATTERN.search(line)
            if m_pre:
                pretrain_time = float(m_pre.group("pre"))

            m = PATTERN.search(line)
            if not m:
                continue

            rows.append(
                {
                    "turn": int(m.group("turn")),
                    "local_train_time": float(m.group("local")),
                    "aggregation_time": float(m.group("agg")) if m.group("agg") else None,
                    "round_time": float(m.group("round")) if m.group("round") else None,
                }
            )
    return rows, pretrain_time


def main():
    if not LOG_FILE.exists():
        raise FileNotFoundError(f"Log file not found: {LOG_FILE}")

    if START_ROUND > END_ROUND:
        raise ValueError("START_ROUND must be <= END_ROUND")

    rows, pretrain_time = parse_times(LOG_FILE)
    selected = [r for r in rows if START_ROUND <= r["turn"] <= END_ROUND]

    if not selected:
        print(f"No [Time] turn records found in round range [{START_ROUND}, {END_ROUND}].")
        return

    local_times = [r["local_train_time"] for r in selected]
    round_times = [r["round_time"] for r in selected if r["round_time"] is not None]

    # Requested stats (local train time)
    avg_local = mean(local_times)
    med_local = median(local_times)
    sum_local = sum(local_times)

    print(f"File: {LOG_FILE}")
    print(f"Round range: [{START_ROUND}, {END_ROUND}]")
    print(f"Matched rounds: {len(selected)}")
    print("-" * 50)
    print(f"Average local_train_time: {avg_local:.4f}s")
    print(f"Median  local_train_time: {med_local:.4f}s")
    print(f"Accumulated local_train_time: {sum_local:.4f}s ({sum_local/60:.2f} min)")

    # Extra useful info
    if round_times:
        sum_round = sum(round_times)
        print(f"Accumulated round_time: {sum_round:.4f}s ({sum_round/60:.2f} min)")
    if pretrain_time is not None:
        print(f"Pretrain time: {pretrain_time:.4f}s ({pretrain_time/60:.2f} min)")
        print(
            f"Total (pretrain + accumulated local_train_time): "
            f"{pretrain_time + sum_local:.4f}s ({(pretrain_time + sum_local)/60:.2f} min)"
        )


if __name__ == "__main__":
    main()