"""
Preprocess Amazon Reviews 2023 5-core CSV for FedNCF / FedPEFT LoRA.

Input:
    ./data/Amazon/5-core/Software.csv
    ./data/Amazon/5-core/Industrial_and_Scientific.csv

Output:
    ./data/Amazon/graph/Software_user.pth
    ./data/Amazon/graph/Software_item.pth
    ./data/Amazon/5-core_clip/Software_fl.pth
    ./data/Amazon/5-core_clip/Software_user.csv
    ./data/Amazon/5-core_clip/Software_item.csv
"""

import os
from pathlib import Path
from collections import defaultdict

import pandas as pd
import torch


BASE_DIR = Path("./data/Amazon").resolve()

review_path = BASE_DIR / "5-core"
review_path_clip = BASE_DIR / "5-core_clip"
graph_path = BASE_DIR / "graph"

os.makedirs(review_path_clip, exist_ok=True)
os.makedirs(graph_path, exist_ok=True)

categories = [
    "Software",
    "Industrial_and_Scientific",
]


def pre_process_review(field: str):
    csv_path = review_path / f"{field}.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Cannot find file: {csv_path}")

    df = pd.read_csv(csv_path)

    required_cols = {"user_id", "parent_asin", "rating", "timestamp"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{field}.csv missing columns: {missing}")

    # Important: sort by user and timestamp.
    # The FedPEFT dataloader uses the last interaction as test sample.
    df = df.sort_values(["user_id", "timestamp"])

    users = []
    items = []
    user2id = {}
    item2id = {}

    users_inter = defaultdict(list)       # {user_id_int: [item_id_int, ...]}
    items_inter = defaultdict(list)       # {item_id_int: [user_id_int, ...]}
    review_data_fl = defaultdict(list)    # {user_id_int: [review_dict, ...]}

    for _, row in df.iterrows():
        raw_user = row["user_id"]
        raw_item = row["parent_asin"]

        if raw_user not in user2id:
            user2id[raw_user] = len(users)
            users.append(raw_user)

        if raw_item not in item2id:
            item2id[raw_item] = len(items)
            items.append(raw_item)

        uid = user2id[raw_user]
        iid = item2id[raw_item]

        users_inter[uid].append(iid)
        items_inter[iid].append(uid)

        # Keep old FedPEFT key names, so dataloader does not need modification.
        review_dict = {
            "reviewerID": uid,
            "asin": iid,
            "overall": float(row["rating"]),
            "unixReviewTime": int(row["timestamp"]),
        }

        review_data_fl[uid].append(review_dict)

    torch.save(dict(users_inter), graph_path / f"{field}_user.pth")
    torch.save(dict(items_inter), graph_path / f"{field}_item.pth")
    torch.save(dict(review_data_fl), review_path_clip / f"{field}_fl.pth")

    pd.DataFrame({"reviewerID": users}).to_csv(
        review_path_clip / f"{field}_user.csv", index=False
    )
    pd.DataFrame({"asin": items}).to_csv(
        review_path_clip / f"{field}_item.csv", index=False
    )

    print(
        f"{field}: "
        f"users={len(users)}, items={len(items)}, interactions={len(df)}"
    )


if __name__ == "__main__":
    for field in categories:
        pre_process_review(field)