#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Amazon 2018 preprocessing.

Outputs:
1. graph/{field}_user.pth
2. graph/{field}_item.pth
3. 5-core_clip/{field}_fl.pth
4. 5-core_clip/{field}_user.csv
5. 5-core_clip/{field}_item.csv

Optional with --use-meta:
6. meta_processed/t5/{field}.pth
"""

import os
import argparse
import torch
import pandas as pd


def pre_process_review(field: str, review_path: str, review_path_clip: str, graph_path: str):
    users = []
    items = []

    users_inter = {}
    items_inter = {}
    review_data_fl = {}

    review_file = os.path.join(review_path, field + "_5.json")

    print(f"Processing reviews: {review_file}")

    for _, review in pd.read_json(review_file, orient="records", lines=True).iterrows():
        reviewer_id = review["reviewerID"]
        asin = review["asin"]

        if reviewer_id not in users:
            users.append(reviewer_id)
            user_idx = len(users) - 1
            users_inter[user_idx] = []
            review_data_fl[user_idx] = []
        else:
            user_idx = users.index(reviewer_id)

        if asin not in items:
            items.append(asin)
            item_idx = len(items) - 1
            items_inter[item_idx] = []
        else:
            item_idx = items.index(asin)

        users_inter[user_idx].append(item_idx)
        items_inter[item_idx].append(user_idx)

        review_dict = {
            "reviewerID": user_idx,
            "asin": item_idx,
            "overall": float(review["overall"]),
        }

        review_data_fl[user_idx].append(review_dict)

    torch.save(users_inter, os.path.join(graph_path, field + "_user.pth"))
    torch.save(items_inter, os.path.join(graph_path, field + "_item.pth"))
    torch.save(review_data_fl, os.path.join(review_path_clip, field + "_fl.pth"))

    pd.DataFrame({"reviewerID": users}).to_csv(
        os.path.join(review_path_clip, field + "_user.csv"), index=False
    )
    pd.DataFrame({"asin": items}).to_csv(
        os.path.join(review_path_clip, field + "_item.csv"), index=False
    )

    print(f"{field}: users={len(users)}, items={len(items)}")
    print(f"Saved interaction files for {field}")

    return items


def pre_process_meta(field: str, items, meta_path: str, meta_path_clip: str, model):
    meta_file = os.path.join(meta_path, "meta_" + field + ".json")

    print(f"Processing metadata: {meta_file}")

    meta_input = [""] * len(items)
    item_to_idx = {asin: idx for idx, asin in enumerate(items)}

    for _, asin_data in pd.read_json(meta_file, orient="records", lines=True).iterrows():
        asin = asin_data.get("asin", None)

        if asin not in item_to_idx:
            continue

        meta_prompt = "A product"

        for attr in ["title", "description", "categories", "brand", "feature"]:
            if attr in asin_data and not isinstance(asin_data[attr], float):
                value = asin_data[attr]
                if isinstance(value, list) and len(value) == 0:
                    continue
                if isinstance(value, str) and len(value.strip()) == 0:
                    continue
                meta_prompt += ", with " + attr + ": " + str(value)

        meta_prompt += "."

        meta_input[item_to_idx[asin]] = meta_prompt

    embedding = model.encode(meta_input, convert_to_tensor=True, show_progress_bar=True)

    print(f"{field} metadata embedding shape: {embedding.shape}")

    torch.save(embedding, os.path.join(meta_path_clip, field + ".pth"))

    print(f"Saved metadata embedding for {field}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--amazon-path",
        type=str,
        default="./data/Amazon/",
        help="Root Amazon data folder",
    )

    parser.add_argument(
        "--categories",
        nargs="+",
        default=["Software", "Industrial_and_Scientific"],
        help="Amazon categories to preprocess",
    )

    parser.add_argument(
        "--use-meta",
        action="store_true",
        help="Whether to process metadata with SentenceTransformer",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/all-mpnet-base-v2",
        help="SentenceTransformer model name or local path",
    )

    args = parser.parse_args()

    review_path = os.path.join(args.amazon_path, "5-core")
    review_path_clip = os.path.join(args.amazon_path, "5-core_clip")
    graph_path = os.path.join(args.amazon_path, "graph")
    meta_path = os.path.join(args.amazon_path, "meta")
    meta_path_clip = os.path.join(args.amazon_path, "meta_processed", "t5")

    os.makedirs(review_path_clip, exist_ok=True)
    os.makedirs(graph_path, exist_ok=True)
    os.makedirs(meta_path_clip, exist_ok=True)

    model = None

    if args.use_meta:
        from sentence_transformers import SentenceTransformer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using device:", device)
        print("Using SentenceTransformer:", args.model)

        model = SentenceTransformer(args.model, device=device)

    for field in args.categories:
        items = pre_process_review(
            field=field,
            review_path=review_path,
            review_path_clip=review_path_clip,
            graph_path=graph_path,
        )

        if args.use_meta:
            pre_process_meta(
                field=field,
                items=items,
                meta_path=meta_path,
                meta_path_clip=meta_path_clip,
                model=model,
            )

    print("All preprocessing finished.")