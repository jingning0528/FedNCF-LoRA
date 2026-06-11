#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import argparse
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config/")
    parser.add_argument("--expid", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seeds", type=str, default="2024,2025,2026,2027,2028")
    parser.add_argument("--log_dir", type=str, default="log/multi_seed")
    args = parser.parse_args()

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        log_file = log_dir / f"{args.expid}_seed{seed}.txt"
        cmd = [
            sys.executable, "main.py",
            "--config", args.config,
            "--expid", args.expid,
            "--gpu", str(args.gpu),
            "--seed", str(seed),
            # no --save_csv => no csv output
        ]
        print(f"[Run] seed={seed} -> {log_file}")
        with log_file.open("w", encoding="utf-8") as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)


if __name__ == "__main__":
    main()