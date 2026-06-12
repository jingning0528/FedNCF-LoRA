#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import argparse
import subprocess
from pathlib import Path

VALID_MODELS = {
    "Full",
    "LoRA",
    "LoRA_FixedA",
    "LoRA_FixedB",
    "LoRA_MomA",
    "LoRA_MomB",
    "LoRA_MomB_FixedA",
    "LoRA_MomA_FixedB",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config/")
    parser.add_argument("--expid", type=str, required=True)  # single base expid, e.g. Software
    parser.add_argument("--models", type=str, required=True)
    parser.add_argument("--seeds", type=str, default="2024,2025,2026,2027,2028")
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--log_dir", type=str, default="log/multi_seed")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    bad = [m for m in models if m not in VALID_MODELS]
    if bad:
        raise RuntimeError(f"Unsupported model names: {bad}")

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    failed = []
    for model in models:
        for seed in seeds:
            log_file = log_dir / f"{args.expid}_{model}_{seed}.txt"  # expid_models_seeds
            cmd = [
                sys.executable, "main.py",
                "--config", args.config,
                "--expid", args.expid,
                "--gpu", str(args.gpu),
                "--seed", str(seed),
                "--model", model,
            ]
            print(f"[Run] expid={args.expid}, model={model}, seed={seed} -> {log_file}")
            with log_file.open("w", encoding="utf-8") as f:
                p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
            if p.returncode != 0:
                failed.append((model, seed, p.returncode, str(log_file)))

    if failed:
        print("\n[Failed]")
        for model, seed, code, lf in failed:
            print(f"  model={model}, seed={seed}, code={code}, log={lf}")
        raise SystemExit(1)

    print("\n[Done] all runs finished.")


if __name__ == "__main__":
    main()