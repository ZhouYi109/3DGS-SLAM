#!/usr/bin/env python3
"""Validate prior NPZ shards and build a leakage-safe training manifest."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from semantic_gaussian_prior_model import INPUT_DIM, OUTPUT_DIM


def scalar_text(data, key: str) -> str:
    if key not in data:
        raise ValueError(f"missing scalar metadata {key!r}")
    return str(np.asarray(data[key]).reshape(-1)[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--r3live-fold", type=Path)
    parser.add_argument("--fold-name")
    args = parser.parse_args()

    if bool(args.r3live_fold) != bool(args.fold_name):
        raise ValueError("--r3live-fold and --fold-name must be provided together")

    sequence_split: dict[str, str] = {}
    if args.r3live_fold:
        protocol = json.loads(args.r3live_fold.read_text(encoding="utf-8"))
        selected = next(
            (fold for fold in protocol["folds"] if fold["name"] == args.fold_name),
            None,
        )
        if selected is None:
            raise ValueError(f"unknown fold {args.fold_name!r}")
        for split in ("train", "validation", "test"):
            sequence_split.update({name: split for name in selected[split]})

    shards = []
    for path in sorted(args.shard_root.rglob("*.npz")):
        data = np.load(path)
        inputs = data["input"]
        targets = data["target"]
        if inputs.ndim != 2 or inputs.shape[1] != INPUT_DIM:
            raise ValueError(f"{path}: expected input [N,{INPUT_DIM}]")
        if targets.shape != (inputs.shape[0], OUTPUT_DIM):
            raise ValueError(f"{path}: expected target [N,{OUTPUT_DIM}]")
        source = scalar_text(data, "source")
        sequence = scalar_text(data, "sequence")
        embedded_split = scalar_text(data, "split")
        if embedded_split not in {"train", "validation", "test"}:
            raise ValueError(f"{path}: invalid split {embedded_split!r}")
        if source == "r3live_teacher" and sequence_split:
            if sequence not in sequence_split:
                raise ValueError(f"{path}: sequence {sequence!r} absent from fold")
            split = sequence_split[sequence]
            if embedded_split != split:
                raise ValueError(
                    f"{path}: embedded split {embedded_split!r} conflicts with "
                    f"fold split {split!r}"
                )
        else:
            split = embedded_split
        shards.append(
            {
                "path": os.path.relpath(path, args.output.parent),
                "count": int(inputs.shape[0]),
                "source": source,
                "sequence": sequence,
                "split": split,
            }
        )
    if not shards:
        raise RuntimeError("no NPZ shards found")
    payload = {
        "format": "semantic-gaussian-prior-v1",
        "input_dim": INPUT_DIM,
        "output_dim": OUTPUT_DIM,
        "fold": args.fold_name,
        "shards": shards,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {len(shards)} shards to {args.output}")


if __name__ == "__main__":
    main()
