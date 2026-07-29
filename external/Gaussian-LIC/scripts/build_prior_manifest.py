#!/usr/bin/env python3
"""Validate prior NPZ shards and build a leakage-safe training manifest."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from semantic_gaussian_prior_model import (
    OUTPUT_DIM,
    SUPPORTED_INPUT_DIMS,
    input_contract,
)


def scalar_text(data, key: str) -> str:
    if key not in data:
        raise ValueError(f"missing scalar metadata {key!r}")
    return str(np.asarray(data[key]).reshape(-1)[0])


def optional_scalar(data, key: str, default):
    if key not in data:
        return default
    return np.asarray(data[key]).reshape(-1)[0].item()


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
    input_dims = set()
    for path in sorted(args.shard_root.rglob("*.npz")):
        data = np.load(path)
        inputs = data["input"]
        targets = data["target"]
        if inputs.ndim != 2 or inputs.shape[1] not in SUPPORTED_INPUT_DIMS:
            raise ValueError(
                f"{path}: expected input [N,D], D in {SUPPORTED_INPUT_DIMS}"
            )
        input_dims.add(int(inputs.shape[1]))
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
        shard = {
            "path": os.path.relpath(path, args.output.parent),
            "count": int(inputs.shape[0]),
            "source": source,
            "sequence": sequence,
            "split": split,
        }
        if source == "r3live_teacher":
            shard["target_encoding"] = optional_scalar(
                data,
                "target_encoding",
                "preactivation_v1",
            )
            shard["mean_offset_limit"] = float(
                optional_scalar(data, "mean_offset_limit", 1.0)
            )
        shards.append(shard)
    if not shards:
        raise RuntimeError("no NPZ shards found")
    if len(input_dims) != 1:
        raise ValueError(f"inconsistent input dimensions: {sorted(input_dims)}")
    input_dim = next(iter(input_dims))
    contract_name, feature_names = input_contract(input_dim)
    r3live_contracts = {
        (shard["target_encoding"], shard["mean_offset_limit"])
        for shard in shards
        if shard["source"] == "r3live_teacher"
    }
    if len(r3live_contracts) > 1:
        raise ValueError(
            f"inconsistent R3LIVE target contracts: {sorted(r3live_contracts)}"
        )
    payload = {
        "format": "semantic-gaussian-prior-v1",
        "input_dim": input_dim,
        "input_contract": contract_name,
        "input_features": feature_names,
        "output_dim": OUTPUT_DIM,
        "fold": args.fold_name,
        "r3live_target_contract": (
            {
                "target_encoding": next(iter(r3live_contracts))[0],
                "mean_offset_limit": next(iter(r3live_contracts))[1],
            }
            if r3live_contracts
            else None
        ),
        "shards": shards,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {len(shards)} shards to {args.output}")


if __name__ == "__main__":
    main()
