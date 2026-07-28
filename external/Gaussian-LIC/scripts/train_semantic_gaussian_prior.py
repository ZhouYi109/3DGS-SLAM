#!/usr/bin/env python3
"""Train the nuScenes -> R3LIVE Semantic Gaussian Prior pipeline.

Each NPZ shard contains:
  input:  float32 [N,24], exactly matching the C++ online decoder input
  target: float32 [N,14], Gaussian residual teacher targets
  weight: optional float32 [N]
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from semantic_gaussian_prior_model import (
    INPUT_DIM,
    OUTPUT_DIM,
    SemanticGaussianPriorHead,
    prior_loss,
)


class PriorShardDataset(Dataset):
    def __init__(self, manifest: Path, split: str, source: str) -> None:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        root = manifest.parent
        self.samples: list[tuple[Path, int]] = []
        self.cache: dict[Path, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for shard in payload["shards"]:
            if shard["split"] != split or (source != "all" and shard["source"] != source):
                continue
            path = root / shard["path"]
            count = int(shard["count"])
            self.samples.extend((path, index) for index in range(count))
        if not self.samples:
            raise ValueError(f"no samples for split={split!r}, source={source!r}")

    def __len__(self) -> int:
        return len(self.samples)

    def _load(self, path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if path not in self.cache:
            data = np.load(path)
            inputs = np.asarray(data["input"], dtype=np.float32)
            targets = np.asarray(data["target"], dtype=np.float32)
            weights = np.asarray(
                data["weight"] if "weight" in data else np.ones(inputs.shape[0]),
                dtype=np.float32,
            )
            if inputs.ndim != 2 or inputs.shape[1] != INPUT_DIM:
                raise ValueError(f"{path}: input must be [N,{INPUT_DIM}]")
            if targets.shape != (inputs.shape[0], OUTPUT_DIM):
                raise ValueError(f"{path}: target must be [N,{OUTPUT_DIM}]")
            if weights.shape != (inputs.shape[0],):
                raise ValueError(f"{path}: weight must be [N]")
            self.cache[path] = inputs, targets, weights
        return self.cache[path]

    def __getitem__(self, index: int):
        path, row = self.samples[index]
        inputs, targets, weights = self._load(path)
        return inputs[row], targets[row], weights[row]


@torch.no_grad()
def evaluate(model, loader, device) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {"loss": 0.0}
    count = 0
    for features, target, weight in loader:
        prediction = model(features.to(device))
        loss, groups = prior_loss(prediction, target.to(device), weight.to(device))
        batch = features.shape[0]
        totals["loss"] += float(loss) * batch
        for name, value in groups.items():
            totals[name] = totals.get(name, 0.0) + float(value) * batch
        count += batch
    return {name: value / max(1, count) for name, value in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--stage", choices=("nuscenes_pretrain", "r3live_distill"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260727)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source = "nuscenes" if args.stage == "nuscenes_pretrain" else "r3live_teacher"
    train_data = PriorShardDataset(args.manifest, "train", source)
    validation_data = PriorShardDataset(args.manifest, "validation", source)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    model = SemanticGaussianPriorHead(args.hidden_dim, args.layers).to(device)
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs)
    )
    args.output.mkdir(parents=True, exist_ok=True)
    history = []
    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_total = 0.0
        train_count = 0
        for features, target, weight in train_loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(features.to(device, non_blocking=True))
            loss, _ = prior_loss(
                prediction,
                target.to(device, non_blocking=True),
                weight.to(device, non_blocking=True),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_total += float(loss.detach()) * features.shape[0]
            train_count += features.shape[0]
        scheduler.step()
        metrics = evaluate(model, validation_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": train_total / max(1, train_count),
            "learning_rate": optimizer.param_groups[0]["lr"],
            **{f"validation_{key}": value for key, value in metrics.items()},
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True))
        checkpoint = {
            "model": model.state_dict(),
            "hidden_dim": args.hidden_dim,
            "layers": args.layers,
            "stage": args.stage,
            "epoch": epoch,
            "metrics": row,
        }
        torch.save(checkpoint, args.output / "last.pt")
        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            torch.save(checkpoint, args.output / "best.pt")

    (args.output / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

