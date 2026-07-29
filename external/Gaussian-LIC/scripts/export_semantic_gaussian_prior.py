#!/usr/bin/env python3
"""Export a trained prior head to the TorchScript contract used by C++."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from semantic_gaussian_prior_model import (
    INPUT_DIM,
    OUTPUT_DIM,
    OBJECT_LATENT_DIM,
    SemanticGaussianPriorHead,
    input_contract,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    input_dim = int(checkpoint.get("input_dim", INPUT_DIM))
    inferred_contract, inferred_features = input_contract(input_dim)
    contract_name = str(
        checkpoint.get("input_contract", inferred_contract)
    )
    feature_names = tuple(
        checkpoint.get("input_features", inferred_features)
    )
    if contract_name != inferred_contract or feature_names != inferred_features:
        raise ValueError(
            "checkpoint input contract does not match its input dimension"
        )
    model = SemanticGaussianPriorHead(
        int(checkpoint["hidden_dim"]),
        int(checkpoint["layers"]),
        input_dim,
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()
    scripted = torch.jit.script(model)
    with torch.no_grad():
        probe = torch.zeros((7, input_dim), dtype=torch.float32)
        output = scripted(probe)
    if output.shape != (7, OUTPUT_DIM) or not torch.isfinite(output).all():
        raise RuntimeError("exported model failed its shape/finite smoke test")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(args.output))
    metadata = {
        "input_dim": input_dim,
        "input_contract": contract_name,
        "input_features": feature_names,
        "object_latent_dim": OBJECT_LATENT_DIM,
        "output_dim": OUTPUT_DIM,
        "stage": checkpoint.get("stage", "unknown"),
        "metrics": checkpoint.get("metrics", {}),
        "run_config": checkpoint.get("run_config", {}),
        "decoder": ["mean_xyz", "log_scale", "quaternion", "rgb", "opacity_logit"],
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
