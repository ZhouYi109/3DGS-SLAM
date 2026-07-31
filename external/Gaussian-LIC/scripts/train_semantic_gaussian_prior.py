#!/usr/bin/env python3
"""Train the nuScenes -> R3LIVE Semantic Gaussian Prior pipeline."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from semantic_gaussian_prior_model import (
    BASE_INPUT_DIM,
    CONTEXT_INPUT_DIM,
    OUTPUT_DIM,
    OBJECT_LATENT_END_INDEX,
    OBJECT_LATENT_START_INDEX,
    SEMANTIC_CONFIDENCE_INDEX,
    SUPPORTED_INPUT_DIMS,
    SemanticGaussianPriorHead,
    input_contract,
    prior_loss,
)


def configure_context_adapter_only(
    model: SemanticGaussianPriorHead,
) -> tuple[torch.nn.Parameter, torch.Tensor]:
    if model.input_dim != CONTEXT_INPUT_DIM:
        raise ValueError("context-adapter-only training requires a 38D model")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    input_weight = model.backbone[0].weight
    input_weight.requires_grad_(True)
    frozen_base_columns = input_weight[:, :BASE_INPUT_DIM].detach().clone()
    gradient_mask = torch.zeros_like(input_weight)
    gradient_mask[:, BASE_INPUT_DIM:] = 1.0
    input_weight.register_hook(lambda gradient: gradient * gradient_mask)
    return input_weight, frozen_base_columns


def restore_context_adapter_base_columns(
    input_weight: torch.nn.Parameter,
    frozen_base_columns: torch.Tensor,
) -> None:
    with torch.no_grad():
        input_weight[:, :BASE_INPUT_DIM].copy_(frozen_base_columns)


def render_aware_rollout_weight(
    sample_weight: torch.Tensor,
    visibility: torch.Tensor,
    gradient: torch.Tensor,
    visibility_floor: float,
    gradient_gain: float,
) -> torch.Tensor:
    if gradient.ndim != 2 or gradient.shape[1] != 5:
        raise ValueError("rollout gradient must have shape [N,5]")
    visibility_factor = visibility_floor + (1.0 - visibility_floor) * (
        visibility.reshape(-1).clamp(0.0, 1.0)
    )
    gradient_strength = torch.log1p(gradient.clamp_min(0.0)).mean(dim=1)
    gradient_scale = gradient_strength / gradient_strength.mean().detach().clamp_min(
        1e-6
    )
    gradient_factor = 1.0 + gradient_gain * gradient_scale.clamp(0.0, 4.0)
    return sample_weight.reshape(-1).clamp_min(0.0) * visibility_factor * gradient_factor


def load_compatible_model_state(
    model: SemanticGaussianPriorHead,
    checkpoint_state: dict[str, torch.Tensor],
) -> None:
    compatible_state = dict(checkpoint_state)
    first_weight_key = "backbone.0.weight"
    current_state = model.state_dict()
    if (
        first_weight_key in compatible_state
        and compatible_state[first_weight_key].shape
        != current_state[first_weight_key].shape
    ):
        old_weight = compatible_state[first_weight_key]
        target_weight = current_state[first_weight_key]
        if old_weight.shape[0] != target_weight.shape[0]:
            raise ValueError(
                "cannot transfer checkpoint with a different hidden dimension"
            )
        copied_columns = min(old_weight.shape[1], target_weight.shape[1])
        new_weight = torch.zeros_like(target_weight)
        new_weight[:, :copied_columns] = old_weight[:, :copied_columns].to(
            device=new_weight.device,
            dtype=new_weight.dtype,
        )
        compatible_state[first_weight_key] = new_weight
    model.load_state_dict(compatible_state)


class PriorShardDataset(Dataset):
    def __init__(
        self,
        manifest: Path,
        split: str,
        source: str,
        zero_input_features: tuple[str, ...] = (),
        include_rollout: bool = False,
    ) -> None:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        root = manifest.parent
        self.input_dim = int(payload["input_dim"])
        if self.input_dim not in SUPPORTED_INPUT_DIMS:
            raise ValueError(
                f"unsupported manifest input dimension: {self.input_dim}"
            )
        inferred_contract, inferred_features = input_contract(self.input_dim)
        self.input_contract = str(
            payload.get("input_contract", inferred_contract)
        )
        self.input_features = tuple(
            payload.get("input_features", inferred_features)
        )
        if (
            self.input_contract != inferred_contract
            or self.input_features != inferred_features
        ):
            raise ValueError(
                "manifest input contract does not match its input dimension"
            )
        unknown_features = sorted(
            set(zero_input_features).difference(self.input_features)
        )
        if unknown_features:
            raise ValueError(
                "cannot zero unknown input features: "
                + ", ".join(unknown_features)
            )
        self.zero_input_indices = tuple(
            self.input_features.index(name) for name in zero_input_features
        )
        self.rollout_contract = payload.get("rollout_contract")
        self.include_rollout = bool(include_rollout)
        if self.include_rollout and not self.rollout_contract:
            raise ValueError("manifest does not contain a rollout contract")
        self.samples: list[tuple[Path, int]] = []
        self.sample_sequence_ids: list[int] = []
        self.sequence_names: list[str] = []
        self.cache: dict[
            Path,
            tuple[
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray | None,
                np.ndarray | None,
                np.ndarray | None,
            ],
        ] = {}
        sequence_ids: dict[str, int] = {}
        for shard in payload["shards"]:
            if shard["split"] != split or (
                source != "all" and shard["source"] != source
            ):
                continue
            path = root / shard["path"]
            count = int(shard["count"])
            sequence = str(shard["sequence"])
            sequence_id = sequence_ids.setdefault(sequence, len(sequence_ids))
            if sequence_id == len(self.sequence_names):
                self.sequence_names.append(sequence)
            self.samples.extend((path, index) for index in range(count))
            self.sample_sequence_ids.extend([sequence_id] * count)
        if not self.samples:
            raise ValueError(f"no samples for split={split!r}, source={source!r}")

    def __len__(self) -> int:
        return len(self.samples)

    def _load(
        self, path: Path
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
    ]:
        if path not in self.cache:
            data = np.load(path)
            inputs = np.asarray(data["input"], dtype=np.float32)
            targets = np.asarray(data["target"], dtype=np.float32)
            weights = np.asarray(
                data["weight"] if "weight" in data else np.ones(inputs.shape[0]),
                dtype=np.float32,
            )
            if inputs.ndim != 2 or inputs.shape[1] != self.input_dim:
                raise ValueError(
                    f"{path}: input must be [N,{self.input_dim}]"
                )
            if targets.shape != (inputs.shape[0], OUTPUT_DIM):
                raise ValueError(f"{path}: target must be [N,{OUTPUT_DIM}]")
            if weights.shape != (inputs.shape[0],):
                raise ValueError(f"{path}: weight must be [N]")
            rollout_target = None
            rollout_visibility = None
            rollout_gradient = None
            if self.include_rollout:
                rollout_target = np.asarray(
                    data["rollout_target"], dtype=np.float32
                )
                rollout_visibility = np.asarray(
                    data["rollout_visibility"], dtype=np.float32
                )
                rollout_gradient = np.asarray(
                    data["rollout_gradient"], dtype=np.float32
                )
                if rollout_target.shape != (inputs.shape[0], OUTPUT_DIM):
                    raise ValueError(
                        f"{path}: rollout_target must be [N,{OUTPUT_DIM}]"
                    )
                if rollout_visibility.shape != (inputs.shape[0],):
                    raise ValueError(
                        f"{path}: rollout_visibility must be [N]"
                    )
                if rollout_gradient.shape != (inputs.shape[0], 5):
                    raise ValueError(
                        f"{path}: rollout_gradient must be [N,5]"
                    )
            self.cache[path] = (
                inputs,
                targets,
                weights,
                rollout_target,
                rollout_visibility,
                rollout_gradient,
            )
        return self.cache[path]

    def __getitem__(self, index: int):
        path, row = self.samples[index]
        (
            inputs,
            targets,
            weights,
            rollout_target,
            rollout_visibility,
            rollout_gradient,
        ) = self._load(path)
        features = inputs[row]
        if self.zero_input_indices:
            features = features.copy()
            features[list(self.zero_input_indices)] = 0.0
        if self.include_rollout:
            return (
                features,
                targets[row],
                weights[row],
                rollout_target[row],
                rollout_visibility[row],
                rollout_gradient[row],
            )
        return features, targets[row], weights[row]

    def _unique_shards(self) -> list[tuple[Path, int, int]]:
        shards: list[tuple[Path, int, int]] = []
        start = 0
        while start < len(self.samples):
            path = self.samples[start][0]
            end = start + 1
            while end < len(self.samples) and self.samples[end][0] == path:
                end += 1
            shards.append((path, start, end))
            start = end
        return shards

    def balanced_sampling_weights(
        self,
        semantic_fraction: float,
        sequence_balanced: bool,
    ) -> tuple[torch.Tensor, dict[str, object]]:
        if not 0.0 <= semantic_fraction < 1.0:
            raise ValueError("semantic_fraction must be in [0,1)")
        semantic = np.zeros(len(self.samples), dtype=bool)
        for path, start, end in self._unique_shards():
            inputs, _, _, _, _, _ = self._load(path)
            semantic[start:end] = (
                (inputs[:, SEMANTIC_CONFIDENCE_INDEX] > 0.0)
                & (
                    np.abs(
                        inputs[
                            :,
                            OBJECT_LATENT_START_INDEX:OBJECT_LATENT_END_INDEX,
                        ]
                    ).sum(axis=1)
                    > 0.0
                )
            )

        sequence_ids = np.asarray(self.sample_sequence_ids, dtype=np.int32)
        sequence_values = np.unique(sequence_ids)
        weights = np.zeros(len(self.samples), dtype=np.float64)
        for sequence_id in sequence_values:
            sequence_mask = sequence_ids == sequence_id
            sequence_mass = (
                1.0 / len(sequence_values)
                if sequence_balanced
                else float(sequence_mask.sum()) / len(self.samples)
            )
            semantic_mask = sequence_mask & semantic
            nonsemantic_mask = sequence_mask & ~semantic
            semantic_count = int(semantic_mask.sum())
            nonsemantic_count = int(nonsemantic_mask.sum())
            if semantic_fraction > 0.0 and semantic_count and nonsemantic_count:
                weights[semantic_mask] = (
                    sequence_mass * semantic_fraction / semantic_count
                )
                weights[nonsemantic_mask] = (
                    sequence_mass * (1.0 - semantic_fraction) / nonsemantic_count
                )
            else:
                weights[sequence_mask] = sequence_mass / int(sequence_mask.sum())
        weights /= weights.sum()
        summary = {
            "rows": len(self.samples),
            "sequences": {
                self.sequence_names[int(sequence_id)]: {
                    "rows": int((sequence_ids == sequence_id).sum()),
                    "semantic_rows": int(
                        ((sequence_ids == sequence_id) & semantic).sum()
                    ),
                }
                for sequence_id in sequence_values
            },
            "natural_semantic_fraction": float(semantic.mean()),
            "target_semantic_fraction": semantic_fraction,
            "sequence_balanced": sequence_balanced,
        }
        return torch.from_numpy(weights), summary


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    loss_weights,
    decoded_residual_targets,
    rollout_loss_weight=0.0,
    rollout_visibility_floor=0.1,
    rollout_gradient_gain=0.1,
) -> dict[str, float]:
    model.eval()
    subsets = {
        "": {"numerator": {}, "denominator": 0.0},
        "semantic_": {"numerator": {}, "denominator": 0.0},
        "nonsemantic_": {"numerator": {}, "denominator": 0.0},
    }
    for batch in loader:
        features, target, weight = batch[:3]
        rollout_target = batch[3] if len(batch) == 6 else None
        rollout_visibility = batch[4] if len(batch) == 6 else None
        rollout_gradient = batch[5] if len(batch) == 6 else None
        features_device = features.to(device)
        target_device = target.to(device)
        weight_device = weight.to(device)
        rollout_target_device = (
            rollout_target.to(device) if rollout_target is not None else None
        )
        rollout_visibility_device = (
            rollout_visibility.to(device)
            if rollout_visibility is not None
            else None
        )
        rollout_gradient_device = (
            rollout_gradient.to(device)
            if rollout_gradient is not None
            else None
        )
        prediction = model(features_device)
        semantic_mask = (
            (features_device[:, SEMANTIC_CONFIDENCE_INDEX] > 0.0)
            & (
                features_device[
                    :,
                    OBJECT_LATENT_START_INDEX:OBJECT_LATENT_END_INDEX,
                ].abs().sum(dim=1)
                > 0.0
            )
        )
        masks = {
            "": torch.ones_like(semantic_mask, dtype=torch.bool),
            "semantic_": semantic_mask,
            "nonsemantic_": ~semantic_mask,
        }
        for prefix, mask in masks.items():
            if not mask.any():
                continue
            selected_weight = weight_device[mask]
            denominator = float(selected_weight.sum())
            if denominator <= 0.0:
                continue
            loss, groups = prior_loss(
                prediction[mask],
                target_device[mask],
                selected_weight,
                loss_weights,
                decoded_residual_targets,
            )
            values = {"loss": loss.detach(), **groups}
            if rollout_target_device is not None:
                rollout_weight = render_aware_rollout_weight(
                    selected_weight,
                    rollout_visibility_device[mask],
                    rollout_gradient_device[mask],
                    rollout_visibility_floor,
                    rollout_gradient_gain,
                )
                rollout_loss, rollout_groups = prior_loss(
                    prediction[mask],
                    rollout_target_device[mask],
                    rollout_weight,
                    loss_weights,
                    decoded_residual_targets,
                )
                values["rollout_loss"] = rollout_loss.detach()
                values["task_loss"] = (
                    loss + rollout_loss_weight * rollout_loss
                ).detach()
                values.update(
                    {
                        f"rollout_{name}": value
                        for name, value in rollout_groups.items()
                    }
                )
            subsets[prefix]["denominator"] += denominator
            for name, value in values.items():
                numerator = subsets[prefix]["numerator"]
                numerator[name] = numerator.get(name, 0.0) + float(value) * denominator

    result = {}
    for prefix, subset in subsets.items():
        denominator = max(1e-6, subset["denominator"])
        for name, numerator in subset["numerator"].items():
            result[f"{prefix}{name}"] = numerator / denominator
        mean_key = f"{prefix}mean"
        scale_key = f"{prefix}scale"
        if mean_key in result and scale_key in result:
            result[f"{prefix}geometry_loss"] = (
                result[mean_key] + result[scale_key]
            )
    if "semantic_loss" in result and "nonsemantic_loss" in result:
        result["balanced_loss"] = 0.5 * (
            result["semantic_loss"] + result["nonsemantic_loss"]
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=("nuscenes_pretrain", "r3live_distill"),
        required=True,
    )
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
    parser.add_argument("--semantic-sample-fraction", type=float, default=0.0)
    parser.add_argument("--sequence-balanced-sampling", action="store_true")
    parser.add_argument("--freeze-backbone-blocks", type=int, default=0)
    parser.add_argument("--backbone-learning-rate-scale", type=float, default=1.0)
    parser.add_argument("--context-adapter-only", action="store_true")
    parser.add_argument(
        "--zero-input-feature",
        action="append",
        default=[],
        help=(
            "Set a named manifest feature to zero in both training and "
            "validation; repeat for controlled feature ablations."
        ),
    )
    parser.add_argument("--mean-loss-weight", type=float, default=1.0)
    parser.add_argument("--scale-loss-weight", type=float, default=1.0)
    parser.add_argument("--rotation-loss-weight", type=float, default=0.5)
    parser.add_argument("--color-loss-weight", type=float, default=0.5)
    parser.add_argument("--opacity-loss-weight", type=float, default=0.25)
    parser.add_argument("--saturated-mean-threshold", type=float, default=0.0)
    parser.add_argument("--saturated-mean-weight", type=float, default=1.0)
    parser.add_argument("--decoded-residual-targets", action="store_true")
    parser.add_argument("--rollout-loss-weight", type=float, default=0.0)
    parser.add_argument("--rollout-visibility-floor", type=float, default=0.1)
    parser.add_argument("--rollout-gradient-gain", type=float, default=0.1)
    parser.add_argument(
        "--selection-metric",
        choices=(
            "validation_loss",
            "validation_balanced_loss",
            "validation_semantic_loss",
            "validation_geometry_loss",
            "validation_semantic_geometry_loss",
            "validation_task_loss",
        ),
        default="validation_loss",
    )
    args = parser.parse_args()

    if not 0.0 <= args.semantic_sample_fraction < 1.0:
        raise ValueError("--semantic-sample-fraction must be in [0,1)")
    if args.context_adapter_only and not args.init_checkpoint:
        raise ValueError("--context-adapter-only requires --init-checkpoint")
    if args.context_adapter_only and args.freeze_backbone_blocks != 0:
        raise ValueError(
            "--context-adapter-only cannot be combined with "
            "--freeze-backbone-blocks"
        )
    if not 0.0 < args.backbone_learning_rate_scale <= 1.0:
        raise ValueError("--backbone-learning-rate-scale must be in (0,1]")
    if not 0.0 <= args.saturated_mean_weight <= 1.0:
        raise ValueError("--saturated-mean-weight must be in [0,1]")
    if args.rollout_loss_weight < 0.0:
        raise ValueError("--rollout-loss-weight must be non-negative")
    if not 0.0 <= args.rollout_visibility_floor <= 1.0:
        raise ValueError("--rollout-visibility-floor must be in [0,1]")
    if args.rollout_gradient_gain < 0.0:
        raise ValueError("--rollout-gradient-gain must be non-negative")
    if args.rollout_loss_weight > 0.0 and args.stage != "r3live_distill":
        raise ValueError("rollout supervision is only valid for R3LIVE distillation")
    loss_weights = (
        args.mean_loss_weight,
        args.scale_loss_weight,
        args.rotation_loss_weight,
        args.color_loss_weight,
        args.opacity_loss_weight,
    )
    if any(value < 0.0 for value in loss_weights):
        raise ValueError("loss weights must be non-negative")
    manifest_payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    target_contract = manifest_payload.get("r3live_target_contract")
    if args.stage == "r3live_distill" and target_contract:
        expected_decoded = target_contract["target_encoding"] == "decoded_v2"
        if args.decoded_residual_targets != expected_decoded:
            raise ValueError(
                "--decoded-residual-targets does not match the manifest's "
                f"{target_contract['target_encoding']!r} contract"
            )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source = "nuscenes" if args.stage == "nuscenes_pretrain" else "r3live_teacher"
    zero_input_features = tuple(dict.fromkeys(args.zero_input_feature))
    include_rollout = args.rollout_loss_weight > 0.0
    train_data = PriorShardDataset(
        args.manifest,
        "train",
        source,
        zero_input_features,
        include_rollout,
    )
    validation_data = PriorShardDataset(
        args.manifest,
        "validation",
        source,
        zero_input_features,
        include_rollout,
    )

    sampler = None
    sampling_summary = None
    if args.semantic_sample_fraction > 0.0 or args.sequence_balanced_sampling:
        sampling_weights, sampling_summary = train_data.balanced_sampling_weights(
            args.semantic_sample_fraction,
            args.sequence_balanced_sampling,
        )
        sampler = WeightedRandomSampler(
            sampling_weights,
            num_samples=len(train_data),
            replacement=True,
            generator=torch.Generator().manual_seed(args.seed),
        )
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
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

    if train_data.input_dim != validation_data.input_dim:
        raise ValueError("training and validation input dimensions differ")
    if (
        train_data.input_contract != validation_data.input_contract
        or train_data.input_features != validation_data.input_features
    ):
        raise ValueError("training and validation input contracts differ")
    model = SemanticGaussianPriorHead(
        args.hidden_dim,
        args.layers,
        train_data.input_dim,
    ).to(device)
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location="cpu")
        load_compatible_model_state(model, checkpoint["model"])
    backbone_blocks = args.layers - 1
    if not 0 <= args.freeze_backbone_blocks <= backbone_blocks:
        raise ValueError(
            f"--freeze-backbone-blocks must be in [0,{backbone_blocks}]"
        )
    backbone_modules = list(model.backbone.children())
    for module in backbone_modules[: 3 * args.freeze_backbone_blocks]:
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    context_adapter_weight = None
    frozen_base_columns = None
    if args.context_adapter_only:
        context_adapter_weight, frozen_base_columns = (
            configure_context_adapter_only(model)
        )
        backbone_parameters = [context_adapter_weight]
        parameter_groups = [
            {
                "params": backbone_parameters,
                "lr": args.learning_rate,
            }
        ]
    else:
        backbone_parameters = [
            parameter
            for parameter in model.backbone.parameters()
            if parameter.requires_grad
        ]
        parameter_groups = [
            {
                "params": [
                    parameter
                    for parameter in model.output.parameters()
                    if parameter.requires_grad
                ],
                "lr": args.learning_rate,
            }
        ]
        if backbone_parameters:
            parameter_groups.insert(
                0,
                {
                    "params": backbone_parameters,
                    "lr": args.learning_rate
                    * args.backbone_learning_rate_scale,
                },
            )
    optimizer = torch.optim.AdamW(
        parameter_groups,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs),
    )

    args.output.mkdir(parents=True, exist_ok=True)
    run_config = {
        **vars(args),
        "manifest": str(args.manifest),
        "output": str(args.output),
        "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint else None,
        "loss_weights": loss_weights,
        "sampling": sampling_summary,
        "target_contract": target_contract,
        "rollout_contract": train_data.rollout_contract,
        "input_dim": train_data.input_dim,
        "input_contract": train_data.input_contract,
        "input_features": train_data.input_features,
        "zero_input_features": zero_input_features,
    }
    (args.output / "run_config.json").write_text(
        json.dumps(run_config, indent=2),
        encoding="utf-8",
    )

    history = []
    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_total = 0.0
        train_denominator = 0.0
        for batch in train_loader:
            features, target, weight = batch[:3]
            rollout_target = batch[3] if len(batch) == 6 else None
            rollout_visibility = batch[4] if len(batch) == 6 else None
            rollout_gradient = batch[5] if len(batch) == 6 else None
            optimizer.zero_grad(set_to_none=True)
            features_device = features.to(device, non_blocking=True)
            prediction = model(features_device)
            training_weight = weight.to(device, non_blocking=True)
            if args.saturated_mean_threshold > 0.0:
                saturated = (
                    target[:, 0:3].abs().amax(dim=1)
                    >= args.saturated_mean_threshold
                ).to(device, non_blocking=True)
                training_weight = training_weight * torch.where(
                    saturated,
                    torch.full_like(training_weight, args.saturated_mean_weight),
                    torch.ones_like(training_weight),
                )
            loss, _ = prior_loss(
                prediction,
                target.to(device, non_blocking=True),
                training_weight,
                loss_weights,
                args.decoded_residual_targets,
            )
            if rollout_target is not None:
                rollout_weight = render_aware_rollout_weight(
                    training_weight,
                    rollout_visibility.to(device, non_blocking=True),
                    rollout_gradient.to(device, non_blocking=True),
                    args.rollout_visibility_floor,
                    args.rollout_gradient_gain,
                )
                rollout_loss, _ = prior_loss(
                    prediction,
                    rollout_target.to(device, non_blocking=True),
                    rollout_weight,
                    loss_weights,
                    args.decoded_residual_targets,
                )
                loss = loss + args.rollout_loss_weight * rollout_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            if context_adapter_weight is not None:
                restore_context_adapter_base_columns(
                    context_adapter_weight,
                    frozen_base_columns,
                )
            batch_weight = float(training_weight.sum())
            train_total += float(loss.detach()) * batch_weight
            train_denominator += batch_weight
        scheduler.step()
        metrics = evaluate(
            model,
            validation_loader,
            device,
            loss_weights,
            args.decoded_residual_targets,
            args.rollout_loss_weight,
            args.rollout_visibility_floor,
            args.rollout_gradient_gain,
        )
        row = {
            "epoch": epoch,
            "train_loss": train_total / max(1e-6, train_denominator),
            "learning_rate": optimizer.param_groups[-1]["lr"],
            "backbone_learning_rate": (
                optimizer.param_groups[0]["lr"] if backbone_parameters else 0.0
            ),
            **{f"validation_{key}": value for key, value in metrics.items()},
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True))
        checkpoint = {
            "model": model.state_dict(),
            "hidden_dim": args.hidden_dim,
            "layers": args.layers,
            "input_dim": train_data.input_dim,
            "input_contract": train_data.input_contract,
            "input_features": train_data.input_features,
            "stage": args.stage,
            "epoch": epoch,
            "metrics": row,
            "run_config": run_config,
        }
        torch.save(checkpoint, args.output / "last.pt")
        selection_value = row.get(args.selection_metric)
        if selection_value is None:
            raise RuntimeError(
                f"selection metric {args.selection_metric!r} was not produced"
            )
        if selection_value < best_loss:
            best_loss = selection_value
            torch.save(checkpoint, args.output / "best.pt")

    (args.output / "history.json").write_text(
        json.dumps(history, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
