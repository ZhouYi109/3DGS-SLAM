# Semantic Gaussian Prior Training

The fixed protocol is:

```text
nuScenes LiDAR/camera geometry pretraining
  -> best.pt
R3LIVE full-optimization Teacher distillation
  -> best.pt
TorchScript export
  -> r3live_distilled.ts
R3LIVE leave-one-sequence-out evaluation
```

## Shard contract

Every `.npz` shard contains:

- `input`: float32 `[N,24]`
- `target`: float32 `[N,14]`
- `weight`: float32 `[N]`
- `source`: scalar string, `nuscenes` or `r3live_teacher`
- `sequence`: scalar string
- `split`: scalar string for nuScenes; R3LIVE split is overridden by the fold

The 24 inputs exactly match the online C++ order:

```text
tanh(xyz / 50), RGB, log1p(depth), object_latent[16], confidence
```

The 14 targets are:

```text
mean residual[3], log-scale residual[3], quaternion[4],
RGB residual[3], opacity-logit residual[1]
```

## Commands

```bash
python scripts/prepare_nuscenes_prior_shards.py \
  --dataroot /root/autodl-fs/datasets/nuscenes \
  --version v1.0-mini \
  --output /root/autodl-fs/datasets/nuscenes/prior_shards

python scripts/prepare_r3live_cross_sequence_split.py \
  --bag-root /root/autodl-tmp/datasets/r3live \
  --output data/r3live_folds.json

python scripts/build_prior_manifest.py \
  --shard-root data/prior_shards \
  --r3live-fold data/r3live_folds.json \
  --fold-name test_hku_campus_seq_00 \
  --output data/fold_hku_campus_seq_00/manifest.json

python scripts/train_semantic_gaussian_prior.py \
  --manifest data/fold_hku_campus_seq_00/manifest.json \
  --stage nuscenes_pretrain \
  --epochs 30 \
  --output checkpoints/nuscenes

python scripts/train_semantic_gaussian_prior.py \
  --manifest data/fold_hku_campus_seq_00/manifest.json \
  --stage r3live_distill \
  --init-checkpoint checkpoints/nuscenes/best.pt \
  --learning-rate 0.0002 \
  --epochs 20 \
  --output checkpoints/r3live

python scripts/export_semantic_gaussian_prior.py \
  --checkpoint checkpoints/r3live/best.pt \
  --output /root/autodl-fs/models/semantic_gaussian_prior/r3live_distilled.ts
```

Run the online student with:

```bash
SEMANTIC_GAUSSIAN_PRIOR_MODEL=/root/autodl-fs/models/semantic_gaussian_prior/r3live_distilled.ts \
PRIOR_RESIDUAL_OPTIMIZATION_ITERS=20 \
bash scripts/run_full_object_memory_r3live.sh \
  prior_fold_test dynamic r3live_prior backend_contract all_dynamic 20260727 object
```

The frozen baseline uses `r3live_paper`, not `r3live_prior`.
