# Semantic Feature Alignment

This helper validates static alignment between:

- `Gaussian-LIC` exported `point_cloud.ply`
- `Semantic Gaussians` exported `*.pt`

Current recommended use is offline verification before wiring semantics into the online SLAM loop.

## What It Checks

The script inspects:

- PLY vertex count from `point_cloud.ply`
- semantic feature rows from `.pt["feat"]`
- semantic mask length and `True` count from `.pt["mask_full"]`

It then classifies alignment into one of:

- `dense_full`
  `feat.shape[0] == ply_vertex_count`
- `masked_sparse`
  `feat.shape[0] == mask_full.sum()` and `mask_full.shape[0] == ply_vertex_count`
- `mask_length_mismatch`
- `feat_length_mismatch`

## Consumer Bundle

For downstream SLAM-3DGS integration, use:

```bash
python external/Gaussian-LIC/scripts/semantic_feature_consumer.py \
  --point-cloud /path/to/point_cloud.ply \
  --semantic-pt /path/to/0.pt \
  --output-dir /path/to/semantic_bundle
```

It writes:

- `semantic_feat.npy`
- `semantic_mask.npy`
- `semantic_alignment_summary.json`
- `semantic_feature_bundle.json`

This is the recommended first bridge into the main project because it turns the original
`.pt` into stable dense arrays plus explicit metadata.

## Preview Export

For an unambiguous readback test, generate a colored preview PLY:

```bash
python external/Gaussian-LIC/scripts/semantic_feature_preview.py \
  --point-cloud /path/to/point_cloud.ply \
  --semantic-feat /path/to/semantic_feat.npy \
  --semantic-mask /path/to/semantic_mask.npy \
  --output-dir /path/to/semantic_preview
```

It writes:

- `semantic_preview_colored.ply`
- `semantic_preview_summary.json`

The color is a PCA projection of the semantic feature vectors and is only meant for
readback validation and qualitative inspection.

## Standard Sidecar Loader

Use the standard loader in downstream tools:

```python
from semantic_bundle_loader import load_semantic_bundle

bundle = load_semantic_bundle("/path/to/result/semantic_sidecar")
print(bundle.num_gaussians, bundle.semantic_dim, bundle.active_count)
```

## Top-K Query Prototype

Use the query prototype to test whether semantic retrieval is meaningful:

```bash
python external/Gaussian-LIC/scripts/semantic_query_tool.py \
  --bundle-dir /path/to/result/semantic_sidecar \
  --query-npy /path/to/query_embedding.npy \
  --topk 10 \
  --out-json /path/to/query_topk.json
```

This is the recommended next step before adding any semantic optimization term.

## Usage

```bash
python external/Gaussian-LIC/scripts/semantic_feature_alignment.py \
  --point-cloud /path/to/point_cloud.ply \
  --semantic-pt /path/to/0.pt \
  --out-json /path/to/semantic_alignment_summary.json
```

## Recommended Current Inputs

For the current remote smoke tests, the two validated pairs are:

- `samclip`
  - `point_cloud.ply`
    `/root/autodl-tmp/semantic-gaussians/output/semantic_smoke_001/point_cloud/iteration_10/point_cloud.ply`
  - `semantic .pt`
    `/root/autodl-tmp/semantic-gaussians/output/fusion_smoke_samclip20/0.pt`

- `vlpart`
  - `point_cloud.ply`
    `/root/autodl-tmp/semantic-gaussians/output/semantic_smoke_001/point_cloud/iteration_10/point_cloud.ply`
  - `semantic .pt`
    `/root/autodl-tmp/semantic-gaussians/output/fusion_smoke_vlpart20/0.pt`

## Integration Intent

This module is meant to be the first bridge into the SLAM-3DGS main project:

1. verify that semantic rows match Gaussian order
2. materialize dense per-Gaussian semantic features when needed
3. keep online integration out of the critical path until static alignment is proven stable
