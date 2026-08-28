# Detail-Guided Dual-Channel Densification

This phase keeps FAST-LIVO2 as the metric geometry source and adds two bounded
Gaussian-map capacity actions:

1. **Pixel spawn** computes a robustly normalized
   `DoG(real)-DoG(render)` map. Projected LiDAR candidates are scored by missing
   detail, rendered alpha hole, frontend geometry reliability and available
   semantic static confidence. Image-cell NMS and a fixed Top-K budget prevent
   spatial concentration and unbounded map growth.
2. **Reliable clone** keeps the existing multi-view screen-space-gradient EMA,
   visibility-count and variance score. An optional projected missing-detail
   gate reweights that score with a non-zero floor, so a weak current frame
   cannot erase accumulated evidence.

Both paths are disabled in `config/r3live_p1.yaml` by default. Therefore the
existing P1 and geometry A-F baselines retain their old behavior.

## Controlled matrix

Run the five controlled groups on the same frozen backend bag and seed:

```bash
cd /root/autodl-tmp/catkin_gaussian/src/Gaussian-LIC
bash scripts/run_detail_densification_matrix.sh \
  /autodl-fs/data/remote_code_frozen/frozen_fast_backend_contract_002.bag
```

The matrix executes:

| Group | Pixel spawn | Detail-reweighted reliable clone |
|---|---:|---:|
| `legacy` | off; historical unbounded admission | off |
| `coverage_budget` | Top-512, DoG exponent 0 | off |
| `detail_spawn` | Top-512, DoG exponent 1 | off |
| `reliable_clone` | budget 0; detail map only | on |
| `dual` | Top-512 | on |

Override the result directory with `DETAIL_RESULT_ROOT`, and the fixed seed
with `DETAIL_SEED`. The runner restores the YAML file on exit.

## Telemetry

`p1_telemetry.csv` now includes:

- `spawn_visible_candidates`
- `spawn_detail_candidates`
- `spawn_inserted`
- `spawn_score_mean`, `spawn_score_max`
- `reliable_clone_inserted`

Compare PSNR/SSIM/LPIPS, final Gaussian count, `extend_ms`, `optimize_ms`, spawn
acceptance ratio and reliable-clone count. A quality gain is not accepted unless it is
reported together with map size and runtime.

## Important interpretation

The tensors historically named `densify_residual_*` contain screen-space
position-gradient evidence, not direct photometric residuals. Their names are
retained for source/checkpoint compatibility; papers and reports must use the
accurate term.
