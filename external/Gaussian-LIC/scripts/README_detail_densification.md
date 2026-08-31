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

Run the six controlled groups on the same frozen backend bag and seed:

```bash
cd /root/autodl-tmp/catkin_gaussian/src/Gaussian-LIC
bash scripts/run_detail_densification_matrix.sh \
  /root/autodl-tmp/frozen_contracts/frozen_fast_backend_contract_planes_003.bag
```

The matrix executes:

| Group | Pixel spawn | Reliable clone | DoG clone reweighting |
|---|---:|---:|---:|
| `legacy` | off; historical admission | off | off |
| `coverage_budget` | Top-512, DoG exponent 0 | off | off |
| `detail_spawn` | Top-512, DoG exponent 1 | off | off |
| `reliable_base` | off | E027 full | off |
| `reliable_detail` | off; detail map still computed | E027 full | on |
| `dual` | Top-512, DoG exponent 1 | E027 full | on |

Override result/log directories with `DETAIL_RESULT_ROOT` and `DETAIL_LOG_ROOT`,
and the fixed seed with `DETAIL_SEED`. Use
`DETAIL_GROUPS="coverage_budget detail_spawn"` for selected groups. The runner
restores the YAML file on exit and stops on the first invalid replay.

Detail-map computation is independent from pixel-spawn admission. In
`reliable_detail`, Gaussian-LIC builds the DoG map because clone detail weight is
non-zero while retaining the historical LiDAR extension path; it must not emulate
this case with `detail_spawn=true, top_k=0`, which would reject every new LiDAR
candidate.

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

## First controlled result

On the 3024-frame frozen six-topic R3LIVE replay (seed 20260725, Full=20), all
six valid groups completed 604 keyframes. With the same Top-512 budget,
`detail_spawn` improved novel-view PSNR from 22.06 to 22.48 dB over
`coverage_budget`, while using 303758 instead of 328099 final Gaussians. Thus the
DoG ranking has a positive causal signal. Both remain below the 23.07 dB legacy
admission baseline.

The current best group is `reliable_detail`: 23.14 PSNR, 0.761 SSIM and 0.274
LPIPS versus `reliable_base` at 23.08/0.759/0.280. It keeps the same 7680 clone
budget but increases final map size from 691547 to 721489 and mean extension time
from 16.440 to 33.986 ms/keyframe. Therefore only clone re-ranking is a current
candidate; pixel spawn and dual mode remain disabled by default.

## Important interpretation

The tensors historically named `densify_residual_*` contain screen-space
position-gradient evidence, not direct photometric residuals. Their names are
retained for source/checkpoint compatibility; papers and reports must use the
accurate term.
