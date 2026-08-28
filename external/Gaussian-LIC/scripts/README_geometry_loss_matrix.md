# Geometry Loss A-F Matrix

This experiment keeps the immutable FAST-LIVO2 five-topic bag, seed, P1 Full=20
budget, appearance weighting, and geometry-capacity policy fixed. It changes only
the geometry loss, except group F which additionally enables the fixed Top-256
reliable-densification policy.

| Group | Appearance | Depth | Normal | Point-to-plane | Reliable densification |
|---|---:|---:|---:|---:|---:|
| A | yes | no | no | no | no |
| B | yes | yes | no | no | no |
| C | yes | no | yes | no | no |
| D | yes | no | no | yes | no |
| E | yes | yes | yes | yes | no |
| F | yes | yes | yes | yes | full |

The first implementation derives reference and rendered normals from metric depth.
Both depth maps are back-projected with the camera intrinsics. A pixel is valid only
when its center and four direct neighbors contain finite positive depth and do not
cross the configured relative-depth discontinuity threshold.

The sign-invariant normal loss is

```text
L_normal = mean(1 - abs(dot(n_rendered, n_reference))).
```

The point-to-plane residual and robust loss are

```text
r_plane = dot(n_reference, P_rendered - P_reference)
L_plane = mean(sqrt(r_plane^2 + epsilon^2) - epsilon).
```

This is a compatibility-first implementation. It does not yet consume the
FAST-LIVO2 voxel-plane normal, and it must not be described as such. Before a full
matrix run, use a short smoke run to verify that the sparse metric-depth contract
provides enough valid normal neighborhoods. The next upgrade is to transport
FAST-LIVO2 plane normals and planarity confidence explicitly.

Run from the monorepo root on the validated remote layout:

```bash
export GEOMETRY_RESULT_ROOT=/root/autodl-tmp/experiments/geometry_loss_matrix_YYYYMMDD
export GEOMETRY_LOG_ROOT=/root/autodl-tmp/runtime_logs/geometry_loss_matrix_YYYYMMDD
export GEOMETRY_SEED=20260725
export GEOMETRY_LAMBDA_DEPTH=0.05
export GEOMETRY_LAMBDA_NORMAL=0.05
export GEOMETRY_LAMBDA_POINT_PLANE=0.5

bash scripts/run_geometry_loss_matrix.sh \
  /autodl-fs/data/remote_code_frozen/frozen_fast_backend_contract_002.bag
```

The runner restores `config/r3live_p1.yaml` on exit. Each run stores the resolved
configuration and geometry flags/weights in `wall_times.txt`. Use
`scripts/summarize_p1_budget_runs.py` to include those fields in a CSV summary.
