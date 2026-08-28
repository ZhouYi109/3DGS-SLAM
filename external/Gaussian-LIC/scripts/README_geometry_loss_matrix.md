# Geometry Loss A-F Matrix

This experiment keeps the immutable FAST-LIVO2 six-topic bag, seed, P1 Full=20
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

For C/D/E/F, the reference plane comes from FAST-LIVO2's uncertainty-aware voxel
map through `/planes_for_gs`. Each message is synchronized with image, depth, pose,
points, and adaptive weights. Gaussian-LIC transforms world planes to the camera,
projects their associated current-scan observations, resolves visibility with a
z-buffer, and splats a small disk to form confidence-weighted supervision maps.
The rendered normal is differentiated from rendered metric depth; reference normals
do not require a dense LiDAR depth neighborhood.

The sign-invariant normal loss is

```text
L_normal = mean(1 - abs(dot(n_rendered, n_reference))).
```

The point-to-plane residual and robust loss are

```text
r_plane = dot(n_frontend, P_rendered - plane_center_frontend)
L_plane = mean(sqrt(r_plane^2 + epsilon^2) - epsilon).
```

The frozen input must contain all six topics, including `/planes_for_gs`. Create it
with `scripts/record_fast_backend_contract_r3live.sh` after rebuilding FAST-LIVO2.
The runner deliberately disables depth-derived fallback, so zero frontend-plane
coverage is visible in telemetry instead of silently changing the experiment.

Run from the monorepo root on the validated remote layout:

```bash
export GEOMETRY_RESULT_ROOT=/root/autodl-tmp/experiments/geometry_loss_matrix_YYYYMMDD
export GEOMETRY_LOG_ROOT=/root/autodl-tmp/runtime_logs/geometry_loss_matrix_YYYYMMDD
export GEOMETRY_SEED=20260725
export GEOMETRY_LAMBDA_DEPTH=0.05
export GEOMETRY_LAMBDA_NORMAL=0.05
export GEOMETRY_LAMBDA_POINT_PLANE=0.5

bash scripts/run_geometry_loss_matrix.sh \
  /autodl-fs/data/remote_code_frozen/frozen_fast_backend_contract_planes_003.bag
```

The runner restores `config/r3live_p1.yaml` on exit. Each run stores the resolved
configuration and geometry flags/weights in `wall_times.txt`. `p1_telemetry.csv`
also records plane sample count, valid pixel count, and valid image ratio. Use
`scripts/summarize_p1_budget_runs.py` to include those fields in a CSV summary.
