#!/usr/bin/env python3
"""Evaluate two TUM trajectories against one shared reference trajectory.

The reference must be an independently measured trajectory.  Do not pass one
SLAM estimate as the reference for another estimate unless it is explicitly
labelled as a proxy reference in the experiment record.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def load_tum(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        values = [float(x) for x in line.split()]
        if len(values) < 8:
            raise ValueError(f"{path}: expected timestamp tx ty tz qx qy qz qw")
        rows.append(values[:8])
    if not rows:
        raise ValueError(f"{path}: no trajectory samples")
    data = np.asarray(rows, dtype=float)
    order = np.argsort(data[:, 0])
    data = data[order]
    return data[:, 0], data[:, 1:4], data[:, 4:8]


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n == 0:
        raise ValueError("zero-norm quaternion")
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def rot_angle_deg(r: np.ndarray) -> float:
    value = np.clip((np.trace(r) - 1.0) * 0.5, -1.0, 1.0)
    return math.degrees(math.acos(value))


def interpolate_reference(times: np.ndarray, positions: np.ndarray, quats: np.ndarray,
                          query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    keep = (query >= times[0]) & (query <= times[-1])
    query = query[keep]
    idx = np.searchsorted(times, query, side="right").clip(1, len(times) - 1)
    left = idx - 1
    alpha = ((query - times[left]) / (times[idx] - times[left])).reshape(-1, 1)
    pos = positions[left] * (1 - alpha) + positions[idx] * alpha
    # Slerp is unnecessary for the scalar RPE-free position interpolation; for
    # orientation, use the nearest reference sample to avoid invalid lerps.
    nearest = np.where(query - times[left] < times[idx] - query, left, idx)
    return pos, quats[nearest]


def align_se3(est: np.ndarray, ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    est_mean = est.mean(axis=0)
    ref_mean = ref.mean(axis=0)
    h = (est - est_mean).T @ (ref - ref_mean)
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T
    t = ref_mean - r @ est_mean
    return (r @ est.T).T + t, r


def evaluate(estimate: Path, reference: Path, rpe_delta: float) -> dict:
    te, pe, qe = load_tum(estimate)
    tr, pr, qr = load_tum(reference)
    common = (te >= tr[0]) & (te <= tr[-1])
    te, pe, qe = te[common], pe[common], qe[common]
    if len(te) < 2:
        raise ValueError(f"{estimate}: insufficient time overlap with {reference}")
    pref, qref = interpolate_reference(tr, pr, qr, te)
    aligned, _ = align_se3(pe, pref)
    errors = np.linalg.norm(aligned - pref, axis=1)
    target = te + rpe_delta
    valid = target <= te[-1]
    rpe_t = []
    rpe_r = []
    for i in np.flatnonzero(valid):
        j = np.searchsorted(te, target[i])
        if j >= len(te):
            continue
        rpe_t.append(np.linalg.norm((aligned[j] - aligned[i]) - (pref[j] - pref[i])))
        rpe_r.append(rot_angle_deg(quat_to_rot(qe[j]) @ quat_to_rot(qe[i]).T))
    return {
        "estimate": str(estimate),
        "reference": str(reference),
        "samples": int(len(te)),
        "duration_s": float(te[-1] - te[0]),
        "ate_rmse_m": float(np.sqrt(np.mean(errors ** 2))),
        "ate_mean_m": float(np.mean(errors)),
        "ate_median_m": float(np.median(errors)),
        "ate_max_m": float(np.max(errors)),
        "rpe_delta_s": rpe_delta,
        "rpe_trans_rmse_m": float(np.sqrt(np.mean(np.asarray(rpe_t) ** 2))) if rpe_t else None,
        "rpe_rot_rmse_deg": float(np.sqrt(np.mean(np.asarray(rpe_r) ** 2))) if rpe_r else None,
    }


def closure(path: Path) -> dict:
    times, pos, quat = load_tum(path)
    return {
        "estimate": str(path),
        "samples": int(len(times)),
        "duration_s": float(times[-1] - times[0]),
        "start_end_translation_m": float(np.linalg.norm(pos[-1] - pos[0])),
        "start_end_rotation_deg": rot_angle_deg(quat_to_rot(quat[-1]) @ quat_to_rot(quat[0]).T),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--estimate", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--rpe-delta", type=float, default=1.0)
    parser.add_argument("--closure-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = closure(args.estimate) if args.closure_only else evaluate(args.estimate, args.reference, args.rpe_delta)
    text = json.dumps(result, indent=2, ensure_ascii=True) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
