#!/usr/bin/env python3
"""Create leakage-safe leave-one-sequence-out R3LIVE split manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--validation-sequence",
        default="hku_campus_seq_01",
        help="Validation sequence used in every fold unless it is the held-out test.",
    )
    args = parser.parse_args()

    sequences = sorted(path.stem for path in args.bag_root.glob("*.bag"))
    if len(sequences) < 3:
        raise RuntimeError("cross-sequence testing requires at least three rosbags")
    folds = []
    for test in sequences:
        candidates = [name for name in sequences if name != test]
        validation = (
            args.validation_sequence
            if args.validation_sequence in candidates
            else candidates[-1]
        )
        train = [name for name in candidates if name != validation]
        folds.append(
            {
                "name": f"test_{test}",
                "train": train,
                "validation": [validation],
                "test": [test],
            }
        )
    payload = {
        "protocol": "R3LIVE leave-one-sequence-out; no random frame split",
        "bag_root": str(args.bag_root.resolve()),
        "sequences": sequences,
        "folds": folds,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {len(folds)} folds for {len(sequences)} sequences to {args.output}")


if __name__ == "__main__":
    main()

