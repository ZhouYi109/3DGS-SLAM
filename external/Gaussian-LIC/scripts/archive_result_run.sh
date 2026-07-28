#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <cocolic|fastlivo2> [source_result_dir]" >&2
  exit 1
fi

frontend="$1"
source_dir="${2:-$(cd "$(dirname "$0")/.." && pwd)/result}"
repo_dir="$(cd "$(dirname "$0")/.." && pwd)"

case "$frontend" in
  cocolic|fastlivo2) ;;
  *)
    echo "Unsupported frontend: $frontend" >&2
    exit 2
    ;;
esac

if [[ ! -d "$source_dir" ]]; then
  echo "Source result directory not found: $source_dir" >&2
  exit 3
fi

archive_root="$repo_dir/result_runs/$frontend"
mkdir -p "$archive_root"

next_id=1
if compgen -G "$archive_root/run_*" > /dev/null; then
  last_run="$(find "$archive_root" -maxdepth 1 -mindepth 1 -type d -name 'run_*' | sort | tail -n 1)"
  last_num="${last_run##*/run_}"
  next_id=$((10#$last_num + 1))
fi

run_name="$(printf 'run_%03d' "$next_id")"
target_dir="$archive_root/$run_name"
mkdir -p "$target_dir"

if find "$source_dir" -mindepth 1 -print -quit | grep -q .; then
  cp -a "$source_dir"/. "$target_dir"/
else
  touch "$target_dir/.empty_result_dir"
fi

cat > "$target_dir/RUN_INFO.txt" <<EOF
frontend=$frontend
archived_at_utc=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
source_dir=$source_dir
target_dir=$target_dir
EOF

echo "$target_dir"
