#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 {fast|fast-a100|opportunistic|t4} SCRIPT [sbatch arguments...]" >&2
  exit 2
fi

route=$1
script=$2
shift 2

case "$route" in
  fast)
    # Short exploratory jobs.  This account is a member of the aicentre group;
    # unprivileged jobs may be preempted but can use currently idle A40s.
    exec sbatch --partition=aicentre --qos=unprivileged --gres=gpu:a40:1 \
      --time=00:45:00 "$@" "$script"
    ;;
  fast-a100)
    # Same exploratory contract on the currently under-used A100 pool.
    exec sbatch --partition=aicentre-a100 --qos=unprivileged --gres=gpu:a100:1 \
      --time=00:45:00 "$@" "$script"
    ;;
  opportunistic)
    # Allow any GPU type in the default partition.  Use only when exact GPU
    # architecture is not part of the experimental control.
    exec sbatch --partition=prioritized --qos=normal --gres=gpu:1 \
      --time=01:00:00 "$@" "$script"
    ;;
  t4)
    # Confirmation/main-table route: exact architecture for paired comparisons.
    exec sbatch --partition=prioritized --qos=normal --gres=gpu:t4:1 \
      --time=06:00:00 "$@" "$script"
    ;;
  *)
    echo "unknown route: $route" >&2
    exit 2
    ;;
esac
