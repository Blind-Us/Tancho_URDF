#!/usr/bin/env bash
set -euo pipefail

# Portable agent/automation entrypoint. All paths are resolved from this file.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASK="TanchoV3-Flat-v0"

cd "$PROJECT_ROOT"

case "${1:-help}" in
  list-envs)
    exec python scripts/list_envs.py
    ;;
  zero-agent)
    exec python -u scripts/zero_agent.py \
      --task="$TASK" --num_envs=8 --headless \
      --max_steps=200 --fail_mean_episode_steps=10 \
      --fail_non_timeout_ratio=1.01
    ;;
  train)
    exec python -u scripts/rsl_rl/train.py \
      --task="$TASK" --num_envs=4096 --headless \
      --max_iterations=1500
    ;;
  tensorboard)
    exec tensorboard --logdir logs/rsl_rl --host 127.0.0.1 --port 6006
    ;;
  play)
    if [[ $# -gt 2 ]]; then
      echo "usage: $0 play [checkpoint.pt]" >&2
      exit 2
    fi
    if [[ $# -eq 2 ]]; then
      exec python scripts/rsl_rl/play.py \
        --task="$TASK" --checkpoint="$2" --num_envs=1
    fi
    exec python scripts/rsl_rl/play.py --task="$TASK" --num_envs=1
    ;;
  help|-h|--help)
    echo "usage: $0 {list-envs|zero-agent|train|tensorboard|play [checkpoint.pt]}"
    ;;
  *)
    echo "unknown command: $1" >&2
    echo "usage: $0 {list-envs|zero-agent|train|tensorboard|play [checkpoint.pt]}" >&2
    exit 2
    ;;
esac
