#!/usr/bin/env bash
set -euo pipefail
exec "$(dirname "$0")/train_experiment.sh" ablation_multi_plane "$@"
