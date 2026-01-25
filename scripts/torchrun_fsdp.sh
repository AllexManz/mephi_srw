#!/usr/bin/env bash
set -euo pipefail

NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
TRAINING_CONFIG="${TRAINING_CONFIG:-fsdp_gpt2}"
MODEL_CONFIG="${MODEL_CONFIG:-gpt2}"

torchrun --nproc_per_node="${NPROC_PER_NODE}" \
  src/train.py training="${TRAINING_CONFIG}" model="${MODEL_CONFIG}" \
  "${@}"
