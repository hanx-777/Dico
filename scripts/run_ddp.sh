#!/usr/bin/env bash
# Multi-GPU DDP launcher for DiCo experiments.
#
# Usage:
#   NUM_GPUS=3 bash scripts/run_ddp.sh configs/dico/dico_cd_da_r8.yaml
#   NUM_GPUS=3 bash scripts/run_ddp.sh configs/dico/lora_r8.yaml --override training.max_steps=50
#
# Environment variables:
#   NUM_GPUS        Number of GPUs to use (default: 3)
#   MASTER_PORT     Port for inter-process communication (default: 29500)
set -euo pipefail

NUM_GPUS=${NUM_GPUS:-3}
MASTER_PORT=${MASTER_PORT:-29500}
CONFIG=${1:-configs/dico/dico_cd_da_r8.yaml}
shift || true   # remaining args forwarded to run_experiment.py

EXTRA_ARGS=("$@")
if [[ " ${EXTRA_ARGS[*]} " != *"training.per_gpu_batch_size"* ]]; then
    EXTRA_ARGS+=(--override training.per_gpu_batch_size=3)
fi
if [[ " ${EXTRA_ARGS[*]} " != *"training.gradient_accumulation_steps"* ]]; then
    EXTRA_ARGS+=(--override training.gradient_accumulation_steps=7)
fi

# NCCL defaults for containerised / Kubernetes environments.
# Without these, the first ALLGATHER in DDP.__init__ may hang.
# Override by exporting these variables before running this script.
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export NCCL_P2P_LEVEL=${NCCL_P2P_LEVEL:-NVL}
export NCCL_SHM_DISABLE=${NCCL_SHM_DISABLE:-0}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

echo "=== DiCo DDP Launch ==="
echo "  GPUs:        ${NUM_GPUS}"
echo "  Config:      ${CONFIG}"
echo "  NCCL_IB:     ${NCCL_IB_DISABLE}  P2P_LEVEL: ${NCCL_P2P_LEVEL}"
echo "  Extra args:  ${EXTRA_ARGS[*]}"
echo "  Batch note:  DDP fallback default is 3 GPUs × per-device batch 3 × grad accum 7 = global batch 63"
echo "======================="

accelerate launch \
    --config_file configs/accelerate_3gpu.yaml \
    --num_processes="${NUM_GPUS}" \
    --main_process_port="${MASTER_PORT}" \
    scripts/run_experiment.py \
    --config "${CONFIG}" --num-processes 1 "${EXTRA_ARGS[@]}"
