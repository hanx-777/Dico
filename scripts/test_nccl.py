#!/usr/bin/env python
"""Quick NCCL connectivity test.  Run via accelerate to verify GPU-to-GPU
communication before starting the full training pipeline.

Usage (single machine, 3 GPUs):
    NCCL_IB_DISABLE=1 NCCL_DEBUG=INFO accelerate launch \
        --num_processes=3 --mixed_precision=no \
        scripts/test_nccl.py

Expected output (success):
    [rank 0] all_reduce OK  tensor([3.])  on cuda:0
    [rank 1] all_reduce OK  tensor([3.])  on cuda:1
    [rank 2] all_reduce OK  tensor([3.])  on cuda:2

If this script hangs or crashes, NCCL cannot communicate between your GPUs.
Common fixes for Kubernetes / container environments:
    export NCCL_IB_DISABLE=1          # disable InfiniBand
    export NCCL_P2P_LEVEL=NVL         # prefer NVLink
    export NCCL_SHM_DISABLE=0         # enable shared memory
    export NCCL_SOCKET_IFNAME=eth0    # explicit network interface
    export NCCL_P2P_DISABLE=1         # last resort: disable GPU P2P entirely
"""
from __future__ import annotations
import os, sys, time

def main():
    try:
        import torch
        import torch.distributed as dist
    except ImportError:
        print("torch not available", file=sys.stderr)
        sys.exit(1)

    # Accelerate / torchrun set these automatically
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if world_size <= 1:
        print("Only 1 process detected. Launch with accelerate or torchrun for multi-GPU test.")
        sys.exit(0)

    # Initialize process group if not already done
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    # Simple all_reduce: each rank contributes 1.0, result should be world_size
    t = torch.ones(1, device=device)
    print(f"[rank {rank}] starting all_reduce on {device} ...", flush=True)
    start = time.time()
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize(device)
    elapsed = time.time() - start

    expected = float(world_size)
    if abs(t.item() - expected) < 1e-6:
        print(f"[rank {rank}] all_reduce OK  {t}  on {device}  ({elapsed*1000:.1f}ms)", flush=True)
    else:
        print(f"[rank {rank}] all_reduce FAILED: expected {expected}, got {t.item()}", flush=True)
        sys.exit(1)

    # Barrier to sync before exit
    dist.barrier()
    if rank == 0:
        print(f"\nNCCL connectivity test PASSED ({world_size} GPUs)", flush=True)

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
