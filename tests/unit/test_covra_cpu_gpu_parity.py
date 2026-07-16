from __future__ import annotations

import pytest
import torch

from dico.covra_core import build_response_block, greedy_conditional_coverage


@pytest.mark.skipif(not torch.cuda.is_available(), reason="IMPLEMENTED_NOT_GPU_RUN: CUDA parity requires target GPU")
def test_covra_conditional_selection_matches_cpu_gpu_fp32():
    generator = torch.Generator().manual_seed(42)
    responses = [torch.randn(32, generator=generator, dtype=torch.float32) for _ in range(8)]
    cpu_blocks = [build_response_block("m", idx, value, rho=0.05) for idx, value in enumerate(responses)]
    gpu_blocks = [
        build_response_block("m", idx, value.cuda(), rho=0.05)
        for idx, value in enumerate(responses)
    ]

    cpu = greedy_conditional_coverage(cpu_blocks, r_max=6)
    gpu = greedy_conditional_coverage(gpu_blocks, r_max=6)

    assert gpu.selected_indices == cpu.selected_indices
    assert gpu.marginal_gains == pytest.approx(cpu.marginal_gains, abs=1e-6, rel=1e-5)
