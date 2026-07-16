# Unified SDPA Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the unavailable FlashAttention2 runtime dependency with one shared PyTorch SDPA backend for every formal CovRA comparison run.

**Architecture:** The attention backend remains a public training-protocol field inherited from `configs/base.yaml` and `configs/dico/base.yaml`. Protocol validation, readiness, installation, launch output versioning, and documentation are updated together so no formal method can silently retain a different backend.

**Tech Stack:** Python 3.10, PyTorch SDPA, Hugging Face Transformers, YAML, pytest.

---

### Task 1: Lock formal configs to SDPA

**Files:**
- Modify: `tests/configs/test_aligned_baseline_configs.py`
- Modify: `configs/base.yaml`
- Modify: `configs/dico/base.yaml`
- Modify: `scripts/protocol_preflight.py`

- [x] Change the formal-config test to require `attn_implementation == "sdpa"` and `require_flash_attention_2 is False`.
- [x] Run the focused test and confirm it fails against the current FlashAttention2 config.
- [x] Change both public base configs and the protocol preflight rule to the SDPA contract.
- [x] Run the focused config and protocol tests and confirm they pass.

### Task 2: Verify loader and environment contract

**Files:**
- Modify: `tests/unit/test_model_loader_attention.py`
- Modify: `tests/scripts/test_e00_readiness.py`
- Modify: `src/dico/model_loader.py`
- Modify: `scripts/e00_readiness.py`
- Modify: `scripts/setup_server_env.sh`
- Modify: `requirements-gpu.txt`

- [x] Add a loader test proving `sdpa` reaches `from_pretrained` without a FlashAttention requirement.
- [x] Add a readiness assertion proving `flash-attn` is not required.
- [x] Run both tests and confirm the readiness assertion fails first.
- [x] Keep the generic loader backend pass-through, remove `flash-attn` from required packages, and stop installing/importing it in the server setup.
- [x] Run the focused tests and shell syntax check.

### Task 3: Isolate SDPA output artifacts

**Files:**
- Modify: `tests/scripts/test_launch_covra.py`
- Modify: `tests/scripts/test_experiment_matrix_report.py`
- Modify: `launch_covra.py`
- Modify: `scripts/experiment_matrix.py`

- [x] Add assertions for `e01_llama3_r8_aligned_sdpa_v4` and `e02_llama3_r8_strict_budget_sdpa_v4`.
- [x] Run the focused tests and confirm they fail on v3 paths.
- [x] Update default launch and experiment-matrix output paths only; preserve methods, seeds, GPUs, batch and accumulation.
- [x] Run launch dry-run and focused tests.

### Task 4: Update operator documentation and regressions

**Files:**
- Modify: `README.md`
- Modify: `docs/audits/covra_server_gpu_acceptance_checklist.md`

- [x] Replace formal FlashAttention installation requirements with the SDPA environment-adaptation statement and new v4 paths.
- [x] Retain historical v3 references only where explicitly labeled as old/failed pilot artifacts.
- [x] Run protocol preflight, static acceptance, CPU smoke and full pytest.
- [x] Confirm the dry-run still expands exactly four methods by three seeds with global batch 64.
