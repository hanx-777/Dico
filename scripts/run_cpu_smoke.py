#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dico.trainer import train


METHODS = ("lora", "adalora", "gora_public", "gora_bm", "dico_cd_da")


def _config(output_root: Path, method: str) -> dict[str, Any]:
    config: dict[str, Any] = {
        "_project_root": str(ROOT),
        "_disable_accelerate": True,
        "seed": 42,
        "experiment_name": f"cpu_smoke_{method}",
        "method": method,
        "rank": 1,
        "project": {"output_dir": str(output_root)},
        "model": {
            "type": "tiny",
            "name_or_path": "tiny",
            "hidden_size": 8,
            "vocab_size": 128,
            "torch_dtype": "float32",
        },
        "data": {
            "source": "tiny",
            "train_path": "tiny",
            "eval_path": "tiny",
            "max_length": 24,
            "train_limit": 4,
            "eval_limit": 1,
            "shuffle": True,
            "dataset_seed": 42,
            "token_cache_dir": str(output_root / "token_cache"),
        },
        "training": {
            "max_steps": 1,
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "learning_rate": 5e-5,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 5e-4,
            "max_grad_norm": 1.0,
            "warmup_ratio": 0.03,
            "auto_warmup_steps": 0,
            "lr_decay_ratio": 0.1,
            "sample_exposure_policy": "repeat_from_fixed_order_to_max_steps",
            "optimizer_backend": "adamw",
        },
        "lora": {
            "injection": "static",
            "alpha": 2,
            "dropout": 0.0,
            "scaling": "alpha_over_r",
            "adapter_dtype": "float32",
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "max_rank_multiplier": 2,
        },
        "budget": {
            "mode": "equal_trainable_params",
            "warning_threshold": 0.01,
            "enforce_target_ratio": 1.0,
            "enforce_min_ratio": 0.98,
        },
        "calibration": {
            "enabled": method in {"gora_public", "gora_bm", "dico_cd_da"},
            "num_samples": 4,
            "batch_size": 2,
            "seed": 42,
            "shuffle": False,
            "save_dir": str(output_root / "preallocations"),
        },
        "preallocation": {
            "atom_mode": "svd",
            "allocation_method": "covra_full",
            "top_k_atoms": 2,
            "sketch_dim": 4,
            "sketch_oversample": 2,
            "sketch_seed": 42,
            "compute_device": "cpu",
            "allocation_device": "cpu",
            "module_chunk_size": 4,
            "answer_only": True,
            "profile_norm_mode": "streaming_estimate",
            "lambda_cov": 1.0,
            "response_agg_groups": 2,
            "rho": 0.05,
            "sign_split": True,
            "type_scaling": True,
            "log_compression": True,
            "solver": "dp",
            "subspace_init": "direction_anchored",
            "eta": 0.98,
            "r_min_multiplier": 1.0,
            "r_max_multiplier": 2.0,
            "allow_rank_beyond_selected_evidence": True,
        },
        "dico": {
            "version": "cd_da" if method == "dico_cd_da" else method,
            "profile": {"domain": "sketch", "eps": 1e-6},
            "split": {"enabled": True, "mode": "sign", "physical_merge": True},
            "coverage": {"objective": "group_nsw", "eps": 1e-6, "residual_space": "profile"},
            "init": {
                "mode": "direction_anchored" if method == "dico_cd_da" else (
                    "gora_pseudoinverse" if method in {"gora_public", "gora_bm"} else "kaiming_zero_B"
                ),
                "zero_B": True,
            },
        },
        "evaluation": {
            "metric": "gsm8k_accuracy",
            "compute_loss": False,
            "compute_accuracy": False,
            "batch_size": 1,
            "generation_max_new_tokens": 2,
        },
        "runtime": {
            "require_flash_attention_2": False,
            "protocol_scope": "cpu_tiny_smoke",
        },
    }
    if method == "adalora":
        config["training"]["learning_rate"] = 5e-4
        config["adalora"] = {
            "init_rank": 2,
            "target_rank": 1,
            "tinit": 0,
            "tfinal": 0,
            "deltaT": 1,
            "beta1": 0.85,
            "beta2": 0.85,
            "orth_reg_weight": 0.5,
        }
        config["lora"]["injection"] = "adalora"
    if method in {"gora_public", "gora_bm"}:
        config["lora"]["scaling"] = "alpha_over_sqrt_r"
        config["gora"] = {
            "official_commit": "4037d4d6ba67ff88de87f90b943ff4e3a3649b67",
            "gradient_estimation_samples": 4,
            "aggregation": "union_mean",
            "rounding": "moderate",
            "r_ref": 1,
            "r_min": 1,
            "r_max": 2,
            "rank_stabilize": True,
            "dynamic_scaling": True,
            "scale_by_lr": True,
            "init_lr": 0.05,
            "b_lr_multiplier": 16.0,
            "strict_budget_repair": method == "gora_bm",
            "gradient_collection": "official_weight_grad_hook",
            "gradient_offload_device": "cpu",
            "gradient_accumulation_dtype": "float32",
            "clear_gradient_after_offload": True,
        }
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all formal methods through the shared CPU tiny trainer path.")
    parser.add_argument("--output-root", default="outputs/cpu_smoke_v3")
    parser.add_argument("--report", default="reports/cpu_smoke_v3.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    methods: dict[str, Any] = {}
    for method in METHODS:
        metrics = train(_config(output_root, method))
        run_dir = output_root / f"cpu_smoke_{method}"
        manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        methods[method] = {
            "optimizer_steps": int(manifest["optimizer_steps"]),
            "requires_grad": int(manifest["parameter_counts"]["requires_grad"]),
            "checkpoint_exists": (run_dir / "masked_lora_state.pt").exists(),
            "eval_log_exists": (run_dir / "eval_log.jsonl").exists(),
            "manifest_path": str(run_dir / "run_manifest.json"),
            "gpu_status": "IMPLEMENTED_NOT_GPU_RUN",
            "final_metric": metrics.get("final_metric"),
        }
    payload = {
        "status": "IMPLEMENTED_AND_CPU_VERIFIED",
        "gpu_status": "IMPLEMENTED_NOT_GPU_RUN",
        "methods": methods,
    }
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[run_cpu_smoke] {len(methods)} methods passed; report={report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
