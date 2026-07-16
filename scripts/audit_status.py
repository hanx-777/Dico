#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class RequirementRow:
    id: str
    requirement: str
    status: str
    evidence: tuple[str, ...]
    remaining: str

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["evidence"] = list(self.evidence)
        return payload


def _counter(rows: Sequence[RequirementRow]) -> dict[str, int]:
    return dict(Counter(row.status for row in rows))


def _report_payload(title: str, rows: Sequence[RequirementRow]) -> dict[str, object]:
    return {
        "title": title,
        "requirements_by_status": _counter(rows),
        "requirements": [row.to_dict() for row in rows],
    }


def method_rows() -> list[RequirementRow]:
    return [
        RequirementRow(
            id="covra_full",
            requirement="Final CovRA conditional marginal coverage path is implemented and selected by main config.",
            status="IMPLEMENTED_NOT_GPU_RUN",
            evidence=(
                "src/dico/covra_core.py",
                "src/dico/preallocation.py",
                "configs/dico/dico_cd_da_r8.yaml",
                "tests/unit/test_covra_core.py",
                "tests/unit/test_covra_final_preallocation_path.py",
            ),
            remaining="Run E00/E01 GPU pilot before claiming training-scale completion.",
        ),
        RequirementRow(
            id="covra_independent",
            requirement="CovRA-I keeps CovRA candidates/protocol but replaces conditional value with independent energy curve.",
            status="IMPLEMENTED_NOT_GPU_RUN",
            evidence=(
                "configs/ablations/covra_independent.yaml",
                "src/dico/preallocation.py",
                "tests/unit/test_covra_core.py",
                "tests/unit/test_covra_final_preallocation_path.py",
            ),
            remaining="Needs formal GPU run for main-text claims.",
        ),
        RequirementRow(
            id="covra_module_scalar",
            requirement="CovRA-M collapses direction-level value to module energy plus fixed non-increasing rank template.",
            status="IMPLEMENTED_NOT_GPU_RUN",
            evidence=(
                "configs/ablations/covra_module_scalar.yaml",
                "src/dico/covra_core.py",
                "src/dico/preallocation.py",
                "tests/unit/test_covra_core.py",
                "tests/unit/test_covra_final_preallocation_path.py",
            ),
            remaining="Template sensitivity still belongs in appendix after GPU runs.",
        ),
        RequirementRow(
            id="all_required_ablations",
            requirement="Required CovRA ablations are available as configs/scripts.",
            status="IMPLEMENTED_NOT_GPU_RUN",
            evidence=("configs/ablations/", "scripts/experiment_matrix.py", "tests/configs/test_v03_slim_layout.py"),
            remaining="Real training not executed in this environment.",
        ),
        RequirementRow(
            id="ablation_single_factor_metadata",
            requirement=(
                "Required CovRA mechanism ablations carry machine-readable id, reference config, "
                "mechanism group, single-factor field, and expected-difference metadata so duplicate "
                "or ambiguous controls are caught before formal runs."
            ),
            status="IMPLEMENTED_NOT_GPU_RUN",
            evidence=(
                "configs/ablations/*.yaml",
                "src/dico/config.py",
                "tests/configs/test_v03_slim_layout.py",
            ),
            remaining="Metadata defines the planned controls; real ablation claims still require E04-E08 GPU runs.",
        ),
        RequirementRow(
            id="dp_solver",
            requirement="Real-cost integer-rank DP solver exists and is tested against brute force.",
            status="IMPLEMENTED_NOT_GPU_RUN",
            evidence=("src/dico/rank_budget.py", "tests/unit/test_budget.py"),
            remaining="No remaining CPU-side blocker; GPU artifacts still needed for paper tables.",
        ),
        RequirementRow(
            id="direction_init",
            requirement="Direction-anchored initialization consumes selected CovRA directions and keeps initial delta W zero.",
            status="IMPLEMENTED_NOT_GPU_RUN",
            evidence=("src/dico/init.py", "tests/unit/test_dico_da_init_trainer_integration.py"),
            remaining="Large-model E00 smoke required.",
        ),
        RequirementRow(
            id="legacy_isolation",
            requirement=(
                "Historical taxonomy/procurement/beta/BH-FDR fields do not drive final CovRA path by default, "
                "and final CovRA tiny runs do not emit legacy artifacts."
            ),
            status="IMPLEMENTED_NOT_GPU_RUN",
            evidence=(
                "configs/base.yaml",
                "configs/dico/base.yaml",
                "src/dico/trainer.py",
                "tests/configs/test_v03_slim_layout.py",
                "tests/unit/test_dico_da_init_trainer_integration.py",
            ),
            remaining="Legacy compatibility path remains for old configs and must stay labelled legacy.",
        ),
    ]


def protocol_rows() -> list[RequirementRow]:
    return [
        RequirementRow(
            id="single_gpu_global_batch_64",
            requirement="Default 3xA800 scheduling runs three single-GPU seeds with global batch 64.",
            status="READY_DRY_RUN",
            evidence=("scripts/platform_train.py", "tests/scripts/test_platform_train.py"),
            remaining="Run E00 on actual A800 GPU to verify memory and speed.",
        ),
        RequirementRow(
            id="e00_readiness_report",
            requirement="Pre-E00 readiness report checks protocol preflight, data/model references, runtime dependencies, local baseline statuses, launchers, and optional GPU-count gate.",
            status="READY_DRY_RUN",
            evidence=("scripts/e00_readiness.py", "tests/scripts/test_e00_readiness.py"),
            remaining="Run with --require-gpu-count 3 on the target A800 server before launching E00.",
        ),
        RequirementRow(
            id="ddp_fallback_global_batch_63",
            requirement="DDP fallback uses world size 3 and global batch 63 only after single-GPU OOM.",
            status="READY_DRY_RUN",
            evidence=("scripts/run_ddp.sh", "README.md", "tests/unit/test_ddp_compat.py"),
            remaining="DDP fallback has not been GPU-executed here.",
        ),
        RequirementRow(
            id="r32_pilot_config",
            requirement="High-budget r32 pilot config exists with r_max=128 and K/sketch_dim large enough for startup validation.",
            status="READY_DRY_RUN",
            evidence=("configs/dico/dico_cd_da_r32_pilot.yaml", "scripts/experiment_matrix.py", "tests/configs/test_v03_slim_layout.py", "tests/scripts/test_experiment_matrix_report.py"),
            remaining="Formal r32 training remains recommended-only until E10 GPU pilot passes.",
        ),
        RequirementRow(
            id="ddp_data_loading_manifest",
            requirement="Run manifest records DDP data sharding, sampler type, drop_last, dataloader length, and last accumulation behavior.",
            status="IMPLEMENTED_NOT_GPU_RUN",
            evidence=("src/dico/trainer.py", "tests/unit/test_ddp_trainer_integration.py", "tests/unit/test_dico_da_init_trainer_integration.py"),
            remaining="Real DDP fallback still needs E00/A800 execution to confirm runtime behavior.",
        ),
        RequirementRow(
            id="strict_config_schema",
            requirement="Experiment configs and command-line overrides fail fast on unknown fields instead of silently carrying typos.",
            status="IMPLEMENTED_NOT_GPU_RUN",
            evidence=("src/dico/config.py", "scripts/run_experiment.py", "tests/unit/test_config.py", "tests/configs/test_v03_slim_layout.py"),
            remaining="Schema only proves known-field coverage; it does not prove every known field is semantically consumed.",
        ),
        RequirementRow(
            id="protocol_preflight",
            requirement="Formal configs can be preflight-checked for protocol invariants before launching E00/E01 runs.",
            status="READY_DRY_RUN",
            evidence=("scripts/protocol_preflight.py", "tests/scripts/test_protocol_preflight.py"),
            remaining="Run the preflight report on the target A800 checkout before E00, then archive reports/protocol_preflight.{json,md}.",
        ),
        RequirementRow(
            id="source_control_and_command_manifest",
            requirement="Run manifest records git commit/branch/dirty state, command argv, current working directory, and Python executable.",
            status="IMPLEMENTED_NOT_GPU_RUN",
            evidence=("src/dico/trainer.py", "tests/unit/test_dico_da_init_trainer_integration.py"),
            remaining="E00 should confirm these source and command fields on the real launch path used for A800 experiments.",
        ),
        RequirementRow(
            id="model_revision_manifest",
            requirement=(
                "Run manifest records model/tokenizer revision values; missing locked revisions are explicitly marked UNRESOLVED."
            ),
            status="IMPLEMENTED_NOT_GPU_RUN",
            evidence=(
                "src/dico/trainer.py",
                "scripts/validate_run_manifest.py",
                "tests/unit/test_dico_da_init_trainer_integration.py",
                "tests/scripts/test_validate_run_manifest.py",
            ),
            remaining="E00/E01 should use locked Llama3 model/tokenizer revisions once the exact source revision is confirmed.",
        ),
        RequirementRow(
            id="resolved_config_manifest",
            requirement="Run manifest links to the fully resolved config and records its SHA256 hash.",
            status="IMPLEMENTED_NOT_GPU_RUN",
            evidence=("src/dico/trainer.py", "tests/unit/test_dico_da_init_trainer_integration.py"),
            remaining="E00 should confirm the hash is written for full Llama3/A800 artifacts.",
        ),
        RequirementRow(
            id="seed_manifest",
            requirement="Run manifest records base, model/init, rank-local training, calibration, and sketch seeds.",
            status="IMPLEMENTED_NOT_GPU_RUN",
            evidence=("src/dico/trainer.py", "tests/unit/test_dico_da_init_trainer_integration.py"),
            remaining="E00 should confirm seed fields under single-GPU and DDP fallback runtime paths.",
        ),
        RequirementRow(
            id="unresolved_protocol_fields",
            requirement="Provisional/unresolved protocol choices such as rho, response groups, and r_min are machine-readable in configs.",
            status="IMPLEMENTED_NOT_GPU_RUN",
            evidence=("configs/base.yaml", "configs/dico/base.yaml", "src/dico/config.py", "tests/configs/test_v03_slim_layout.py"),
            remaining="Sensitivity runs must still be executed before treating these choices as settled.",
        ),
        RequirementRow(
            id="data_calibration_manifest",
            requirement="Run manifest records train/eval split counts, hashes, and calibration sample ids/hashes.",
            status="IMPLEMENTED_NOT_GPU_RUN",
            evidence=("src/dico/data.py", "src/dico/trainer.py", "tests/unit/test_dico_da_init_trainer_integration.py"),
            remaining="Full model/data manifest hashes require E00/E01 real dataset runs.",
        ),
        RequirementRow(
            id="runtime_hardware_manifest",
            requirement="Run manifest records Python/PyTorch/CUDA runtime plus visible GPU model names and peak CUDA memory.",
            status="IMPLEMENTED_NOT_GPU_RUN",
            evidence=("src/dico/trainer.py", "tests/unit/test_dico_da_init_trainer_integration.py"),
            remaining="E00 should confirm the visible GPU list reports the expected A800 devices.",
        ),
        RequirementRow(
            id="dependency_versions_manifest",
            requirement="Run manifest records key dependency versions for Python, torch, transformers, accelerate, datasets, numpy/scipy/sklearn/pandas, and optional vLLM.",
            status="IMPLEMENTED_NOT_GPU_RUN",
            evidence=("src/dico/trainer.py", "tests/unit/test_dico_da_init_trainer_integration.py"),
            remaining="E00 should confirm GPU-server package versions, especially transformers/accelerate/vLLM.",
        ),
        RequirementRow(
            id="adapter_fp32_protocol",
            requirement="LoRA adapter parameters stay FP32 even when the frozen base model is configured as BF16.",
            status="IMPLEMENTED_NOT_GPU_RUN",
            evidence=("configs/base.yaml", "configs/dico/base.yaml", "src/dico/trainer.py", "tests/unit/test_dico_da_init_trainer_integration.py", "tests/configs/test_v03_slim_layout.py"),
            remaining="E00 should confirm the same dtype split on the real Llama3/A800 path.",
        ),
        RequirementRow(
            id="optimizer_lr_group_manifest",
            requirement="Run manifest records optimizer identity, AdamW betas/eps, and actual LR parameter groups.",
            status="IMPLEMENTED_NOT_GPU_RUN",
            evidence=("src/dico/trainer.py", "tests/unit/test_dico_da_init_trainer_integration.py"),
            remaining="External baselines with multiple LR groups must be rechecked once their wrappers are implemented.",
        ),
        RequirementRow(
            id="optimizer_state_estimate_manifest",
            requirement="Run manifest records estimated optimizer-state memory for trainable adapter parameters.",
            status="IMPLEMENTED_NOT_GPU_RUN",
            evidence=("src/dico/trainer.py", "tests/unit/test_dico_da_init_trainer_integration.py"),
            remaining="E00 should compare this estimate with observed A800 peak memory and any optimizer variant used by external baselines.",
        ),
        RequirementRow(
            id="scheduler_protocol_manifest",
            requirement="Run manifest records cosine-with-warmup scheduler, warmup ratio/steps, LR floor ratio, and optimizer-step source.",
            status="IMPLEMENTED_NOT_GPU_RUN",
            evidence=("src/dico/trainer.py", "tests/unit/test_dico_da_init_trainer_integration.py"),
            remaining="E00 should confirm scheduler fields under the real Llama3/A800 single-GPU and any DDP fallback paths.",
        ),
        RequirementRow(
            id="module_budget_manifest",
            requirement="Run manifest records per-module d_in/d_out rank cost, initial/final rank, and initial/final active LoRA parameter counts.",
            status="IMPLEMENTED_NOT_GPU_RUN",
            evidence=("src/dico/trainer.py", "tests/unit/test_dico_da_init_trainer_integration.py"),
            remaining="E00 should confirm per-module costs on the real Llama3 q/k/v/o module set.",
        ),
        RequirementRow(
            id="gradient_clipping_protocol",
            requirement="Training protocol applies max_grad_norm before optimizer.step() and records the clipping setting.",
            status="IMPLEMENTED_NOT_GPU_RUN",
            evidence=("configs/base.yaml", "configs/dico/base.yaml", "src/dico/trainer.py", "tests/unit/test_dico_da_init_trainer_integration.py"),
            remaining="E00 should confirm grad clipping and logged grad norms on the real Llama3/A800 training path.",
        ),
        RequirementRow(
            id="evaluation_artifact_manifest",
            requirement="Run manifest records evaluation prediction artifacts with path, SHA256, row count, and required raw/processed/score fields.",
            status="IMPLEMENTED_NOT_GPU_RUN",
            evidence=(
                "src/dico/trainer.py",
                "src/dico/evaluator.py",
                "tests/unit/test_dico_da_init_trainer_integration.py",
                "tests/unit/test_evaluator_accuracy.py",
                "tests/unit/test_evaluator_humaneval.py",
            ),
            remaining="Full Llama3 GSM8K/HumanEval artifacts still need E00/E01 GPU execution.",
        ),
        RequirementRow(
            id="timing_and_throughput_manifest",
            requirement="Run manifest records calibration/allocation/initialization/training time and train tokens/s.",
            status="IMPLEMENTED_NOT_GPU_RUN",
            evidence=("src/dico/trainer.py", "tests/unit/test_dico_da_init_trainer_integration.py"),
            remaining="A800 E00 must confirm these fields under real GPU memory and dataloader behavior.",
        ),
        RequirementRow(
            id="method_artifact_manifest",
            requirement="Run manifest records rank allocation, budget, initialization, diagnostics, and utility JSON artifacts with path, SHA256, size, and format.",
            status="IMPLEMENTED_NOT_GPU_RUN",
            evidence=("src/dico/trainer.py", "tests/unit/test_dico_da_init_trainer_integration.py"),
            remaining="E00 should confirm these artifact hashes for full Llama3/CovRA outputs.",
        ),
        RequirementRow(
            id="run_artifact_manifest",
            requirement="Run manifest records train/eval logs, metrics, evaluation protocol, and run summary with path, SHA256, size, format, and JSONL row counts.",
            status="IMPLEMENTED_NOT_GPU_RUN",
            evidence=("src/dico/trainer.py", "tests/unit/test_dico_da_init_trainer_integration.py"),
            remaining="E00 should confirm run log artifacts for the real A800 launch path.",
        ),
        RequirementRow(
            id="run_manifest_validation",
            requirement="Completed run manifests can be independently validated for protocol fields, budget consistency, artifact hashes, and prediction row schema.",
            status="READY_DRY_RUN",
            evidence=("scripts/validate_run_manifest.py", "tests/scripts/test_validate_run_manifest.py"),
            remaining="Run this validator on E00/E01 artifacts after real A800 runs and archive reports/run_manifest_validation.{json,md}.",
        ),
        RequirementRow(
            id="checkpoint_artifact_manifest",
            requirement="Run manifest records the final LoRA adapter checkpoint artifact with path, SHA256, size, format, and final-checkpoint-only rule.",
            status="IMPLEMENTED_NOT_GPU_RUN",
            evidence=("src/dico/trainer.py", "tests/unit/test_dico_da_init_trainer_integration.py"),
            remaining="E00 should confirm the recorded checkpoint artifact can be restored for the real Llama3/A800 run.",
        ),
        RequirementRow(
            id="checkpoint_restore",
            requirement="Adapter checkpoint saved by trainer can be restored into a fresh LoRA-injected model.",
            status="IMPLEMENTED_NOT_GPU_RUN",
            evidence=("src/dico/lora_checkpoint.py", "src/dico/trainer.py", "tests/unit/test_lora_checkpoint.py", "tests/unit/test_dico_da_init_trainer_integration.py"),
            remaining="Large-model checkpoint restore should be smoke-tested after E00 writes a real A800 artifact.",
        ),
        RequirementRow(
            id="checkpoint_selection_protocol",
            requirement="Evaluation protocol records final-checkpoint-only selection and no validation/test metric selection.",
            status="IMPLEMENTED_NOT_GPU_RUN",
            evidence=("src/dico/trainer.py", "tests/unit/test_dico_da_init_trainer_integration.py"),
            remaining="E00 should confirm the same evaluation_protocol.json is written for real Llama3 runs.",
        ),
        RequirementRow(
            id="gsm8k_greedy",
            requirement="GSM8K main evaluation uses greedy decoding and archives raw/processed predictions.",
            status="IMPLEMENTED_NOT_GPU_RUN",
            evidence=("src/dico/evaluator.py", "tests/unit/test_evaluator_accuracy.py"),
            remaining="Full GSM8K eval not run on Llama3 checkpoint in this environment.",
        ),
        RequirementRow(
            id="humaneval_official_pass_at_1",
            requirement="HumanEval pass@1 uses official unbiased estimator and archives completions.",
            status="IMPLEMENTED_NOT_GPU_RUN",
            evidence=("src/dico/evaluator.py", "tests/unit/test_evaluator_humaneval.py"),
            remaining="Official HumanEval dataset and full evaluation not run here.",
        ),
        RequirementRow(
            id="mtbench_local",
            requirement="MTBench-local judge protocol is locked and a local judge executor can score archived answer sets.",
            status="IMPLEMENTED_NOT_GPU_RUN",
            evidence=(
                "configs/base.yaml",
                "src/dico/trainer.py",
                "src/dico/mtbench_local.py",
                "scripts/mtbench_local_judge.py",
                "tests/unit/test_mtbench_local.py",
                "tests/scripts/test_mtbench_local_judge.py",
            ),
            remaining="Real MTBench answer sets have not been scored with the target local 70B judge in this environment.",
        ),
        RequirementRow(
            id="run_manifest",
            requirement="Each completed run writes JSON manifest and Markdown summary; manifests can be aggregated.",
            status="IMPLEMENTED_NOT_GPU_RUN",
            evidence=("src/dico/trainer.py", "scripts/collect_run_manifests.py", "tests/scripts/test_collect_run_manifests.py"),
            remaining="Aggregation requires real run_manifest.json files from GPU experiments.",
        ),
    ]


def status_rows() -> list[RequirementRow]:
    return [
        RequirementRow(
            id="gpu_e00_pilot",
            requirement="Run E00 single-GPU LoRA/CovRA pilot on A800 and record memory/batch/steps.",
            status="NOT_EXECUTED",
            evidence=("reports/experiment_matrix.md",),
            remaining="Requires access to the target 3xA800 runtime and model/data paths.",
        ),
        RequirementRow(
            id="GoRA-public",
            requirement="Official GoRA-public baseline wrapper/protocol is verified separately from GoRA-BM.",
            status="IMPLEMENTED_NOT_GPU_RUN",
            evidence=(
                "src/dico/gora.py",
                "src/dico/trainer.py",
                "configs/dico/gora_public_r8.yaml",
                "tests/unit/test_gora_aligned.py",
                "tests/unit/test_gora_trainer_integration.py",
            ),
            remaining=(
                "Direct gradient/allocation/init semantics are CPU/tiny verified against locked commit; "
                "A800 E00 and unavailable official final benchmark scripts remain unresolved."
            ),
        ),
        RequirementRow(
            id="EVA",
            requirement="EVA baseline wrapper/protocol is available and budget-audited.",
            status="BLOCKED_BY_UNRESOLVED_PROTOCOL",
            evidence=("src/dico/baselines.py", "reports/baseline_status.md"),
            remaining="Need official implementation/version and budget matching details.",
        ),
        RequirementRow(
            id="AdaLoRA",
            requirement="AdaLoRA baseline is implemented under the shared protocol.",
            status="IMPLEMENTED_NOT_GPU_RUN",
            evidence=(
                "src/dico/adalora.py",
                "configs/dico/adalora_r8.yaml",
                "src/dico/baselines.py",
                "tests/unit/test_adalora.py",
                "tests/unit/test_adalora_trainer_integration.py",
            ),
            remaining="Official commit semantics are covered by CPU/tiny tests; GPU E01 run is still required.",
        ),
        RequirementRow(
            id="MTBench-local executor",
            requirement="Local MTBench judge actually scores answer sets with locked judge config.",
            status="IMPLEMENTED_NOT_GPU_RUN",
            evidence=("src/dico/mtbench_local.py", "scripts/mtbench_local_judge.py", "tests/unit/test_mtbench_local.py"),
            remaining="Run scripts/mtbench_local_judge.py on real MTBench answer artifacts with the configured local judge before reporting scores.",
        ),
    ]


def render_markdown(title: str, rows: Sequence[RequirementRow]) -> str:
    lines = [
        f"# {title}",
        "",
        "| id | status | requirement | evidence | remaining |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        evidence = "<br>".join(row.evidence)
        lines.append(f"| {row.id} | {row.status} | {row.requirement} | {evidence} | {row.remaining} |")
    lines.append("")
    return "\n".join(lines)


def write_report(output_dir: Path, stem: str, title: str, rows: Sequence[RequirementRow]) -> None:
    payload = _report_payload(title, rows)
    (output_dir / f"{stem}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / f"{stem}.md").write_text(render_markdown(title, rows), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write implementation/protocol/status audit reports.")
    parser.add_argument("--output-dir", default="reports/audit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_report(output_dir, "method_implementation_audit", "Method Implementation Audit", method_rows())
    write_report(output_dir, "experiment_protocol_audit", "Experiment Protocol Audit", protocol_rows())
    write_report(output_dir, "status_matrix", "Status Matrix", status_rows())
    print(f"[audit_status] wrote reports to {output_dir}")


if __name__ == "__main__":
    main()
